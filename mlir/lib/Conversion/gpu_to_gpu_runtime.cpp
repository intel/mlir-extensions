// Copyright 2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir-extensions/Conversion/gpu_to_gpu_runtime.hpp"

#include "mlir-extensions/Dialect/gpu_runtime/IR/gpu_runtime_ops.hpp"
#include "mlir-extensions/Dialect/plier_util/dialect.hpp"

#include <mlir/Analysis/BufferViewFlowAnalysis.h>
#include <mlir/Conversion/ArithmeticToSPIRV/ArithmeticToSPIRV.h>
#include <mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h>
#include <mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h>
#include <mlir/Conversion/MathToSPIRV/MathToSPIRV.h>
#include <mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/ParallelLoopMapper.h>
#include <mlir/Dialect/GPU/Passes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Target/SPIRV/Serialization.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/SmallBitVector.h>

namespace {
struct ParallelLoopGPUMappingPass
    : public mlir::PassWrapper<ParallelLoopGPUMappingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto &region = func.getBody();
    if (region.empty())
      return;

    if (!llvm::hasSingleElement(region)) {
      func.emitError("Only strucutred control flow is supported");
      signalPassFailure();
      return;
    }

    auto getProcessor = [](unsigned val) -> mlir::gpu::Processor {
      const mlir::gpu::Processor mapping[] = {
          mlir::gpu::Processor::BlockX,  mlir::gpu::Processor::BlockY,
          mlir::gpu::Processor::BlockZ,  mlir::gpu::Processor::ThreadX,
          mlir::gpu::Processor::ThreadY, mlir::gpu::Processor::ThreadZ,
      };
      if (val >= llvm::array_lengthof(mapping))
        return mlir::gpu::Processor::Sequential;

      return mapping[val];
    };

    mlir::OpBuilder builder(&getContext());
    auto identityMap = builder.getDimIdentityMap();
    llvm::SmallVector<mlir::gpu::ParallelLoopDimMapping> mapping;
    for (auto &op : llvm::make_early_inc_range(region.front())) {
      auto parallel = mlir::dyn_cast<mlir::scf::ParallelOp>(op);
      if (!parallel)
        continue;

      auto numLoops = parallel.getNumLoops();
      mapping.resize(numLoops);
      for (auto i : llvm::seq(0u, numLoops))
        mapping[i] = mlir::gpu::getParallelLoopDimMappingAttr(
            getProcessor(i), identityMap, identityMap);

      if (mlir::failed(mlir::gpu::setMappingAttr(parallel, mapping))) {
        signalPassFailure();
        return;
      }
    }
  }
};

struct InsertGPUAllocs
    : public mlir::PassWrapper<InsertGPUAllocs,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  InsertGPUAllocs() = default;
  InsertGPUAllocs(bool gpuDealloc) { useGpuDealloc = gpuDealloc; }
  InsertGPUAllocs(const InsertGPUAllocs &pass) : PassWrapper(pass) {}

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::arith::ArithmeticDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto &funcBody = func.getBody();
    if (funcBody.empty()) {
      return;
    } else if (!llvm::hasSingleElement(funcBody)) {
      func.emitError("Function must have exactly one block");
      signalPassFailure();
      return;
    }

    struct AccessType {
      bool hostRead = false;
      bool hostWrite = false;
      bool deviceRead = false;
      bool deviceWrite = false;
    };

    llvm::SmallMapVector<mlir::Operation *, AccessType, 8> gpuBufferAllocs;
    llvm::SmallMapVector<unsigned, AccessType, 8> gpuBufferParams;
    auto &aliases = getAnalysis<mlir::BufferViewFlowAnalysis>();

    auto getMemref = [](mlir::Operation *op)
        -> llvm::Optional<mlir::SmallVector<mlir::Value, 4>> {
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
        return {{load.memref()}};
      } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
        return {{store.memref()}};
      } else if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
        mlir::SmallVector<mlir::Value, 4> ret;
        for (auto arg : call.operands()) {
          if (arg.getType().isa<mlir::MemRefType>())
            ret.emplace_back(arg);
        }
        return std::move(ret);
      } else {
        op->emitError("Uhhandled mem op in gpu region");
        return llvm::None;
      }
    };

    auto scfDialect = getContext().getOrLoadDialect<mlir::scf::SCFDialect>();

    auto hasMemAccess = [](mlir::Operation *op) -> bool {
      if (auto memInterface =
              mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
        if (memInterface.hasEffect<mlir::MemoryEffects::Read>() ||
            memInterface.hasEffect<mlir::MemoryEffects::Write>())
          return true;
      }
      if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
        for (auto arg : call.operands()) {
          if (arg.getType().isa<mlir::MemRefType>())
            return true;
        }
      }
      return false;
    };

    auto gpuAccessibleArg = [&]() -> llvm::SmallVector<bool> {
      auto gpuAttr = func->getAttr(gpu_runtime::getGpuAccessibleAttrName())
                         .dyn_cast_or_null<mlir::ArrayAttr>();
      if (!gpuAttr)
        return {};

      auto range = gpuAttr.getAsValueRange<mlir::BoolAttr>();
      return {range.begin(), range.end()};
    }();

    auto isGpuAccessibleArg = [&](unsigned i) {
      if (gpuAccessibleArg.empty())
        return false;

      assert(i < gpuAccessibleArg.size());
      return gpuAccessibleArg[i];
    };

    if (func.walk([&](mlir::Operation *op) {
              if (!op->getParentOfType<mlir::gpu::LaunchOp>())
                return mlir::WalkResult::advance();

              if (!hasMemAccess(op))
                return mlir::WalkResult::advance();

              auto memref = getMemref(op);
              if (!memref)
                return mlir::WalkResult::interrupt();

              for (auto mem : *memref) {
                while (auto parentView =
                           mem.getDefiningOp<mlir::ViewLikeOpInterface>())
                  mem = parentView.getViewSource();

                for (auto alias : aliases.resolve(mem)) {
                  auto op = alias.getDefiningOp();
                  if (op) {
                    if (op->getDialect() == scfDialect ||
                        mlir::isa<mlir::ViewLikeOpInterface>(op))
                      continue;
                    if (mlir::isa<mlir::memref::AllocOp,
                                  mlir::memref::GetGlobalOp>(op)) {
                      gpuBufferAllocs.insert({op, {}});
                    } else if (mlir::isa<mlir::func::CallOp>(op)) {
                      // Ignore
                    } else {
                      op->emitError("Unhandled memref producer");
                      return mlir::WalkResult::interrupt();
                    }

                  } else {
                    auto block = alias.getParentBlock();
                    auto blockArgs = block->getArguments();
                    auto it = llvm::find(blockArgs, alias);
                    assert(it != blockArgs.end());
                    auto index = static_cast<unsigned>(it - blockArgs.begin());
                    if (!isGpuAccessibleArg(index))
                      gpuBufferParams.insert({index, {}});
                  }
                }
              }

              return mlir::WalkResult::advance();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }

    auto getAccessType = [&](mlir::Value memref) {
      AccessType ret;
      for (auto mem : aliases.resolve(memref)) {
        for (auto user : mem.getUsers()) {
          if (mlir::isa<mlir::func::ReturnOp>(user)) {
            ret.hostRead = true;
            ret.hostWrite = true;
            continue;
          }

          if (auto copy = mlir::dyn_cast<mlir::memref::CopyOp>(user)) {
            if (copy.source() == mem)
              ret.hostRead = true;

            if (copy.target() == mem)
              ret.hostWrite = true;

            continue;
          }

          if (auto memInterface =
                  mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            bool onDevice = user->getParentOfType<mlir::gpu::LaunchOp>();
            if (memInterface.hasEffect<mlir::MemoryEffects::Read>())
              (onDevice ? ret.deviceRead : ret.hostRead) = true;

            if (memInterface.hasEffect<mlir::MemoryEffects::Write>())
              (onDevice ? ret.deviceWrite : ret.hostWrite) = true;

            continue;
          }
          if (mlir::isa<mlir::func::CallOp>(user)) {
            bool onDevice = user->getParentOfType<mlir::gpu::LaunchOp>();
            (onDevice ? ret.deviceRead : ret.hostRead) = true;
            (onDevice ? ret.deviceWrite : ret.hostWrite) = true;
            continue;
          }
        }
      }
      return ret;
    };

    for (auto &it : gpuBufferAllocs) {
      auto alloc = mlir::cast<mlir::memref::AllocOp>(it.first);
      if (alloc)
        it.second = getAccessType(alloc);
      else {
        auto memrefGlobal = mlir::cast<mlir::memref::GetGlobalOp>(it.first);
        it.second = getAccessType(memrefGlobal);
      }
    }

    auto &block = funcBody.front();
    for (auto &it : gpuBufferParams) {
      auto param = block.getArgument(it.first);
      it.second = getAccessType(param);

      it.second.hostRead = true;
      it.second.hostWrite = true;
    }

    mlir::OpBuilder builder(func);
    for (auto it : gpuBufferAllocs) {
      auto alloc = mlir::cast<mlir::memref::AllocOp>(it.first);
      auto access = it.second;
      auto loc = alloc.getLoc();
      builder.setInsertionPoint(alloc);
      auto gpuAlloc = builder.create<mlir::gpu::AllocOp>(
          loc, alloc.getType(), /*asyncToken*/ nullptr,
          /*asyncDependencies*/ llvm::None, alloc.dynamicSizes(),
          alloc.symbolOperands());
      alloc->replaceAllUsesWith(gpuAlloc);
      alloc.erase();
      if (access.hostRead || access.hostWrite)
        gpuAlloc->setAttr(gpu_runtime::getAllocSharedAttrName(),
                          builder.getUnitAttr());
    }

    auto term = block.getTerminator();
    assert(term);

    llvm::SmallVector<mlir::Value> dims;
    llvm::SmallPtrSet<mlir::Operation *, 8> filter;
    for (auto it : gpuBufferParams) {
      auto param = block.getArgument(it.first);
      auto access = it.second;
      auto loc = param.getLoc();
      builder.setInsertionPointToStart(&block);
      auto memrefType = param.getType().cast<mlir::MemRefType>();
      auto rank = static_cast<unsigned>(memrefType.getRank());
      filter.clear();
      dims.clear();
      for (auto i : llvm::seq(0u, rank)) {
        if (memrefType.isDynamicDim(i)) {
          auto op = builder.create<mlir::memref::DimOp>(loc, param, i);
          dims.push_back(op);
          filter.insert(op);
        }
      }
      auto allocType = mlir::MemRefType::get(
          memrefType.getShape(), memrefType.getElementType(),
          mlir::MemRefLayoutAttrInterface{}, memrefType.getMemorySpace());
      auto gpuAlloc = builder.create<mlir::gpu::AllocOp>(
          loc, allocType, /*asyncToken*/ nullptr,
          /*asyncDependencies*/ llvm::None, dims,
          /*symbolOperands*/ llvm::None);
      auto allocResult = gpuAlloc.getResult(0);

      if (access.hostRead || access.hostWrite)
        gpuAlloc->setAttr(gpu_runtime::getAllocSharedAttrName(),
                          builder.getUnitAttr());

      if (access.hostWrite && access.deviceRead) {
        auto copy =
            builder.create<mlir::memref::CopyOp>(loc, param, allocResult);
        filter.insert(copy);
      }

      if (allocType != memrefType)
        allocResult =
            builder.create<mlir::memref::CastOp>(loc, memrefType, allocResult);

      param.replaceAllUsesExcept(allocResult, filter);
      builder.setInsertionPoint(term);
      if (access.hostRead && access.deviceWrite)
        builder.create<mlir::memref::CopyOp>(loc, allocResult, param);

      if (useGpuDealloc)
        builder.create<mlir::gpu::DeallocOp>(loc, llvm::None, allocResult);
      else
        builder.create<mlir::memref::DeallocOp>(loc, allocResult);
    }
  }

  Option<bool> useGpuDealloc{
      *this, "use-gpu-dealloc",
      llvm::cl::desc("use gpu.dealloc for gpu allocated memrefs, "
                     "memref.dealloc will be used otherwise"),
      llvm::cl::init(true)};
};

static void setInsertionPointToStart(mlir::OpBuilder &builder,
                                     mlir::Value val) {
  if (auto parentOp = val.getDefiningOp()) {
    builder.setInsertionPointAfter(parentOp);
  } else {
    builder.setInsertionPointToStart(val.getParentBlock());
  }
}

static mlir::Value getFlatIndex(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value memref, mlir::ValueRange indices) {
  auto memrefType = memref.getType().cast<mlir::MemRefType>();
  auto rank = static_cast<unsigned>(memrefType.getRank());
  assert(indices.size() == rank);
  if (memrefType.getLayout().isIdentity()) {
    auto shape = memrefType.getShape();
    auto expr =
        mlir::makeCanonicalStridedLayoutExpr(shape, builder.getContext());
    llvm::SmallVector<mlir::Value> applyOperands;
    if (rank != 0) {
      applyOperands.reserve(rank * 2);
      applyOperands.assign(indices.begin(), indices.end());
      mlir::OpBuilder::InsertionGuard g(builder);
      setInsertionPointToStart(builder, memref);
      mlir::Value size;
      for (auto i : llvm::seq(0u, rank - 1)) {
        auto dimInd = rank - i - 1;
        auto dim =
            builder.createOrFold<mlir::memref::DimOp>(loc, memref, dimInd);
        if (i != 0) {
          size = builder.createOrFold<mlir::arith::MulIOp>(loc, size, dim);
        } else {
          size = dim;
        }

        applyOperands.emplace_back(size);
      }
    }
    auto affineMap = mlir::AffineMap::get(
        rank, static_cast<unsigned>(applyOperands.size()) - rank, expr);
    assert(affineMap.getNumDims() == indices.size());
    return builder.createOrFold<mlir::AffineApplyOp>(loc, affineMap,
                                                     applyOperands);
  } else {
    auto affineMap = memrefType.getLayout().getAffineMap();
    assert(affineMap.getNumDims() == indices.size());
    llvm::SmallVector<mlir::Value> applyOperands;
    if (rank != 0) {
      mlir::OpBuilder::InsertionGuard g(builder);
      setInsertionPointToStart(builder, memref);
      applyOperands.reserve(rank * 2 + 1);
      applyOperands.assign(indices.begin(), indices.end());

      auto numSymbols = affineMap.getNumSymbols();
      if (numSymbols > 0) {
        applyOperands.emplace_back(
            builder.createOrFold<plier::ExtractMemrefMetadataOp>(loc, memref));
        --numSymbols;
        assert(numSymbols <= rank);
        for (auto i : llvm::seq(0u, numSymbols)) {
          applyOperands.emplace_back(
              builder.createOrFold<plier::ExtractMemrefMetadataOp>(loc, memref,
                                                                   i));
        }
      }
    }
    return builder.createOrFold<mlir::AffineApplyOp>(loc, affineMap,
                                                     applyOperands);
  }
}

static mlir::Value getFlatIndex(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value memref,
                                llvm::ArrayRef<mlir::OpFoldResult> indices) {
  llvm::SmallVector<mlir::Value> vals(indices.size());
  for (auto it : llvm::enumerate(indices)) {
    auto i = it.index();
    auto val = it.value();
    if (auto attr = val.dyn_cast<mlir::Attribute>()) {
      auto ind = attr.cast<mlir::IntegerAttr>().getValue().getSExtValue();
      vals[i] = builder.create<mlir::arith::ConstantIndexOp>(loc, ind);
    } else {
      vals[i] = val.get<mlir::Value>();
    }
  }
  return getFlatIndex(builder, loc, memref, vals);
}

static mlir::Value getFlatMemref(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value memref) {
  auto memrefType = memref.getType().cast<mlir::MemRefType>();
  auto resultType = mlir::MemRefType::get(mlir::ShapedType::kDynamicSize,
                                          memrefType.getElementType());
  mlir::OpBuilder::InsertionGuard g(builder);
  setInsertionPointToStart(builder, memref);
  mlir::OpFoldResult offset = builder.getIndexAttr(0);
  mlir::OpFoldResult size =
      builder.createOrFold<plier::UndefOp>(loc, builder.getIndexType());
  mlir::OpFoldResult stride = builder.getIndexAttr(1);
  return builder.createOrFold<mlir::memref::ReinterpretCastOp>(
      loc, resultType, memref, offset, size, stride);
}

static bool needFlatten(mlir::Value val) {
  auto type = val.getType().cast<mlir::MemRefType>();
  return !type.getLayout().isIdentity() || (type.getRank() > 1);
}

struct FlattenLoad : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<mlir::gpu::LaunchOp>())
      return mlir::failure();

    auto memref = op.memref();
    if (!needFlatten(memref))
      return mlir::failure();

    auto loc = op.getLoc();
    auto flatIndex = getFlatIndex(rewriter, loc, memref, op.indices());
    auto flatMemref = getFlatMemref(rewriter, loc, memref);
    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, flatMemref,
                                                      flatIndex);
    return mlir::success();
  }
};

struct FlattenStore : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<mlir::gpu::LaunchOp>())
      return mlir::failure();

    auto memref = op.memref();
    if (!needFlatten(memref))
      return mlir::failure();

    auto loc = op.getLoc();
    auto flatIndex = getFlatIndex(rewriter, loc, memref, op.indices());
    auto flatMemref = getFlatMemref(rewriter, loc, memref);
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, op.value(),
                                                       flatMemref, flatIndex);
    return mlir::success();
  }
};

struct FlattenSubview : public mlir::OpRewritePattern<mlir::memref::SubViewOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::SubViewOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<mlir::gpu::LaunchOp>())
      return mlir::failure();

    auto memref = op.source();
    if (!needFlatten(memref))
      return mlir::failure();

    auto offsets = op.getMixedOffsets();
    auto sizes = op.getMixedSizes();
    auto strides = op.getMixedStrides();

    auto srcType = memref.getType().cast<mlir::MemRefType>();
    auto dstType = mlir::memref::SubViewOp::inferResultType(srcType, offsets,
                                                            sizes, strides)
                       .cast<mlir::MemRefType>();

    int64_t resultOffset; // TODO: remove
    llvm::SmallVector<int64_t, 4> resultStrides;
    if (mlir::failed(
            mlir::getStridesAndOffset(dstType, resultStrides, resultOffset)))
      return mlir::failure();

    auto loc = op.getLoc();
    mlir::OpFoldResult flatIndex = getFlatIndex(rewriter, loc, memref, offsets);
    mlir::OpFoldResult flatSize =
        rewriter.create<plier::UndefOp>(loc, rewriter.getIndexType())
            .getResult();
    mlir::OpFoldResult flatStride = rewriter.getIndexAttr(1);
    auto flatMemref = getFlatMemref(rewriter, loc, memref);
    auto flatMemrefType = flatMemref.getType().cast<mlir::MemRefType>();
    assert(flatMemrefType.getLayout().isIdentity());
    auto flatSubview = rewriter.createOrFold<mlir::memref::SubViewOp>(
        loc, flatMemref, flatIndex, flatSize, flatStride);
    auto dstFlatType = flatSubview.getType();
    if (dstFlatType != flatMemrefType)
      flatSubview = rewriter.createOrFold<mlir::memref::CastOp>(
          loc, dstFlatType, flatSubview);

    auto offset = rewriter.getIndexAttr(0);

    for (auto i : llvm::seq<size_t>(0, strides.size())) {
      if (mlir::ShapedType::isDynamicStrideOrOffset(resultStrides[i])) {
        auto stride = strides[i];
        if (auto c = stride.dyn_cast<mlir::Attribute>()) {
          auto val = c.dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();
          stride = rewriter.create<mlir::arith::ConstantIndexOp>(loc, val)
                       .getResult();
        }

        auto origStride = [&]() {
          mlir::OpBuilder::InsertionGuard g(rewriter);
          setInsertionPointToStart(rewriter, memref);
          return rewriter.createOrFold<plier::ExtractMemrefMetadataOp>(
              loc, memref, i);
        }();
        auto newStride = rewriter.createOrFold<mlir::arith::MulIOp>(
            loc, stride.get<mlir::Value>(), origStride);
        strides[i] = newStride;
      }
    }

    auto resultType = op.getType().cast<mlir::MemRefType>();
    auto srcRank = static_cast<unsigned>(srcType.getRank());
    auto resultRank = static_cast<unsigned>(resultType.getRank());
    mlir::Value result;
    if (srcRank == resultRank) {
      result = rewriter.createOrFold<mlir::memref::ReinterpretCastOp>(
          loc, resultType, flatSubview, offset, sizes, strides);
    } else {
      assert(resultRank < srcRank);
      llvm::SmallVector<mlir::OpFoldResult> filteredSizes;
      llvm::SmallVector<mlir::OpFoldResult> filteredStrides;
      filteredSizes.reserve(resultRank);
      filteredStrides.reserve(resultRank);

      auto droppedDims = op.getDroppedDims();
      for (auto i : llvm::seq(0u, srcRank)) {
        if (!droppedDims[i]) {
          filteredSizes.emplace_back(sizes[i]);
          filteredStrides.emplace_back(strides[i]);
        }
      }
      result = rewriter.createOrFold<mlir::memref::ReinterpretCastOp>(
          loc, resultType, flatSubview, offset, filteredSizes, filteredStrides);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct UnstrideMemrefsPass
    : public mlir::PassWrapper<UnstrideMemrefsPass, mlir::OperationPass<void>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<plier::PlierUtilDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<FlattenLoad, FlattenStore, FlattenSubview>(ctx);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

static llvm::Optional<mlir::Value> getGpuStream(mlir::OpBuilder &builder,
                                                mlir::Operation *op) {
  assert(op);
  auto func = op->getParentOfType<mlir::func::FuncOp>();
  if (!func)
    return {};

  if (!llvm::hasSingleElement(func.getBody()))
    return {};

  auto &block = func.getBody().front();
  auto ops = block.getOps<gpu_runtime::CreateGpuStreamOp>();
  if (!ops.empty())
    return (*ops.begin()).getResult();

  mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(&block);
  auto loc = builder.getUnknownLoc();
  auto stream = builder.create<gpu_runtime::CreateGpuStreamOp>(loc).getResult();
  builder.setInsertionPoint(block.getTerminator());
  builder.create<gpu_runtime::DestroyGpuStreamOp>(loc, stream);
  return stream;
}

class ConvertSubviewOp
    : public mlir::OpConversionPattern<mlir::memref::SubViewOp> {
public:
  using mlir::OpConversionPattern<mlir::memref::SubViewOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::SubViewOp op,
                  mlir::memref::SubViewOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dstType = op.getType().cast<mlir::MemRefType>();
    if (!dstType.hasRank() || dstType.getRank() != 1)
      return mlir::failure();

    auto intType = getTypeConverter()->convertType(rewriter.getIndexType());
    if (!intType)
      return mlir::failure();

    auto loc = op.getLoc();
    auto getValue = [&](mlir::OpFoldResult src) -> mlir::Value {
      if (auto val = src.dyn_cast<mlir::Value>())
        return val;

      auto attr = src.get<mlir::Attribute>();
      return rewriter.create<mlir::spirv::ConstantOp>(loc, intType, attr);
    };

    auto offset =
        getValue(op.isDynamicOffset(0)
                     ? mlir::OpFoldResult(adaptor.offsets()[0])
                     : mlir::OpFoldResult(adaptor.static_offsets()[0]));
    auto stride =
        getValue(op.isDynamicStride(0)
                     ? mlir::OpFoldResult(adaptor.strides()[0])
                     : mlir::OpFoldResult(adaptor.static_strides()[0]));
    auto finalOffset = rewriter.createOrFold<mlir::spirv::IMulOp>(
        loc, intType, offset, stride);

    auto ptr = rewriter
                   .create<mlir::spirv::InBoundsPtrAccessChainOp>(
                       loc, adaptor.source(), finalOffset, llvm::None)
                   .getResult();

    rewriter.replaceOp(op, ptr);
    return mlir::success();
  }
};

template <typename T>
class ConvertCastOp : public mlir::OpConversionPattern<T> {
public:
  using mlir::OpConversionPattern<T>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.source());
    return mlir::success();
  }
};

class ConvertLoadOp : public mlir::OpConversionPattern<mlir::memref::LoadOp> {
public:
  using mlir::OpConversionPattern<mlir::memref::LoadOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::memref::LoadOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memrefType = op.memref().getType().cast<mlir::MemRefType>();
    if (memrefType.getRank() == 0) {
      auto memoryAccess = mlir::spirv::MemoryAccessAttr::get(
          op.getContext(), mlir::spirv::MemoryAccess::Aligned);
      auto alignment =
          rewriter.getI32IntegerAttr(memrefType.getElementTypeBitWidth() / 8);
      rewriter.replaceOpWithNewOp<mlir::spirv::LoadOp>(op, adaptor.memref(),
                                                       memoryAccess, alignment);
      return mlir::success();
    } else if (memrefType.hasRank() && memrefType.getRank() == 1) {
      auto loc = op.getLoc();
      auto ptr = rewriter.create<mlir::spirv::InBoundsPtrAccessChainOp>(
          loc, adaptor.memref(), adaptor.indices().front(), llvm::None);

      auto memoryAccess = mlir::spirv::MemoryAccessAttr::get(
          op.getContext(), mlir::spirv::MemoryAccess::Aligned);
      auto alignment =
          rewriter.getI32IntegerAttr(memrefType.getElementTypeBitWidth() / 8);
      rewriter.replaceOpWithNewOp<mlir::spirv::LoadOp>(op, ptr, memoryAccess,
                                                       alignment);

      return mlir::success();
    } else {
      return mlir::failure();
    }
  }
};

class ConvertStoreOp : public mlir::OpConversionPattern<mlir::memref::StoreOp> {
public:
  using mlir::OpConversionPattern<mlir::memref::StoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::memref::StoreOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memrefType = op.memref().getType().cast<mlir::MemRefType>();
    if (!memrefType.hasRank() || memrefType.getRank() != 1)
      return mlir::failure();

    auto loc = op.getLoc();
    auto ptr = rewriter.create<mlir::spirv::InBoundsPtrAccessChainOp>(
        loc, adaptor.memref(), adaptor.indices().front(), llvm::None);

    auto memoryAccess = mlir::spirv::MemoryAccessAttr::get(
        op.getContext(), mlir::spirv::MemoryAccess::Aligned);
    auto alignment =
        rewriter.getI32IntegerAttr(memrefType.getElementTypeBitWidth() / 8);
    rewriter.replaceOpWithNewOp<mlir::spirv::StoreOp>(op, ptr, adaptor.value(),
                                                      memoryAccess, alignment);

    return mlir::success();
  }
};

template <typename Op>
static mlir::Value lowerIntAtomic(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value ptr, mlir::Value val) {
  return builder.create<Op>(loc, ptr, mlir::spirv::Scope::Device,
                            mlir::spirv::MemorySemantics::None, val);
}

static mlir::Value lowerFloatAddAtomic(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value ptr,
                                       mlir::Value val) {
  return builder.create<mlir::spirv::AtomicFAddEXTOp>(
      loc, val.getType(), ptr, mlir::spirv::Scope::Device,
      mlir::spirv::MemorySemantics::None, val);
}

static mlir::Value lowerFloatSubAtomic(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value ptr,
                                       mlir::Value val) {
  auto neg = builder.create<mlir::spirv::FNegateOp>(loc, val).getResult();
  return builder.create<mlir::spirv::AtomicFAddEXTOp>(
      loc, neg.getType(), ptr, mlir::spirv::Scope::Device,
      mlir::spirv::MemorySemantics::None, neg);
}

class ConvertAtomicOps : public mlir::OpConversionPattern<mlir::func::CallOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op, mlir::func::CallOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.operands();
    if (operands.size() != 2)
      return mlir::failure();

    if (op.getNumResults() != 1)
      return mlir::failure();

    auto ptr = operands[0];
    auto ptrType = ptr.getType().dyn_cast<mlir::spirv::PointerType>();
    if (!ptrType)
      return mlir::failure();

    auto val = operands[1];
    auto valType = val.getType();
    if (ptrType.getPointeeType() != valType)
      return mlir::failure();

    bool isInt;
    if (valType.isSignlessInteger())
      isInt = true;
    else if (valType.isa<mlir::FloatType>())
      isInt = false;
    else
      return mlir::failure();

    auto funcName = op.getCallee();

    using func_t = mlir::Value (*)(mlir::OpBuilder &, mlir::Location,
                                   mlir::Value, mlir::Value);

    struct Desc {
      mlir::StringRef name;
      func_t intOp;
      func_t floatOp;
    };

    const Desc handlers[] = {
        {"atomic_add", &lowerIntAtomic<mlir::spirv::AtomicIAddOp>,
         &lowerFloatAddAtomic},
        {"atomic_sub", &lowerIntAtomic<mlir::spirv::AtomicISubOp>,
         &lowerFloatSubAtomic},
    };

    auto handler = [&]() -> func_t {
      for (auto &h : handlers) {
        if (funcName.consume_front(h.name))
          return (isInt ? h.intOp : h.floatOp);
      }
      return nullptr;
    }();

    if (handler) {
      auto res = handler(rewriter, op.getLoc(), ptr, val);
      if (res) {
        rewriter.replaceOp(op, res);
        return mlir::success();
      }
    }

    return mlir::failure();
  }
};

class ConvertBarrierOp
    : public mlir::OpConversionPattern<gpu_runtime::GPUBarrierOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUBarrierOp op,
                  gpu_runtime::GPUBarrierOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto scope = mlir::spirv::Scope::Workgroup;
    mlir::spirv::MemorySemantics semantics;
    if (adaptor.flags() == gpu_runtime::FenceFlags::global) {
      semantics = mlir::spirv::MemorySemantics::SequentiallyConsistent |
                  mlir::spirv::MemorySemantics::CrossWorkgroupMemory;
    } else if (adaptor.flags() == gpu_runtime::FenceFlags::local) {
      semantics = mlir::spirv::MemorySemantics::SequentiallyConsistent |
                  mlir::spirv::MemorySemantics::WorkgroupMemory;
    } else {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<mlir::spirv::ControlBarrierOp>(op, scope, scope,
                                                               semantics);
    return mlir::success();
  }
};

class ConvertMemFenceOp
    : public mlir::OpConversionPattern<gpu_runtime::GPUMemFenceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUMemFenceOp op,
                  gpu_runtime::GPUMemFenceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto scope = mlir::spirv::Scope::Workgroup;
    mlir::spirv::MemorySemantics semantics;
    if (adaptor.flags() == gpu_runtime::FenceFlags::global) {
      semantics = mlir::spirv::MemorySemantics::SequentiallyConsistent |
                  mlir::spirv::MemorySemantics::CrossWorkgroupMemory;
    } else if (adaptor.flags() == gpu_runtime::FenceFlags::local) {
      semantics = mlir::spirv::MemorySemantics::SequentiallyConsistent |
                  mlir::spirv::MemorySemantics::WorkgroupMemory;
    } else {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<mlir::spirv::MemoryBarrierOp>(op, scope,
                                                              semantics);
    return mlir::success();
  }
};

static llvm::Optional<mlir::spirv::StorageClass>
convertStorageClass(mlir::Attribute src) {
  auto attr = src.dyn_cast_or_null<gpu_runtime::StorageClassAttr>();
  if (!attr)
    return llvm::None;

  auto sc = attr.getValue();
  if (sc == gpu_runtime::StorageClass::local)
    return mlir::spirv::StorageClass::Workgroup;

  return llvm::None;
}

static mlir::spirv::StorageClass
convertStorageClass(mlir::Attribute src, mlir::spirv::StorageClass def) {
  auto ret = convertStorageClass(src);
  if (ret)
    return *ret;

  return def;
}

class ConvertGlobalOp
    : public mlir::OpConversionPattern<mlir::memref::GlobalOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::GlobalOp op,
                  mlir::memref::GlobalOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memrefType = op.type();
    if (!memrefType.hasStaticShape())
      return mlir::failure();

    auto storageClass = convertStorageClass(memrefType.getMemorySpace());
    if (!storageClass)
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter);

    auto elemType = converter->convertType(memrefType.getElementType());
    if (!elemType)
      return mlir::failure();

    auto elemCount = memrefType.getNumElements();
    auto newType = mlir::spirv::ArrayType::get(elemType, elemCount);
    auto ptrType = mlir::spirv::PointerType::get(newType, *storageClass);

    rewriter.replaceOpWithNewOp<mlir::spirv::GlobalVariableOp>(
        op, ptrType, adaptor.sym_name());
    return mlir::success();
  }
};

class ConvertGetGlobalOp
    : public mlir::OpConversionPattern<mlir::memref::GetGlobalOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::GetGlobalOp op,
                  mlir::memref::GetGlobalOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memrefType = op.getType().dyn_cast<mlir::MemRefType>();
    if (!memrefType)
      return mlir::failure();

    auto storageClass = convertStorageClass(memrefType.getMemorySpace());
    if (!storageClass)
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter);
    auto resType = converter->convertType(memrefType);
    if (!resType)
      return mlir::failure();

    auto elemType = converter->convertType(memrefType.getElementType());
    if (!elemType)
      return mlir::failure();

    auto elemCount = memrefType.getNumElements();
    auto newType = mlir::spirv::ArrayType::get(elemType, elemCount);
    auto ptrType = mlir::spirv::PointerType::get(newType, *storageClass);

    auto loc = op->getLoc();
    mlir::Value res =
        rewriter.create<mlir::spirv::AddressOfOp>(loc, ptrType, adaptor.name());
    if (res.getType() != resType)
      res = rewriter.create<mlir::spirv::BitcastOp>(loc, resType, res);

    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};

// TODO: something better
class ConvertFunc : public mlir::OpConversionPattern<mlir::func::FuncOp> {
public:
  using mlir::OpConversionPattern<mlir::func::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp op,
                  mlir::func::FuncOp::Adaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.getBody().empty())
      return mlir::failure();

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ConvertAssert : public mlir::OpConversionPattern<mlir::cf::AssertOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::AssertOp op, mlir::cf::AssertOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::spirv::AssumeTrueKHROp>(op,
                                                              adaptor.getArg());
    return mlir::success();
  }
};

class ConvertUndef : public mlir::OpConversionPattern<plier::UndefOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::UndefOp op, plier::UndefOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter);

    auto resType = converter->convertType(op.getType());
    if (!resType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::spirv::UndefOp>(op, resType);
    return mlir::success();
  }
};

struct GPUToSpirvPass
    : public mlir::PassWrapper<GPUToSpirvPass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect>();
    registry.insert<mlir::spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    auto module = getOperation();

    llvm::SmallVector<mlir::Operation *, 1> kernelModules;
    mlir::OpBuilder builder(context);
    module.walk([&builder, &kernelModules](mlir::gpu::GPUModuleOp moduleOp) {
      // For each kernel module (should be only 1 for now, but that is not a
      // requirement here), clone the module for conversion because the
      // gpu.launch function still needs the kernel module.
      builder.setInsertionPoint(moduleOp.getOperation());
      kernelModules.push_back(builder.clone(*moduleOp.getOperation()));
    });

    auto targetAttr = mlir::spirv::lookupTargetEnvOrDefault(module);
    auto target = mlir::SPIRVConversionTarget::get(targetAttr);

    mlir::SPIRVTypeConverter::Options options;
    options.use64bitIndex = true;

    mlir::SPIRVTypeConverter typeConverter(targetAttr, options);
    mlir::RewritePatternSet patterns(context);

    typeConverter.addConversion(
        [&typeConverter](mlir::MemRefType type) -> llvm::Optional<mlir::Type> {
          if (!type.hasRank() || !type.getElementType().isIntOrFloat())
            return mlir::Type(nullptr);

          auto elemType = typeConverter.convertType(type.getElementType());
          if (!elemType)
            return mlir::Type(nullptr);

          auto sc = convertStorageClass(
              type.getMemorySpace(), mlir::spirv::StorageClass::CrossWorkgroup);

          return mlir::spirv::PointerType::get(elemType, sc);
        });

    mlir::ScfToSPIRVContext scfToSpirvCtx;
    mlir::populateSCFToSPIRVPatterns(typeConverter, scfToSpirvCtx, patterns);
    mlir::populateGPUToSPIRVPatterns(typeConverter, patterns);
    mlir::populateFuncToSPIRVPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToSPIRVPatterns(typeConverter, patterns);
    mlir::arith::populateArithmeticToSPIRVPatterns(typeConverter, patterns);
    mlir::populateMathToSPIRVPatterns(typeConverter, patterns);

    patterns
        .insert<ConvertSubviewOp, ConvertCastOp<mlir::memref::CastOp>,
                ConvertCastOp<mlir::memref::ReinterpretCastOp>, ConvertLoadOp,
                ConvertStoreOp, ConvertAtomicOps, ConvertFunc, ConvertAssert,
                ConvertBarrierOp, ConvertMemFenceOp, ConvertUndef,
                ConvertGlobalOp, ConvertGetGlobalOp>(typeConverter, context);

    if (failed(
            applyFullConversion(kernelModules, *target, std::move(patterns))))
      return signalPassFailure();
  }
};

template <typename Op, typename F>
static mlir::LogicalResult createGpuKernelLoad(mlir::PatternRewriter &builder,
                                               Op &&op, F &&func) {
  auto mod = op->template getParentOfType<mlir::ModuleOp>();
  if (!mod)
    return mlir::failure();

  auto gpuMod = mod.template lookupSymbol<mlir::gpu::GPUModuleOp>(
      op.getKernelModuleName());
  if (!gpuMod)
    return mlir::failure();

  auto gpuKernel =
      gpuMod.template lookupSymbol<mlir::gpu::GPUFuncOp>(op.getKernelName());
  if (!gpuKernel)
    return mlir::failure();

  auto stream = getGpuStream(builder, op);
  if (!stream)
    return mlir::failure();

  auto loc = op.getLoc();
  auto module =
      builder.create<gpu_runtime::LoadGpuModuleOp>(loc, *stream, gpuMod);
  auto kernel =
      builder.create<gpu_runtime::GetGpuKernelOp>(loc, module, gpuKernel);
  auto newOp = func(builder, loc, *stream, kernel);
  builder.replaceOp(op, newOp.getResults());
  return mlir::success();
}

struct ExpandLaunchOp : public mlir::OpRewritePattern<mlir::gpu::LaunchFuncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::LaunchFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    return createGpuKernelLoad(
        rewriter, op,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value stream,
            mlir::Value kernel) {
          return builder.create<gpu_runtime::LaunchGpuKernelOp>(
              loc, stream, kernel, op.getGridSizeOperandValues(),
              op.getBlockSizeOperandValues(), op.operands());
        });
  }
};

struct ExpandAllocOp : public mlir::OpRewritePattern<mlir::gpu::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::AllocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto stream = getGpuStream(rewriter, op);
    if (!stream)
      return mlir::failure();

    auto sharedAttrName =
        rewriter.getStringAttr(gpu_runtime::getAllocSharedAttrName());
    auto shared = op->hasAttr(sharedAttrName);

    mlir::Type token = op.asyncToken() ? op.asyncToken().getType() : nullptr;
    auto res = rewriter.replaceOpWithNewOp<gpu_runtime::GPUAllocOp>(
        op, op.getType(), token, op.asyncDependencies(), *stream,
        op.dynamicSizes(), op.symbolOperands());

    if (shared)
      res->setAttr(sharedAttrName, rewriter.getUnitAttr());

    return mlir::success();
  }
};

struct ExpandDeallocOp : public mlir::OpRewritePattern<mlir::gpu::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::DeallocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto stream = getGpuStream(rewriter, op);
    if (!stream)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<gpu_runtime::GPUDeallocOp>(
        op, op.asyncDependencies(), op.memref(), *stream);

    return mlir::success();
  }
};

struct ExpandSuggestBlockSizeOp
    : public mlir::OpRewritePattern<gpu_runtime::GPUSuggestBlockSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUSuggestBlockSizeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.kernel())
      return mlir::failure();

    assert(op.kernelRef());
    return createGpuKernelLoad(
        rewriter, op,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value stream,
            mlir::Value kernel) {
          return builder.create<gpu_runtime::GPUSuggestBlockSizeOp>(
              loc, stream, kernel, op.gridSize());
        });
  }
};

struct AbiAttrsPass
    : public mlir::PassWrapper<AbiAttrsPass,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto gpuModule = getOperation();
    auto *context = &getContext();
    auto attrName =
        mlir::StringAttr::get(context, mlir::spirv::getEntryPointABIAttrName());
    auto abi = mlir::spirv::getEntryPointABIAttr(llvm::None, context);
    for (auto gpuFunc : gpuModule.getOps<mlir::gpu::GPUFuncOp>()) {
      if (!mlir::gpu::GPUDialect::isKernel(gpuFunc) ||
          gpuFunc->getAttr(attrName))
        continue;

      gpuFunc->setAttr(attrName, abi);
    }
  }
};

struct SetSPIRVCapabilitiesPass
    : public mlir::PassWrapper<SetSPIRVCapabilitiesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    namespace spirv = mlir::spirv;
    auto context = &getContext();
    spirv::Capability caps[] = {
        // clang-format off
        spirv::Capability::Addresses,
        spirv::Capability::Float16Buffer,
        spirv::Capability::Int64,
        spirv::Capability::Int16,
        spirv::Capability::Int8,
        spirv::Capability::Kernel,
        spirv::Capability::Linkage,
        spirv::Capability::Vector16,
        spirv::Capability::GenericPointer,
        spirv::Capability::Groups,
        spirv::Capability::Float16,
        spirv::Capability::Float64,
        spirv::Capability::AtomicFloat32AddEXT,
        spirv::Capability::ExpectAssumeKHR,
        // clang-format on
    };
    spirv::Extension exts[] = {
        spirv::Extension::SPV_EXT_shader_atomic_float_add,
        spirv::Extension::SPV_KHR_expect_assume};
    auto triple =
        spirv::VerCapExtAttr::get(spirv::Version::V_1_0, caps, exts, context);
    auto attr = spirv::TargetEnvAttr::get(
        triple, spirv::Vendor::Unknown, spirv::DeviceType::Unknown,
        spirv::TargetEnvAttr::kUnknownDeviceID,
        spirv::getDefaultResourceLimits(context));
    auto module = getOperation();
    module->setAttr(spirv::getTargetEnvAttrName(), attr);
  }
};

struct SerializeSPIRVPass
    : public mlir::PassWrapper<SerializeSPIRVPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto mod = getOperation();

    namespace gpu = mlir::gpu;
    namespace spirv = mlir::spirv;
    llvm::SmallVector<uint32_t, 0> spvBinary;
    for (auto gpuMod : mod.getOps<gpu::GPUModuleOp>()) {
      auto name = gpuMod.getName();
      auto isSameMod = [&](spirv::ModuleOp spvMod) -> bool {
        auto spvModName = spvMod.getName();
        return spvModName->consume_front("__spv__") && spvModName == name;
      };
      auto spvMods = mod.getOps<spirv::ModuleOp>();
      auto it = llvm::find_if(spvMods, isSameMod);
      if (it == spvMods.end()) {
        gpuMod.emitError() << "Unable to find corresponding SPIR-V module";
        signalPassFailure();
        return;
      }
      auto spvMod = *it;

      spvBinary.clear();
      if (mlir::failed(spirv::serialize(spvMod, spvBinary))) {
        spvMod.emitError() << "Failed to serialize SPIR-V module";
        signalPassFailure();
        return;
      }

      auto spvData =
          llvm::StringRef(reinterpret_cast<const char *>(spvBinary.data()),
                          spvBinary.size() * sizeof(uint32_t));
      auto spvAttr = mlir::StringAttr::get(&getContext(), spvData);
      gpuMod->setAttr(gpu::getDefaultGpuBinaryAnnotation(), spvAttr);
      spvMod->erase();
    }
  }
};

struct GPUExPass
    : public mlir::PassWrapper<GPUExPass, mlir::OperationPass<void>> {

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<gpu_runtime::GpuRuntimeDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<ExpandLaunchOp, ExpandAllocOp, ExpandDeallocOp,
                    ExpandSuggestBlockSizeOp>(ctx);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

} // namespace

// Expose the passes to the outside world
std::unique_ptr<mlir::Pass> gpu_runtime::createAbiAttrsPass() {
  return std::make_unique<AbiAttrsPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createSetSPIRVCapabilitiesPass() {
  return std::make_unique<SetSPIRVCapabilitiesPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createGPUToSpirvPass() {
  return std::make_unique<GPUToSpirvPass>();
}

std::unique_ptr<mlir::Pass>
gpu_runtime::createInsertGPUAllocsPass(bool useGpuDealloc) {
  return std::make_unique<InsertGPUAllocs>(useGpuDealloc);
}

std::unique_ptr<mlir::Pass> gpu_runtime::createUnstrideMemrefsPass() {
  return std::make_unique<UnstrideMemrefsPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createSerializeSPIRVPass() {
  return std::make_unique<SerializeSPIRVPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createGPUExPass() {
  return std::make_unique<GPUExPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createParallelLoopGPUMappingPass() {
  return std::make_unique<ParallelLoopGPUMappingPass>();
}
