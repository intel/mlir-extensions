// Copyright 2021 Intel Corporation
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

#include "pipelines/lower_to_gpu.hpp"

#include <llvm/Support/FormatVariadic.h>
#include <mlir/Analysis/BufferViewFlowAnalysis.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithmeticToSPIRV/ArithmeticToSPIRV.h>
#include <mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h>
#include <mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h>
#include <mlir/Conversion/GPUCommon/GPUCommonPass.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MathToSPIRV/MathToSPIRV.h>
#include <mlir/Conversion/SCFToGPU/SCFToGPUPass.h>
#include <mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/GPU/ParallelLoopMapper.h>
#include <mlir/Dialect/GPU/Passes.h>
#include <mlir/Dialect/GPU/Utils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/SPIRV/Serialization.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/SmallBitVector.h>

#include "base_pipeline.hpp"
#include "loop_utils.hpp"
#include "pipelines/lower_to_llvm.hpp"
#include "pipelines/plier_to_linalg.hpp"
#include "pipelines/plier_to_std.hpp"
#include "pipelines/pre_low_simplifications.hpp"
#include "py_linalg_resolver.hpp"

#include "mlir-extensions/compiler/pipeline_registry.hpp"
#include "mlir-extensions/dialect/plier/dialect.hpp"
#include "mlir-extensions/dialect/plier_util/dialect.hpp"
#include "mlir-extensions/transforms/call_lowering.hpp"
#include "mlir-extensions/transforms/cast_utils.hpp"
#include "mlir-extensions/transforms/common_opts.hpp"
#include "mlir-extensions/transforms/const_utils.hpp"
#include "mlir-extensions/transforms/func_utils.hpp"
#include "mlir-extensions/transforms/pipeline_utils.hpp"
#include "mlir-extensions/transforms/rewrite_wrapper.hpp"
#include "mlir-extensions/transforms/type_conversion.hpp"

namespace {
static void moveOpsIntoParallel(mlir::scf::ParallelOp outer, int depth = 0) {
  auto &outerBody = outer.getLoopBody().front();
  auto parallelIt = llvm::find_if(
      outerBody, [](auto &op) { return mlir::isa<mlir::scf::ParallelOp>(op); });
  if (outerBody.end() == parallelIt)
    return;

  auto parallelOp = mlir::cast<mlir::scf::ParallelOp>(*parallelIt);
  auto &parallelOpBody = parallelOp.getLoopBody().front();
  auto it = std::prev(parallelIt);
  auto begin = outerBody.begin();
  while (true) {
    bool first = (it == begin);
    auto &op = *it;
    auto isParallelOpOperand = [&](mlir::Operation &op) {
      auto operands = parallelOp->getOperands();
      for (auto r : op.getResults())
        if (llvm::is_contained(operands, r))
          return true;

      return false;
    };

    if (!mlir::MemoryEffectOpInterface::hasNoEffect(&op) ||
        isParallelOpOperand(op))
      break;

    if (first) {
      op.moveBefore(&parallelOpBody.front());
      break;
    }

    --it;
    op.moveBefore(&parallelOpBody.front());
  }
  depth += outer.getStep().size();
  if (depth >= 6)
    return;

  moveOpsIntoParallel(parallelOp, depth);
}

struct PrepareForGPUPass
    : public mlir::PassWrapper<PrepareForGPUPass,
                               mlir::OperationPass<mlir::FuncOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    for (auto &block : getOperation().getBody()) {
      for (auto &op : block) {
        if (auto parallel = mlir::dyn_cast<mlir::scf::ParallelOp>(op)) {
          moveOpsIntoParallel(parallel);
        }
      }
    }
  }
};

static mlir::LogicalResult
convertParallelToFor(mlir::scf::ParallelOp op,
                     mlir::PatternRewriter &rewriter) {
  auto lowerBounds = op.getLowerBound();
  auto upperBound = op.getUpperBound();
  auto steps = op.getStep();
  auto initVals = op.getInitVals();
  assert(!steps.empty());
  if (steps.size() > 1)
    return mlir::failure();

  auto &srcBlock = op.getLoopBody().front();
  assert(srcBlock.getNumArguments() == steps.size());

  auto buildFunc = [&](mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value index, mlir::ValueRange args) {
    llvm::SmallVector<mlir::Value> yieldArgs(initVals.size());
    mlir::BlockAndValueMapping mapping;
    mapping.map(srcBlock.getArgument(0), index);
    unsigned reduceIndex = 0;
    for (auto &bodyOp : srcBlock.without_terminator()) {
      if (auto reduce = mlir::dyn_cast<mlir::scf::ReduceOp>(bodyOp)) {
        auto &reduceBlock = reduce.getRegion().front();
        assert(reduceBlock.getNumArguments() == 2);
        mapping.map(reduceBlock.getArgument(0), args[reduceIndex]);
        mapping.map(reduceBlock.getArgument(1),
                    mapping.lookupOrDefault(reduce.getOperand()));
        for (auto &reduceOp : reduceBlock.without_terminator())
          builder.clone(reduceOp, mapping);

        auto yieldResult =
            mlir::cast<mlir::scf::ReduceReturnOp>(reduceBlock.getTerminator())
                .getResult();
        yieldArgs[reduceIndex] = mapping.lookupOrDefault(yieldResult);
        ++reduceIndex;
      } else {
        builder.clone(bodyOp, mapping);
      }
    }
    builder.create<mlir::scf::YieldOp>(loc, yieldArgs);
  };

  auto loc = op.getLoc();
  auto res = rewriter.create<mlir::scf::ForOp>(
      loc, lowerBounds.front(), upperBound.front(), steps.front(), initVals,
      buildFunc);
  rewriter.replaceOp(op, res.getResults());
  return mlir::success();
}

struct RemoveNestedParallel
    : public mlir::OpRewritePattern<mlir::scf::ParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ParallelOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<mlir::scf::ParallelOp>())
      return mlir::failure();

    return convertParallelToFor(op, rewriter);
  }
};

// TODO: fix ParallelLoopToGpuPass
struct RemoveNestedParallelPass
    : public plier::RewriteWrapperPass<RemoveNestedParallelPass, void, void,
                                       RemoveNestedParallel> {};

struct ParallelLoopGPUMappingPass
    : public mlir::PassWrapper<ParallelLoopGPUMappingPass,
                               mlir::OperationPass<mlir::FuncOp>> {
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

    //    mlir::greedilyMapParallelSCFToGPU(region);
  }
};

struct RemoveKernelMarkerPass
    : public mlir::PassWrapper<RemoveKernelMarkerPass,
                               mlir::OperationPass<void>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    func->walk([&](mlir::func::CallOp op) {
      if (op.getCallee() != "kernel_marker")
        return;

      if (!op.use_empty()) {
        op.emitError("Cannot erase kernel_marker with uses");
        signalPassFailure();
        return;
      }

      op.erase();
    });
  }
};

static constexpr llvm::StringLiteral kGpuArgAttr("plier.gpu_accessible");
static constexpr llvm::StringLiteral kGpuAllocShared("gpu.alloc_shared");

struct InsertGPUAllocs
    : public mlir::PassWrapper<InsertGPUAllocs,
                               mlir::OperationPass<mlir::FuncOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
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
      auto gpuAttr =
          func->getAttr(kGpuArgAttr).dyn_cast_or_null<mlir::ArrayAttr>();
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

                    auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op);
                    if (!allocOp) {
                      op->emitError("Unhandled memref producer");
                      return mlir::WalkResult::interrupt();
                    }

                    gpuBufferAllocs.insert({allocOp, {}});
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
      it.second = getAccessType(alloc);
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
        gpuAlloc->setAttr(kGpuAllocShared, builder.getUnitAttr());
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
      dims.resize(rank);
      filter.clear();
      for (auto i : llvm::seq(0u, rank)) {
        auto op = builder.create<mlir::memref::DimOp>(loc, param, i);
        dims[i] = op;
        filter.insert(op);
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
        gpuAlloc->setAttr(kGpuAllocShared, builder.getUnitAttr());

      if (access.hostWrite && access.deviceRead) {
        auto copy =
            builder.create<mlir::memref::CopyOp>(loc, param, allocResult);
        filter.insert(copy);
      }

      if (allocType != memrefType) {
        allocResult =
            builder.create<mlir::memref::CastOp>(loc, memrefType, allocResult);
      }

      param.replaceAllUsesExcept(allocResult, filter);
      builder.setInsertionPoint(term);
      if (access.hostRead && access.deviceWrite)
        builder.create<mlir::memref::CopyOp>(loc, allocResult, param);

      builder.create<mlir::memref::DeallocOp>(loc, allocResult);
    }
  }
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
  return !type.getLayout().isIdentity() ||
         (type.getRank() > 1 && !type.hasStaticShape());
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
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<FlattenLoad, FlattenStore, FlattenSubview>(ctx);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

struct KernelMemrefOpsMovementPass
    : public mlir::PassWrapper<KernelMemrefOpsMovementPass,
                               mlir::OperationPass<mlir::FuncOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto &body = func.getBody();
    if (body.empty())
      return;

    mlir::DominanceInfo dom(func);
    body.walk([&](mlir::gpu::LaunchOp launch) {
      launch.body().walk([&](mlir::Operation *op) {
        if (!mlir::isa<mlir::memref::DimOp, plier::ExtractMemrefMetadataOp>(op))
          return;

        for (auto &arg : op->getOpOperands()) {
          auto argOp = [&]() -> mlir::Operation * {
            auto val = arg.get();
            auto defOp = val.getDefiningOp();
            if (defOp)
              return defOp;

            return val.getParentBlock()->getParentOp();
          }();

          if (!dom.dominates(argOp, launch))
            return;
        }

        op->moveBefore(launch);
      });
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
        // clang-format on
    };
    spirv::Extension exts[] = {
        spirv::Extension::SPV_EXT_shader_atomic_float_add};
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

static llvm::Optional<mlir::Value> getGpuStream(mlir::OpBuilder &builder,
                                                mlir::Operation *op) {
  assert(op);
  auto func = op->getParentOfType<mlir::FuncOp>();
  if (!func)
    return {};

  if (!llvm::hasSingleElement(func.getBody()))
    return {};

  auto &block = func.getBody().front();
  auto ops = block.getOps<plier::CreateGpuStreamOp>();
  if (!ops.empty())
    return (*ops.begin()).getResult();

  mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(&block);
  auto loc = builder.getUnknownLoc();
  auto stream = builder.create<plier::CreateGpuStreamOp>(loc).getResult();
  builder.setInsertionPoint(block.getTerminator());
  builder.create<plier::DestroyGpuStreamOp>(loc, stream);
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
    if (!memrefType.hasRank() || memrefType.getRank() != 1)
      return mlir::failure();

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

// TODO: something better
class ConvertFunc : public mlir::OpConversionPattern<mlir::FuncOp> {
public:
  using mlir::OpConversionPattern<mlir::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::FuncOp op, mlir::FuncOp::Adaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.body().empty())
      return mlir::failure();

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct GPUToSpirvPass
    : public mlir::PassWrapper<GPUToSpirvPass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
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
        [](mlir::MemRefType type) -> llvm::Optional<mlir::Type> {
          if (type.hasRank() && type.getElementType().isIntOrFloat())
            return mlir::spirv::PointerType::get(
                type.getElementType(),
                mlir::spirv::StorageClass::CrossWorkgroup);
          return mlir::Type(nullptr);
        });

    mlir::ScfToSPIRVContext scfToSpirvCtx;
    mlir::populateSCFToSPIRVPatterns(typeConverter, scfToSpirvCtx, patterns);
    mlir::populateGPUToSPIRVPatterns(typeConverter, patterns);
    mlir::populateFuncToSPIRVPatterns(typeConverter, patterns);
    mlir::arith::populateArithmeticToSPIRVPatterns(typeConverter, patterns);
    mlir::populateMathToSPIRVPatterns(typeConverter, patterns);

    patterns
        .insert<ConvertSubviewOp, ConvertCastOp<mlir::memref::CastOp>,
                ConvertCastOp<mlir::memref::ReinterpretCastOp>, ConvertLoadOp,
                ConvertStoreOp, ConvertAtomicOps, ConvertFunc>(typeConverter,
                                                               context);

    if (failed(
            applyFullConversion(kernelModules, *target, std::move(patterns))))
      return signalPassFailure();
  }
};

struct GPULowerDefaultLocalSize
    : public mlir::PassWrapper<GPULowerDefaultLocalSize,
                               mlir::OperationPass<mlir::FuncOp>> {

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<plier::PlierUtilDialect>();
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

    auto skipCast = [](mlir::Value val) -> mlir::Value {
      if (auto parent = val.getDefiningOp<mlir::arith::IndexCastOp>())
        return parent.getIn();
      return val;
    };

    llvm::StringRef funcName("get_default_local_size");
    mlir::OpBuilder builder(&getContext());
    func.walk([&](mlir::gpu::LaunchFuncOp op) {
      if (auto call =
              skipCast(op.blockSizeX()).getDefiningOp<mlir::func::CallOp>()) {
        if (call.getCallee() != funcName || call.operands().size() != 3)
          return;

        assert(skipCast(op.blockSizeY()).getDefiningOp<mlir::func::CallOp>() ==
               call);
        assert(skipCast(op.blockSizeZ()).getDefiningOp<mlir::func::CallOp>() ==
               call);

        auto loc = call.getLoc();
        auto kernel = op.kernel();
        builder.setInsertionPoint(call);

        auto operands = call.operands();
        auto count = static_cast<unsigned>(operands.size());
        llvm::SmallVector<mlir::Value, 3> globalSize(count);
        for (auto i : llvm::seq(0u, count))
          globalSize[i] = plier::index_cast(builder, loc, operands[i]);

        auto res = builder
                       .create<plier::GPUSuggestBlockSizeOp>(
                           loc, /*stream*/ llvm::None, kernel, globalSize)
                       .getResults();

        for (auto i : llvm::seq(0u, count)) {
          auto castedRes = plier::index_cast(builder, loc, res[i],
                                             call.getResult(i).getType());
          call.getResult(i).replaceAllUsesWith(castedRes);
        }
      }
    });

    func.walk([&](mlir::func::CallOp op) {
      if (op.getCallee() == funcName) {
        if (!op->use_empty()) {
          op.emitError() << funcName << " call wasn't removed";
          signalPassFailure();
          return;
        }
        op->erase();
      }
    });
  }
};

struct FlattenScfIf : public mlir::OpRewritePattern<mlir::scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumResults() == 0)
      return mlir::failure();

    auto arithDialect =
        getContext()->getOrLoadDialect<mlir::arith::ArithmeticDialect>();
    auto canFlatten = [&](mlir::Operation *op) {
      return op->getDialect() == arithDialect;
    };

    auto &trueBody = op.getThenRegion().front();
    auto &falseBody = op.getElseRegion().front();
    for (auto *block : {&trueBody, &falseBody})
      for (auto &op : block->without_terminator())
        if (!canFlatten(&op))
          return mlir::failure();

    mlir::BlockAndValueMapping mapper;
    for (auto *block : {&trueBody, &falseBody})
      for (auto &op : block->without_terminator())
        rewriter.clone(op, mapper);

    auto trueYield = mlir::cast<mlir::scf::YieldOp>(trueBody.getTerminator());
    auto falseYield = mlir::cast<mlir::scf::YieldOp>(falseBody.getTerminator());

    llvm::SmallVector<mlir::Value> results;
    results.reserve(op->getNumResults());

    auto loc = op->getLoc();
    auto cond = op.getCondition();
    for (auto it : llvm::zip(trueYield.getResults(), falseYield.getResults())) {
      auto trueVal = mapper.lookupOrDefault(std::get<0>(it));
      auto falseVal = mapper.lookupOrDefault(std::get<1>(it));
      auto res =
          rewriter.create<mlir::arith::SelectOp>(loc, cond, trueVal, falseVal);
      results.emplace_back(res);
    }

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct FlattenScfPass : public plier::RewriteWrapperPass<FlattenScfPass, void,
                                                         void, FlattenScfIf> {};

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
  auto module = builder.create<plier::LoadGpuModuleOp>(loc, *stream, gpuMod);
  auto kernel = builder.create<plier::GetGpuKernelOp>(loc, module, gpuKernel);
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
          return builder.create<plier::LaunchGpuKernelOp>(
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

    auto shared = op->hasAttr(kGpuAllocShared);

    mlir::Type token = op.asyncToken() ? op.asyncToken().getType() : nullptr;
    auto res = rewriter.replaceOpWithNewOp<plier::GPUAllocOp>(
        op, op.getType(), token, op.asyncDependencies(), *stream,
        op.dynamicSizes(), op.symbolOperands());

    if (shared)
      res->setAttr(kGpuAllocShared, rewriter.getUnitAttr());

    return mlir::success();
  }
};

struct ExpandSuggestBlockSizeOp
    : public mlir::OpRewritePattern<plier::GPUSuggestBlockSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::GPUSuggestBlockSizeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.kernel())
      return mlir::failure();

    assert(op.kernelRef());
    return createGpuKernelLoad(
        rewriter, op,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value stream,
            mlir::Value kernel) {
          return builder.create<plier::GPUSuggestBlockSizeOp>(
              loc, stream, kernel, op.gridSize());
        });
  }
};

struct GPUExPass
    : public mlir::PassWrapper<GPUExPass, mlir::OperationPass<void>> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<ExpandLaunchOp, ExpandAllocOp, ExpandSuggestBlockSizeOp>(
        ctx);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

static mlir::LogicalResult processAllocUser(mlir::Operation *user,
                                            mlir::Operation *allocParent,
                                            mlir::DominanceInfo &dom,
                                            mlir::Operation *&lastUser) {
  auto origUser = user;
  if (user->hasTrait<mlir::OpTrait::IsTerminator>())
    return mlir::failure();

  auto parent = user->getParentOp();
  while (parent != allocParent) {
    user = parent;
    parent = user->getParentOp();
    if (parent == nullptr)
      return mlir::failure();
  }

  if (dom.properlyDominates(lastUser, user))
    lastUser = user;

  for (auto resUser : origUser->getUsers())
    if (mlir::failed(processAllocUser(resUser, allocParent, dom, lastUser)))
      return mlir::failure();

  return mlir::success();
}

template <typename AllocOp, typename DeallocOp>
struct CreateDeallocOp : public mlir::OpRewritePattern<AllocOp> {
  using mlir::OpRewritePattern<AllocOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AllocOp op, mlir::PatternRewriter &rewriter) const override {
    auto allocParent = op->getParentOp();
    mlir::Operation *lastUser = op;
    mlir::DominanceInfo dom;
    for (auto user : op->getUsers())
      if (mlir::isa<DeallocOp>(user) ||
          mlir::failed(processAllocUser(user, allocParent, dom, lastUser)))
        return mlir::failure();

    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(lastUser);
    rewriter.create<DeallocOp>(lastUser->getLoc(), op);
    return mlir::success();
  }
};

struct GPUExDeallocPass
    : public mlir::PassWrapper<GPUExDeallocPass, mlir::OperationPass<void>> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<
        CreateDeallocOp<plier::LoadGpuModuleOp, plier::DestroyGpuModuleOp>,
        CreateDeallocOp<plier::GetGpuKernelOp, plier::DestroyGpuKernelOp>>(ctx);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

template <typename Op, typename ReleaseOp>
static bool outlineOp(mlir::Operation &op,
                      llvm::SmallVectorImpl<mlir::Operation *> &deinit) {
  if (!mlir::isa<Op>(op))
    return false;

  auto opParent = op.getParentOp();
  auto origSize = deinit.size();
  for (auto user : op.getUsers()) {
    if (!mlir::isa<ReleaseOp>(user) || llvm::is_contained(deinit, user))
      continue;

    if (user->getParentOp() != opParent || user->getNumResults() != 0) {
      deinit.resize(origSize);
      return false;
    }
    deinit.emplace_back(user);
  }
  return true;
}

constexpr static llvm::StringLiteral kOutlinedInitAttr("plier.outlined_init");
constexpr static llvm::StringLiteral
    kOutlinedDeinitAttr("plier.outlined_deinit");

struct OutlineInitPass
    : public mlir::PassWrapper<OutlineInitPass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  void runOnOperation() override {
    auto mod = getOperation();

    using outline_func_t =
        bool (*)(mlir::Operation &, llvm::SmallVectorImpl<mlir::Operation *> &);
    const outline_func_t outlineHandlers[] = {
        &outlineOp<plier::CreateGpuStreamOp, plier::DestroyGpuStreamOp>,
        &outlineOp<plier::LoadGpuModuleOp, plier::DestroyGpuModuleOp>,
        &outlineOp<plier::GetGpuKernelOp, plier::DestroyGpuKernelOp>,
    };

    llvm::SmallVector<mlir::Operation *> initOps;
    llvm::SmallVector<mlir::Operation *> deinitOps;
    llvm::SmallVector<mlir::Type> types;
    llvm::SmallVector<mlir::Value> values;
    mlir::BlockAndValueMapping mapper;
    auto tryOutlineOp = [&](mlir::Operation &op) {
      for (auto arg : op.getOperands()) {
        auto argOp = arg.getDefiningOp();
        if (!argOp || !llvm::is_contained(initOps, argOp))
          return;
      }

      for (auto handler : outlineHandlers) {
        if (handler(op, deinitOps)) {
          initOps.emplace_back(&op);
          return;
        }
      }
    };

    mlir::OpBuilder builder(&getContext());
    auto unknownLoc = builder.getUnknownLoc();
    for (auto func : mod.getOps<mlir::FuncOp>()) {
      auto &body = func.getBody();
      if (!llvm::hasSingleElement(body))
        continue;

      auto funcName = func.getName();
      initOps.clear();
      deinitOps.clear();
      for (auto &op : body.front())
        tryOutlineOp(op);

      if (!initOps.empty()) {
        builder.setInsertionPointToStart(mod.getBody());
        types.clear();
        for (auto *op : initOps)
          for (auto type : op->getResultTypes())
            types.emplace_back(type);

        auto funcType = builder.getFunctionType(llvm::None, types);
        auto func = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(), (funcName + "outlined_init").str(),
            funcType);
        func.setPrivate();
        func->setAttr(kOutlinedInitAttr, builder.getUnitAttr());
        auto block = func.addEntryBlock();
        builder.setInsertionPointToStart(block);
        mapper.clear();
        values.clear();
        for (auto *op : initOps) {
          auto *newOp = builder.clone(*op, mapper);
          for (auto res : newOp->getResults())
            values.emplace_back(res);
        }
        builder.create<mlir::func::ReturnOp>(unknownLoc, values);

        builder.setInsertionPoint(initOps.front());
        auto call = builder.create<mlir::func::CallOp>(unknownLoc, func);
        call->setAttr(kOutlinedInitAttr, builder.getUnitAttr());
        auto results = call.getResults();
        values.assign(results.begin(), results.end());
        for (auto *op : llvm::reverse(initOps)) {
          auto numRes = op->getNumResults();
          assert(results.size() >= numRes);
          auto newRes = results.take_back(numRes);
          op->replaceAllUsesWith(newRes);
          results = results.drop_back(numRes);
          op->erase();
        }
      }

      if (!deinitOps.empty()) {
        assert(!initOps.empty());
        builder.setInsertionPointToStart(mod.getBody());
        assert(!types.empty());
        auto funcType = builder.getFunctionType(types, llvm::None);
        auto func = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(), (funcName + "outlined_deinit").str(),
            funcType);
        func.setPrivate();
        func->setAttr(kOutlinedDeinitAttr, builder.getUnitAttr());

        auto block = func.addEntryBlock();
        builder.setInsertionPointToStart(block);
        mapper.clear();
        mapper.map(values, block->getArguments());
        for (auto *op : llvm::reverse(deinitOps))
          builder.clone(*op, mapper);

        builder.create<mlir::func::ReturnOp>(unknownLoc);

        builder.setInsertionPoint(deinitOps.front());
        auto call =
            builder.create<mlir::func::CallOp>(unknownLoc, func, values);
        call->setAttr(kOutlinedDeinitAttr, builder.getUnitAttr());
        for (auto *op : deinitOps)
          op->erase();
      }
    }
  }
};

struct GenerateOutlineContextPass
    : public mlir::PassWrapper<GenerateOutlineContextPass,
                               mlir::OperationPass<mlir::FuncOp>> {

  void runOnOperation() override {
    auto func = getOperation();
    auto &body = func.getBody();
    if (body.empty())
      return;

    if (!llvm::hasSingleElement(body)) {
      func.emitError("Only strucutred control flow is supported");
      signalPassFailure();
      return;
    }

    mlir::OpBuilder builder(&getContext());
    auto initAttr = builder.getStringAttr(kOutlinedInitAttr);
    auto deinitAttr = builder.getStringAttr(kOutlinedDeinitAttr);

    mlir::func::CallOp init;
    mlir::func::CallOp deinit;
    for (auto &op : body.front()) {
      auto call = mlir::dyn_cast<mlir::func::CallOp>(op);
      if (!call)
        continue;

      if (call->hasAttr(initAttr)) {
        if (init) {
          call.emitError("More than one init function");
          signalPassFailure();
          return;
        }
        init = call;
      }

      if (call->hasAttr(deinitAttr)) {
        if (deinit) {
          call.emitError("More than one deinit function");
          signalPassFailure();
          return;
        }
        deinit = call;
      }
    }

    if (!init)
      return;

    mlir::SymbolRefAttr initSym = init.getCalleeAttr();
    mlir::SymbolRefAttr deinitSym = (deinit ? deinit.getCalleeAttr() : nullptr);

    builder.setInsertionPoint(init);
    auto res =
        builder
            .create<plier::TakeContextOp>(init->getLoc(), initSym, deinitSym,
                                          init->getResultTypes())
            .getResults();
    assert(res.size() > 1);
    auto ctx = res.front();
    auto resValues = res.drop_front(1);
    init->replaceAllUsesWith(resValues);
    init->erase();

    if (deinit) {
      builder.setInsertionPoint(deinit);
      builder.create<plier::ReleaseContextOp>(deinit->getLoc(), ctx);
      deinit->erase();
    } else {
      builder.setInsertionPoint(body.front().getTerminator());
      builder.create<plier::ReleaseContextOp>(builder.getUnknownLoc(), ctx);
    }
  }
};

struct FunctionCallBuilder {
  FunctionCallBuilder(mlir::StringRef functionName, mlir::Type returnType,
                      mlir::ArrayRef<mlir::Type> argumentTypes)
      : functionName(functionName),
        functionType(
            mlir::LLVM::LLVMFunctionType::get(returnType, argumentTypes)) {}
  mlir::LLVM::CallOp create(mlir::Location loc, mlir::OpBuilder &builder,
                            mlir::ArrayRef<mlir::Value> arguments) const {
    auto module =
        builder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
    auto function = [&] {
      if (auto function =
              module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(functionName))
        return function;
      return mlir::OpBuilder::atBlockEnd(module.getBody())
          .create<mlir::LLVM::LLVMFuncOp>(loc, functionName, functionType);
    }();
    return builder.create<mlir::LLVM::CallOp>(loc, function, arguments);
  }

private:
  mlir::StringRef functionName;
  mlir::LLVM::LLVMFunctionType functionType;
};

static const char *kEventCountAttrName = "gpu.event_count";
static const char *kEventIndexAttrName = "gpu.event_index";

template <typename OpTy>
class ConvertOpToGpuRuntimeCallPattern
    : public mlir::ConvertOpToLLVMPattern<OpTy> {
public:
  explicit ConvertOpToGpuRuntimeCallPattern(mlir::LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<OpTy>(converter) {}

protected:
  mlir::MLIRContext *context = &this->getTypeConverter()->getContext();

  mlir::Type llvmVoidType = mlir::LLVM::LLVMVoidType::get(context);
  mlir::Type llvmPointerType =
      mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
  mlir::Type llvmPointerPointerType =
      mlir::LLVM::LLVMPointerType::get(llvmPointerType);
  mlir::Type llvmInt8Type = mlir::IntegerType::get(context, 8);
  mlir::Type llvmInt32Type = mlir::IntegerType::get(context, 32);
  mlir::Type llvmInt64Type = mlir::IntegerType::get(context, 64);
  mlir::Type llvmIndexType = mlir::IntegerType::get(
      context, this->getTypeConverter()->getPointerBitwidth(0));

  mlir::Type llvmI32PtrType = mlir::LLVM::LLVMPointerType::get(llvmIndexType);

  mlir::Type llvmRangeType = mlir::LLVM::LLVMStructType::getLiteral(
      context, {llvmPointerType, llvmIndexType});
  mlir::Type llvmRangePointerType =
      mlir::LLVM::LLVMPointerType::get(llvmRangeType);
  mlir::Type llvmAllocResType = mlir::LLVM::LLVMStructType::getLiteral(
      context, {llvmPointerType, llvmPointerType, llvmPointerType});
  mlir::Type llvmAllocResPtrType =
      mlir::LLVM::LLVMPointerType::get(llvmAllocResType);

  FunctionCallBuilder streamCreateCallBuilder = {
      "dpcompGpuStreamCreate",
      llvmPointerType, // stream
      {
          llvmIndexType // events count
      }};

  FunctionCallBuilder streamDestroyCallBuilder = {"dpcompGpuStreamDestroy",
                                                  llvmVoidType,
                                                  {
                                                      llvmPointerType // stream
                                                  }};

  FunctionCallBuilder moduleLoadCallBuilder = {"dpcompGpuModuleLoad",
                                               llvmPointerType, // module
                                               {
                                                   llvmPointerType, // stream
                                                   llvmPointerType, // data ptr
                                                   llvmIndexType,   // data size
                                               }};

  FunctionCallBuilder moduleDestroyCallBuilder = {"dpcompGpuModuleDestroy",
                                                  llvmVoidType,
                                                  {
                                                      llvmPointerType // module
                                                  }};

  FunctionCallBuilder kernelGetCallBuilder = {"dpcompGpuKernelGet",
                                              llvmPointerType, // kernel
                                              {
                                                  llvmPointerType, // module
                                                  llvmPointerType, // name
                                              }};

  FunctionCallBuilder kernelDestroyCallBuilder = {"dpcompGpuKernelDestroy",
                                                  llvmVoidType,
                                                  {
                                                      llvmPointerType // kernel
                                                  }};

  FunctionCallBuilder launchKernelCallBuilder = {
      "dpcompGpuLaunchKernel",
      llvmPointerType, // dep
      {
          llvmPointerType,        // stream
          llvmPointerType,        // kernel
          llvmIndexType,          // gridXDim
          llvmIndexType,          // gridyDim
          llvmIndexType,          // gridZDim
          llvmIndexType,          // blockXDim
          llvmIndexType,          // blockYDim
          llvmIndexType,          // blockZDim
          llvmPointerPointerType, // deps (null-term)
          llvmRangePointerType,   // params (null-term)
          llvmIndexType,          // eventIndex
      }};

  FunctionCallBuilder waitEventCallBuilder = {"dpcompGpuWait",
                                              llvmVoidType,
                                              {
                                                  llvmPointerType // dep
                                              }};

  FunctionCallBuilder allocCallBuilder = {
      "dpcompGpuAlloc",
      llvmVoidType,
      {
          llvmPointerType,        // stream
          llvmIndexType,          // size
          llvmIndexType,          // alignment
          llvmInt32Type,          // shared
          llvmPointerPointerType, // deps (null-term)
          llvmIndexType,          // eventIndex
          llvmAllocResPtrType,    // result
      }};

  FunctionCallBuilder suggestBlockSizeBuilder = {
      "dpcompGpuSuggestBlockSize",
      llvmVoidType,
      {
          llvmPointerType, // stream
          llvmPointerType, // kernel
          llvmI32PtrType,  // grid sizes
          llvmI32PtrType,  // ret block sizes
          llvmIndexType,   // dim count
      }};

  mlir::Value createDepsArray(mlir::OpBuilder &rewriter, mlir::Location loc,
                              mlir::Operation *op,
                              mlir::ValueRange deps) const {
    auto depsArraySize = static_cast<unsigned>(deps.size());
    auto depsArrayType =
        mlir::LLVM::LLVMArrayType::get(llvmPointerType, depsArraySize + 1);
    mlir::Value depsArray =
        rewriter.create<mlir::LLVM::UndefOp>(loc, depsArrayType);
    for (auto i : llvm::seq(0u, depsArraySize)) {
      auto index = rewriter.getI64ArrayAttr(i);
      depsArray = rewriter.create<mlir::LLVM::InsertValueOp>(loc, depsArray,
                                                             deps[i], index);
    }
    auto nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, llvmPointerType);
    depsArray = rewriter.create<mlir::LLVM::InsertValueOp>(
        loc, depsArray, nullPtr, rewriter.getI64ArrayAttr(depsArraySize));

    auto depsArrayPtrType = mlir::LLVM::LLVMPointerType::get(depsArrayType);
    plier::AllocaInsertionPoint allocaHelper(op);
    auto depsArrayPtr = allocaHelper.insert(rewriter, [&]() {
      auto size = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(1));
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, depsArrayPtrType, size,
                                                   0);
    });

    rewriter.create<mlir::LLVM::StoreOp>(loc, depsArray, depsArrayPtr);

    return rewriter.create<mlir::LLVM::BitcastOp>(loc, llvmPointerPointerType,
                                                  depsArrayPtr);
  }

  mlir::Value createEventIndexVar(mlir::OpBuilder &rewriter, mlir::Location loc,
                                  mlir::Operation *op) const {
    auto eventIndex = [&]() -> int64_t {
      auto value = mlir::getConstantIntValue(op->getAttr(kEventIndexAttrName));
      if (!value)
        return -1;

      return *value;
    }();
    return rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, eventIndex));
  }
};

class ConvertGpuStreamCreatePattern
    : public ConvertOpToGpuRuntimeCallPattern<plier::CreateGpuStreamOp> {
public:
  ConvertGpuStreamCreatePattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<plier::CreateGpuStreamOp>(converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(plier::CreateGpuStreamOp op,
                  plier::CreateGpuStreamOp::Adaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto eventsCount =
        mlir::getConstantIntValue(mod->getAttr(kEventCountAttrName));
    if (!eventsCount)
      return mlir::failure();

    auto loc = op.getLoc();
    auto eventsCountVar =
        rewriter
            .create<mlir::LLVM::ConstantOp>(
                loc, llvmIndexType,
                rewriter.getIntegerAttr(llvmIndexType, *eventsCount))
            .getResult();
    auto res = streamCreateCallBuilder.create(loc, rewriter, eventsCountVar);
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuStreamDestroyPattern
    : public ConvertOpToGpuRuntimeCallPattern<plier::DestroyGpuStreamOp> {
public:
  ConvertGpuStreamDestroyPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<plier::DestroyGpuStreamOp>(converter) {
  }

private:
  mlir::LogicalResult
  matchAndRewrite(plier::DestroyGpuStreamOp op,
                  plier::DestroyGpuStreamOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto res = streamDestroyCallBuilder.create(loc, rewriter, adaptor.source());
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

static std::string getUniqueLLVMGlobalName(mlir::ModuleOp mod,
                                           mlir::StringRef srcName) {
  auto globals = mod.getOps<mlir::LLVM::GlobalOp>();
  for (int i = 0;; ++i) {
    auto name =
        (i == 0 ? std::string(srcName) : (srcName + llvm::Twine(i)).str());
    auto isSameName = [&](mlir::LLVM::GlobalOp global) {
      return global.getName() == name;
    };
    if (llvm::find_if(globals, isSameName) == globals.end())
      return name;
  }
}

class ConvertGpuModuleLoadPattern
    : public ConvertOpToGpuRuntimeCallPattern<plier::LoadGpuModuleOp> {
public:
  ConvertGpuModuleLoadPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<plier::LoadGpuModuleOp>(converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(plier::LoadGpuModuleOp op,
                  plier::LoadGpuModuleOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto gpuMod = mod.lookupSymbol<mlir::gpu::GPUModuleOp>(op.module());
    if (!gpuMod)
      return mlir::failure();

    auto blobAttr = gpuMod->getAttrOfType<mlir::StringAttr>(
        mlir::gpu::getDefaultGpuBinaryAnnotation());
    if (!blobAttr)
      return mlir::failure();

    auto blob = blobAttr.getValue();

    auto loc = op.getLoc();
    auto name = getUniqueLLVMGlobalName(mod, "gpu_blob");
    auto data = mlir::LLVM::createGlobalString(loc, rewriter, name, blob,
                                               mlir::LLVM::Linkage::Internal);
    auto size = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType,
        mlir::IntegerAttr::get(llvmIndexType,
                               static_cast<int64_t>(blob.size())));
    auto res = moduleLoadCallBuilder.create(loc, rewriter,
                                            {adaptor.stream(), data, size});
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuModuleDestroyPattern
    : public ConvertOpToGpuRuntimeCallPattern<plier::DestroyGpuModuleOp> {
public:
  ConvertGpuModuleDestroyPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<plier::DestroyGpuModuleOp>(converter) {
  }

private:
  mlir::LogicalResult
  matchAndRewrite(plier::DestroyGpuModuleOp op,
                  plier::DestroyGpuModuleOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto res = moduleDestroyCallBuilder.create(loc, rewriter, adaptor.source());
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuKernelGetPattern
    : public ConvertOpToGpuRuntimeCallPattern<plier::GetGpuKernelOp> {
public:
  ConvertGpuKernelGetPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<plier::GetGpuKernelOp>(converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(plier::GetGpuKernelOp op,
                  plier::GetGpuKernelOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto loc = op.getLoc();
    llvm::SmallString<64> name = op.kernel().getLeafReference().getValue();

    auto varName = getUniqueLLVMGlobalName(mod, "kernel_name");
    name.push_back('\0');
    auto data = mlir::LLVM::createGlobalString(loc, rewriter, varName, name,
                                               mlir::LLVM::Linkage::Internal);
    auto res =
        kernelGetCallBuilder.create(loc, rewriter, {adaptor.module(), data});
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuKernelDestroyPattern
    : public ConvertOpToGpuRuntimeCallPattern<plier::DestroyGpuKernelOp> {
public:
  ConvertGpuKernelDestroyPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<plier::DestroyGpuKernelOp>(converter) {
  }

private:
  mlir::LogicalResult
  matchAndRewrite(plier::DestroyGpuKernelOp op,
                  plier::DestroyGpuKernelOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto res = kernelDestroyCallBuilder.create(loc, rewriter, adaptor.source());
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuKernelLaunchPattern
    : public ConvertOpToGpuRuntimeCallPattern<plier::LaunchGpuKernelOp> {
public:
  ConvertGpuKernelLaunchPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<plier::LaunchGpuKernelOp>(converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(plier::LaunchGpuKernelOp op,
                  plier::LaunchGpuKernelOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto depsArrayPtr =
        createDepsArray(rewriter, loc, op, adaptor.asyncDependencies());

    plier::AllocaInsertionPoint allocaHelper(op);
    auto kernelParams = adaptor.operands();
    auto paramsCount = static_cast<unsigned>(kernelParams.size());
    auto paramsArrayType =
        mlir::LLVM::LLVMArrayType::get(llvmRangeType, paramsCount + 1);
    auto paramsArrayPtrType = mlir::LLVM::LLVMPointerType::get(paramsArrayType);

    auto getKernelParamType = [&](unsigned i) -> mlir::Type {
      if (op.operands()[i].getType().isa<mlir::MemRefType>()) {
        mlir::MemRefDescriptor desc(kernelParams[i]);
        return desc.getElementPtrType();
      }

      return kernelParams[i].getType();
    };

    llvm::SmallVector<mlir::Value> paramsStorage(paramsCount);
    auto paramsArrayPtr = allocaHelper.insert(rewriter, [&]() {
      auto size = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(paramsCount));
      for (auto i : llvm::seq(0u, paramsCount)) {
        auto ptrType = mlir::LLVM::LLVMPointerType::get(getKernelParamType(i));
        paramsStorage[i] =
            rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrType, size, 0);
      }
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, paramsArrayPtrType,
                                                   size, 0);
    });

    auto getKernelParam = [&](unsigned i) -> mlir::Value {
      if (op.operands()[i].getType().isa<mlir::MemRefType>()) {
        mlir::MemRefDescriptor desc(kernelParams[i]);
        return desc.alignedPtr(rewriter, loc);
      }

      return kernelParams[i];
    };

    mlir::Value paramsArray =
        rewriter.create<mlir::LLVM::UndefOp>(loc, paramsArrayType);
    auto one = rewriter
                   .create<mlir::LLVM::ConstantOp>(
                       loc, llvmInt32Type, rewriter.getI32IntegerAttr(1))
                   .getResult();
    for (auto i : llvm::seq(0u, paramsCount)) {
      rewriter.create<mlir::LLVM::StoreOp>(loc, getKernelParam(i),
                                           paramsStorage[i]);
      auto ptr = rewriter.create<mlir::LLVM::BitcastOp>(loc, llvmPointerType,
                                                        paramsStorage[i]);
      // %Size = getelementptr %T* null, int 1
      // %SizeI = ptrtoint %T* %Size to i32
      auto paramPtrType = paramsStorage[i].getType();
      auto nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, paramPtrType);
      auto gep =
          rewriter.create<mlir::LLVM::GEPOp>(loc, paramPtrType, nullPtr, one);
      auto typeSize =
          rewriter.create<mlir::LLVM::PtrToIntOp>(loc, llvmIndexType, gep);

      mlir::Value range =
          rewriter.create<mlir::LLVM::UndefOp>(loc, llvmRangeType);
      range = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, range, ptr, rewriter.getI64ArrayAttr(0));
      range = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, range, typeSize, rewriter.getI64ArrayAttr(1));

      paramsArray = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, paramsArray, range, rewriter.getI64ArrayAttr(i));
    }

    auto nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, llvmPointerType);
    auto nullRange = [&]() {
      auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 0));
      mlir::Value range =
          rewriter.create<mlir::LLVM::UndefOp>(loc, llvmRangeType);
      range = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, range, nullPtr, rewriter.getI64ArrayAttr(0));
      range = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, range, zero, rewriter.getI64ArrayAttr(1));
      return range;
    }();
    paramsArray = rewriter.create<mlir::LLVM::InsertValueOp>(
        loc, paramsArray, nullRange, rewriter.getI64ArrayAttr(paramsCount));
    rewriter.create<mlir::LLVM::StoreOp>(loc, paramsArray, paramsArrayPtr);

    auto eventIndexVar = createEventIndexVar(rewriter, loc, op);

    auto paramsArrayVoidPtr = rewriter.create<mlir::LLVM::BitcastOp>(
        loc, llvmRangePointerType, paramsArrayPtr);
    mlir::Value params[] = {
        // clang-format off
        adaptor.stream(),
        adaptor.kernel(),
        adaptor.gridSizeX(),
        adaptor.gridSizeY(),
        adaptor.gridSizeZ(),
        adaptor.blockSizeX(),
        adaptor.blockSizeY(),
        adaptor.blockSizeZ(),
        depsArrayPtr,
        paramsArrayVoidPtr,
        eventIndexVar,
        // clang-format on
    };
    auto res = launchKernelCallBuilder.create(loc, rewriter, params);
    if (op.getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      assert(res.getNumResults() == op.getNumResults());
      rewriter.replaceOp(op, res.getResults());
    }
    return mlir::success();
  }
};

class ConvertGpuAllocPattern
    : public ConvertOpToGpuRuntimeCallPattern<plier::GPUAllocOp> {
public:
  ConvertGpuAllocPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<plier::GPUAllocOp>(converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(plier::GPUAllocOp op, plier::GPUAllocOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.symbolOperands().empty())
      return mlir::failure();

    auto memrefType = op.getType();
    auto converter = getTypeConverter();
    auto dstType = converter->convertType(memrefType);
    if (!dstType)
      return mlir::failure();

    auto loc = op.getLoc();

    mlir::SmallVector<mlir::Value, 4> shape;
    mlir::SmallVector<mlir::Value, 4> strides;
    mlir::Value sizeBytes;
    getMemRefDescriptorSizes(loc, memrefType, adaptor.dynamicSizes(), rewriter,
                             shape, strides, sizeBytes);

    assert(shape.size() == strides.size());

    auto alignment = rewriter.getIntegerAttr(llvmIndexType, 64);
    auto alignmentVar =
        rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmIndexType, alignment);

    bool shared = op->hasAttr(kGpuAllocShared);
    auto sharedVar = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmInt32Type,
        rewriter.getI32IntegerAttr(static_cast<int>(shared)));

    auto depsArrayPtr =
        createDepsArray(rewriter, loc, op, adaptor.asyncDependencies());

    auto eventIndexVar = createEventIndexVar(rewriter, loc, op);

    plier::AllocaInsertionPoint allocaHelper(op);
    auto resultPtr = allocaHelper.insert(rewriter, [&]() {
      auto size = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(1));
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, llvmAllocResPtrType,
                                                   size, 0);
    });

    mlir::Value params[] = {
        // clang-format off
        adaptor.stream(),
        sizeBytes,
        alignmentVar,
        sharedVar,
        depsArrayPtr,
        eventIndexVar,
        resultPtr,
        // clang-format on
    };
    allocCallBuilder.create(loc, rewriter, params);
    auto res = rewriter.create<mlir::LLVM::LoadOp>(loc, resultPtr);
    auto meminfo = rewriter.create<mlir::LLVM::ExtractValueOp>(
        loc, llvmPointerType, res, rewriter.getI64ArrayAttr(0));
    auto dataPtr = rewriter.create<mlir::LLVM::ExtractValueOp>(
        loc, llvmPointerType, res, rewriter.getI64ArrayAttr(1));

    auto memrefDesc = mlir::MemRefDescriptor::undef(rewriter, loc, dstType);
    auto elemPtrTye = memrefDesc.getElementPtrType();
    memrefDesc.setAllocatedPtr(
        rewriter, loc,
        rewriter.create<mlir::LLVM::BitcastOp>(loc, elemPtrTye, meminfo));
    memrefDesc.setAlignedPtr(
        rewriter, loc,
        rewriter.create<mlir::LLVM::BitcastOp>(loc, elemPtrTye, dataPtr));

    auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 0));

    memrefDesc.setOffset(rewriter, loc, zero);
    for (auto i : llvm::seq(0u, static_cast<unsigned>(shape.size()))) {
      memrefDesc.setSize(rewriter, loc, i, shape[i]);
      memrefDesc.setStride(rewriter, loc, i, strides[i]);
    }

    mlir::Value resMemref = memrefDesc;
    if (op.getNumResults() == 1) {
      rewriter.replaceOp(op, resMemref);
    } else {
      auto event = rewriter.create<mlir::LLVM::ExtractValueOp>(
          loc, llvmPointerType, res, rewriter.getI64ArrayAttr(2));
      mlir::Value vals[] = {
          resMemref,
          event,
      };
      rewriter.replaceOp(op, vals);
    }
    return mlir::success();
  }
};

class ConvertGpuSuggestBlockSizePattern
    : public ConvertOpToGpuRuntimeCallPattern<plier::GPUSuggestBlockSizeOp> {
public:
  ConvertGpuSuggestBlockSizePattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<plier::GPUSuggestBlockSizeOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(plier::GPUSuggestBlockSizeOp op,
                  plier::GPUSuggestBlockSizeOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto numDims = op.getNumResults();
    auto loc = op.getLoc();
    plier::AllocaInsertionPoint allocaHelper(op);
    auto gridArrayPtr = allocaHelper.insert(rewriter, [&]() {
      auto size = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(numDims));
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, llvmI32PtrType, size,
                                                   0);
    });
    auto blockArrayPtr = allocaHelper.insert(rewriter, [&]() {
      auto size = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(numDims));
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, llvmI32PtrType, size,
                                                   0);
    });

    auto sizesType = mlir::LLVM::LLVMArrayType::get(llvmInt32Type, numDims);
    auto sizesPtrType = mlir::LLVM::LLVMPointerType::get((sizesType));
    auto castToSizesPtrType = [&](mlir::Value val) {
      return rewriter.create<mlir::LLVM::BitcastOp>(loc, sizesPtrType, val);
    };

    mlir::Value gridArray =
        rewriter.create<mlir::LLVM::UndefOp>(loc, sizesType);
    for (auto i : llvm::seq(0u, numDims)) {
      auto index = rewriter.getI64ArrayAttr(i);
      auto gridSize = rewriter.create<mlir::LLVM::TruncOp>(
          loc, llvmInt32Type, adaptor.gridSize()[i]);
      gridArray = rewriter.create<mlir::LLVM::InsertValueOp>(loc, gridArray,
                                                             gridSize, index);
    }

    rewriter.create<mlir::LLVM::StoreOp>(loc, gridArray,
                                         castToSizesPtrType(gridArrayPtr));
    mlir::Value numDimsVal = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, numDims));

    mlir::Value params[] = {
        // clang-format off
        adaptor.stream(),
        adaptor.kernel(),
        gridArrayPtr,
        blockArrayPtr,
        numDimsVal,
        // clang-format on
    };

    suggestBlockSizeBuilder.create(loc, rewriter, params);

    mlir::Value blockSizeArray = rewriter.create<mlir::LLVM::LoadOp>(
        loc, castToSizesPtrType(blockArrayPtr));
    llvm::SmallVector<mlir::Value, 3> result(numDims);
    for (auto i : llvm::seq(0u, numDims)) {
      auto ind = rewriter.getI64ArrayAttr(i);
      auto blockSize = rewriter.create<mlir::LLVM::ExtractValueOp>(
          loc, llvmInt32Type, blockSizeArray, ind);
      result[i] =
          rewriter.create<mlir::LLVM::ZExtOp>(loc, llvmIndexType, blockSize);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct EnumerateEventsPass
    : public mlir::PassWrapper<EnumerateEventsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto mod = getOperation();
    int64_t eventCount = 0;
    auto intType = mlir::IntegerType::get(&getContext(), 64);
    mod.walk([&](mlir::gpu::AsyncOpInterface op) {
      if (op.getAsyncToken()) {
        op->setAttr(kEventIndexAttrName,
                    mlir::IntegerAttr::get(intType, eventCount));
        ++eventCount;
      }
    });
    mod->setAttr(kEventCountAttrName,
                 mlir::IntegerAttr::get(intType, eventCount));
  }
};

struct GPUToLLVMPass
    : public mlir::PassWrapper<GPUToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    mlir::LLVMTypeConverter converter(&getContext());
    mlir::RewritePatternSet patterns(&getContext());
    mlir::LLVMConversionTarget target(getContext());

    auto llvmPointerType = mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(&getContext(), 8));
    converter.addConversion([llvmPointerType](plier::OpaqueType) -> mlir::Type {
      return llvmPointerType;
    });

    target.addIllegalDialect<mlir::gpu::GPUDialect>();
    target.addIllegalOp<
        // clang-format off
        plier::CreateGpuStreamOp,
        plier::DestroyGpuStreamOp,
        plier::LoadGpuModuleOp,
        plier::DestroyGpuModuleOp,
        plier::GetGpuKernelOp,
        plier::DestroyGpuKernelOp,
        plier::LaunchGpuKernelOp,
        plier::GPUAllocOp,
        plier::GPUSuggestBlockSizeOp
        // clang-format on
        >();

    mlir::populateAsyncStructuralTypeConversionsAndLegality(converter, patterns,
                                                            target);
    mlir::populateGpuToLLVMConversionPatterns(
        converter, patterns, mlir::gpu::getDefaultGpuBinaryAnnotation());

    patterns.insert<
        // clang-format off
        ConvertGpuStreamCreatePattern,
        ConvertGpuStreamDestroyPattern,
        ConvertGpuModuleLoadPattern,
        ConvertGpuModuleDestroyPattern,
        ConvertGpuKernelGetPattern,
        ConvertGpuKernelDestroyPattern,
        ConvertGpuKernelLaunchPattern,
        ConvertGpuAllocPattern,
        ConvertGpuSuggestBlockSizePattern
        // clang-format on
        >(converter);

    auto mod = getOperation();
    if (mlir::failed(
            mlir::applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};

void rerun_std_pipeline(mlir::Operation *op) {
  assert(nullptr != op);
  auto marker =
      mlir::StringAttr::get(op->getContext(), plierToStdPipelineName());
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  assert(nullptr != mod);
  plier::add_pipeline_jump_marker(mod, marker);
}

struct LowerGpuRange final : public plier::CallOpLowering {
  using CallOpLowering::CallOpLowering;

protected:
  virtual mlir::LogicalResult
  resolveCall(plier::PyCallOp op, mlir::StringRef name, mlir::Location /*loc*/,
              mlir::PatternRewriter &rewriter, mlir::ValueRange args,
              KWargs kwargs) const override {
    if (name != "_gpu_range")
      return mlir::failure();

    auto parent = op->getParentOp();
    auto setAttr = [](mlir::scf::ForOp op) {
      auto unitAttr = mlir::UnitAttr::get(op->getContext());
      op->setAttr(plier::attributes::getParallelName(), unitAttr);
      op->setAttr(plier::attributes::getGpuRangeName(), unitAttr);
    };
    if (mlir::failed(lowerRange(op, args, kwargs, rewriter, setAttr)))
      return mlir::failure();

    rerun_std_pipeline(parent);
    return mlir::success();
  }
};

template <typename Op> struct ConvertOp : public mlir::OpConversionPattern<Op> {
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto origResTypes = op->getResultTypes();
    llvm::SmallVector<mlir::Type, 2> newResTypes;

    auto typeConverter = this->getTypeConverter();
    assert(typeConverter);
    if (mlir::failed(typeConverter->convertTypes(origResTypes, newResTypes)))
      return mlir::failure();

    auto attrs = adaptor.getAttributes();
    llvm::SmallVector<mlir::NamedAttribute> attrsList;
    attrsList.reserve(attrs.size());
    for (auto it : attrs)
      attrsList.emplace_back(it.getName(), it.getValue());

    rewriter.replaceOpWithNewOp<Op>(op, newResTypes, adaptor.getOperands(),
                                    attrsList);
    return mlir::success();
  }
};

static bool isGpuArray(llvm::StringRef &name) {
  return name.consume_front("USM:ndarray(") && name.consume_back(")");
}

static bool isGpuArray(mlir::Type type) {
  auto pyType = type.dyn_cast<plier::PyType>();
  if (!pyType)
    return false;

  auto name = pyType.getName();
  return isGpuArray(name);
}

struct MarkGpuArraysInputs
    : public mlir::PassWrapper<MarkGpuArraysInputs,
                               mlir::OperationPass<mlir::FuncOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override;
};

template <typename F>
static void visitTypeRecursive(mlir::Type type, F &&visitor) {
  if (auto tupleType = type.dyn_cast<mlir::TupleType>()) {
    for (auto t : tupleType.getTypes())
      visitTypeRecursive(t, std::forward<F>(visitor));
  } else {
    visitor(type);
  }
}

void MarkGpuArraysInputs::runOnOperation() {
  auto func = getOperation();
  auto funcType = func.getType();

  mlir::OpBuilder builder(&getContext());
  auto attrStr = builder.getStringAttr(kGpuArgAttr);
  if (func->hasAttr(attrStr)) {
    markAllAnalysesPreserved();
    return;
  }

  bool needAttr = false;
  llvm::SmallVector<bool> result;
  result.reserve(funcType.getNumInputs());

  auto visitor = [&](mlir::Type type) {
    auto res = isGpuArray(type);
    result.emplace_back(res);
    needAttr = needAttr || res;
  };

  for (auto type : (func.getType().getInputs()))
    visitTypeRecursive(type, visitor);

  if (needAttr)
    func->setAttr(attrStr, builder.getBoolArrayAttr(result));

  markAllAnalysesPreserved();
}

struct ConvertGpuArrays
    : public mlir::PassWrapper<ConvertGpuArrays,
                               mlir::OperationPass<mlir::ModuleOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override;
};

void ConvertGpuArrays::runOnOperation() {
  auto &context = getContext();

  mlir::TypeConverter typeConverter;
  // Convert unknown types to itself
  typeConverter.addConversion([](mlir::Type type) { return type; });
  populateStdTypeConverter(context, typeConverter);
  plier::populateTupleTypeConverter(context, typeConverter);
  typeConverter.addConversion(
      [&](plier::PyType type) -> llvm::Optional<mlir::Type> {
        auto name = type.getName();
        if (isGpuArray(name)) {
          auto newTypename = ("array(" + name + ")").str();
          return plier::PyType::get(type.getContext(), newTypename);
        }

        return llvm::None;
      });

  auto materializeCast = [](mlir::OpBuilder &builder, mlir::Type type,
                            mlir::ValueRange inputs,
                            mlir::Location loc) -> llvm::Optional<mlir::Value> {
    if (inputs.size() == 1)
      return builder
          .create<mlir::UnrealizedConversionCastOp>(loc, type, inputs.front())
          .getResult(0);

    return llvm::None;
  };
  typeConverter.addArgumentMaterialization(materializeCast);
  typeConverter.addSourceMaterialization(materializeCast);
  typeConverter.addTargetMaterialization(materializeCast);

  mlir::RewritePatternSet patterns(&context);
  mlir::ConversionTarget target(context);

  plier::populateTupleTypeConversionRewritesAndTarget(typeConverter, patterns,
                                                      target);

  plier::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                            patterns, target);

  target.addDynamicallyLegalOp<plier::GetItemOp, plier::SetItemOp>(
      [&](mlir::Operation *op) { return typeConverter.isLegal(op); });

  patterns.insert<
      // clang-format off
      ConvertOp<plier::GetItemOp>,
      ConvertOp<plier::SetItemOp>
      // clang-format on
      >(typeConverter, &context);

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns))))
    signalPassFailure();
}

struct LowerGpuRangePass
    : public plier::RewriteWrapperPass<LowerGpuRangePass, void, void,
                                       LowerGpuRange> {};

struct LowerPlierCalls final : public plier::CallOpLowering {
  LowerPlierCalls(mlir::MLIRContext *context)
      : CallOpLowering(context),
        resolver("numba_dpcomp.mlir.kernel_impl", "registry") {}

protected:
  virtual mlir::LogicalResult
  resolveCall(plier::PyCallOp op, mlir::StringRef name, mlir::Location loc,
              mlir::PatternRewriter &rewriter, mlir::ValueRange args,
              KWargs kwargs) const override {
    auto res = resolver.rewriteFunc(name, loc, rewriter, args, kwargs);
    if (!res)
      return mlir::failure();

    auto results = std::move(res).getValue();
    assert(results.size() == op->getNumResults());
    for (auto it : llvm::enumerate(results)) {
      auto i = it.index();
      auto r = it.value();
      auto dstType = op->getResultTypes()[i];
      if (dstType != r.getType())
        results[i] = rewriter.create<plier::CastOp>(loc, dstType, r);
    }

    rerun_std_pipeline(op);
    rewriter.replaceOp(op, results);
    return mlir::success();
  }

private:
  PyLinalgResolver resolver;
};

static mlir::LogicalResult
lowerGetGlobalId(mlir::func::CallOp op, mlir::ValueRange /*globalSizes*/,
                 mlir::ValueRange localSizes, mlir::ValueRange gridArgs,
                 mlir::ValueRange blockArgs, mlir::PatternRewriter &builder,
                 unsigned index) {
  rerun_std_pipeline(op);
  auto loc = op.getLoc();
  auto indexType = builder.getIndexType();
  auto indexCast = [&](mlir::Value val) -> mlir::Value {
    if (val.getType() != indexType)
      return builder.createOrFold<plier::CastOp>(loc, indexType, val);
    return val;
  };
  auto localSize = indexCast(localSizes[index]);
  auto gridArg = indexCast(gridArgs[index]);
  auto blockArg = indexCast(blockArgs[index]);
  mlir::Value res =
      builder.create<mlir::arith::MulIOp>(loc, gridArg, localSize);
  res = builder.create<mlir::arith::AddIOp>(loc, res, blockArg);
  auto resType = op.getResult(0).getType();
  if (res.getType() != resType)
    res = builder.createOrFold<plier::CastOp>(loc, resType, res);

  builder.replaceOp(op, res);
  return mlir::success();
}

static mlir::LogicalResult
lowerGetLocallId(mlir::func::CallOp op, mlir::ValueRange /*globalSizes*/,
                 mlir::ValueRange /*localSizes*/, mlir::ValueRange /*gridArgs*/,
                 mlir::ValueRange blockArgs, mlir::PatternRewriter &builder,
                 unsigned index) {
  rerun_std_pipeline(op);
  auto loc = op.getLoc();
  auto res = blockArgs[index];
  auto resType = op.getResult(0).getType();
  if (res.getType() != resType)
    res = builder.createOrFold<plier::CastOp>(loc, resType, res);

  builder.replaceOp(op, res);
  return mlir::success();
}

static mlir::LogicalResult lowerGetGlobalSize(mlir::func::CallOp op,
                                              mlir::ValueRange globalSizes,
                                              mlir::ValueRange /*localSizes*/,
                                              mlir::ValueRange /*gridArgs*/,
                                              mlir::ValueRange /*blockArgs*/,
                                              mlir::PatternRewriter &builder,
                                              unsigned index) {
  rerun_std_pipeline(op);
  auto loc = op.getLoc();
  auto indexType = builder.getIndexType();
  auto indexCast = [&](mlir::Value val) -> mlir::Value {
    if (val.getType() != indexType)
      return builder.createOrFold<plier::CastOp>(loc, indexType, val);
    return val;
  };
  mlir::Value res = indexCast(globalSizes[index]);
  auto resType = op.getResult(0).getType();
  if (res.getType() != resType)
    res = builder.createOrFold<plier::CastOp>(loc, resType, res);

  builder.replaceOp(op, res);
  return mlir::success();
}

static mlir::LogicalResult
lowerGetLocalSize(mlir::func::CallOp op, mlir::ValueRange /*globalSizes*/,
                  mlir::ValueRange localSizes, mlir::ValueRange /*gridArgs*/,
                  mlir::ValueRange /*blockArgs*/,
                  mlir::PatternRewriter &builder, unsigned index) {
  rerun_std_pipeline(op);
  auto loc = op.getLoc();
  auto indexType = builder.getIndexType();
  auto indexCast = [&](mlir::Value val) -> mlir::Value {
    if (val.getType() != indexType)
      return builder.createOrFold<plier::CastOp>(loc, indexType, val);
    return val;
  };
  mlir::Value res = indexCast(localSizes[index]);
  auto resType = op.getResult(0).getType();
  if (res.getType() != resType)
    res = builder.createOrFold<plier::CastOp>(loc, resType, res);

  builder.replaceOp(op, res);
  return mlir::success();
}

struct LowerBuiltinCalls : public mlir::OpRewritePattern<mlir::func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    using handler_func_t = mlir::LogicalResult (*)(
        mlir::func::CallOp, mlir::ValueRange, mlir::ValueRange,
        mlir::ValueRange, mlir::ValueRange, mlir::PatternRewriter &, unsigned);
    auto func = op->getParentOfType<mlir::FuncOp>();
    if (!func || !llvm::hasSingleElement(func.getBody()))
      return mlir::failure();

    auto kernelMarker = [&]() -> mlir::func::CallOp {
      for (auto &funcOp : func.getBody().front()) {
        auto call = mlir::dyn_cast<mlir::func::CallOp>(funcOp);
        if (call && call.getCallee() == "kernel_marker")
          return call;
      }
      return {};
    }();

    if (!kernelMarker || kernelMarker.getNumOperands() != 6)
      return mlir::failure();

    auto globalSize = kernelMarker.operands().take_front(3);
    auto localSize = kernelMarker.operands().drop_front(3);

    auto handler = [&]() -> handler_func_t {
      static const std::pair<mlir::StringRef, handler_func_t> handlers[] = {
          {"get_global_id", &lowerGetGlobalId},
          {"get_local_id", &lowerGetLocallId},
          {"get_global_size", &lowerGetGlobalSize},
          {"get_local_size", &lowerGetLocalSize},
      };
      auto name = op.getCallee();
      for (auto h : handlers)
        if (h.first == name)
          return h.second;

      return nullptr;
    }();

    if (!handler)
      return mlir::failure();

    if (op.getNumOperands() != 1 || op.getNumResults() != 1 ||
        !op.getOperand(0).getType().isa<mlir::IntegerType>() ||
        !op.getResult(0).getType().isa<mlir::IntegerType>())
      return mlir::failure();

    auto skipCasts = [](mlir::Value val) -> mlir::Value {
      auto getParent = [](mlir::Value v) -> mlir::Value {
        auto op = v.getDefiningOp();
        if (!op)
          return {};

        if (auto cast = mlir::dyn_cast<plier::SignCastOp>(op))
          return cast.value();
        if (auto cast = mlir::dyn_cast<plier::CastOp>(op))
          return cast.value();
        if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op))
          return cast.inputs()[0];

        return {};
      };
      while (auto parent = getParent(val))
        val = parent;

      return val;
    };

    auto indAttr = mlir::getConstantIntValue(skipCasts(op.operands()[0]));
    if (!indAttr)
      return mlir::failure();

    auto ind = *indAttr;
    if (ind < 0 || ind >= 3)
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 6> indexArgs;
    auto attrId = mlir::StringAttr::get(op.getContext(),
                                        plier::attributes::getGpuRangeName());
    mlir::Operation *parent = op;
    while (true) {
      parent = parent->getParentOfType<mlir::scf::ForOp>();
      if (!parent)
        break;

      if (parent->hasAttr(attrId)) {
        auto arg =
            mlir::cast<mlir::scf::ForOp>(parent).getBody()->getArgument(0);
        indexArgs.emplace_back(arg);
      }
    }

    if (indexArgs.size() != 6)
      return mlir::failure();

    std::reverse(indexArgs.begin(), indexArgs.end());
    auto gridArgs = llvm::makeArrayRef(indexArgs).take_front(3);
    auto blockArgs = llvm::makeArrayRef(indexArgs).drop_front(3);

    auto uind = static_cast<unsigned>(ind);
    return handler(op, globalSize, localSize, gridArgs, blockArgs, rewriter,
                   uind);
  }
};

struct LowerGpuBuiltinsPass
    : public plier::RewriteWrapperPass<LowerGpuBuiltinsPass, void, void,
                                       LowerPlierCalls, LowerBuiltinCalls> {};

class GpuLaunchSinkOpsPass
    : public mlir::PassWrapper<GpuLaunchSinkOpsPass,
                               mlir::OperationPass<void>> {
public:
  void runOnOperation() override {
    using namespace mlir;

    Operation *op = getOperation();
    if (op->walk([](gpu::LaunchOp launch) {
            auto isSinkingBeneficiary = [](mlir::Operation *op) -> bool {
              return isa<arith::ConstantOp, func::ConstantOp, arith::SelectOp,
                         arith::CmpIOp, arith::IndexCastOp, arith::MulIOp,
                         arith::SubIOp, arith::AddIOp>(op);
            };

            // Pull in instructions that can be sunk
            if (failed(
                    sinkOperationsIntoLaunchOp(launch, isSinkingBeneficiary)))
              return WalkResult::interrupt();

            return WalkResult::advance();
          }).wasInterrupted())
      signalPassFailure();
  }
};

struct SinkGpuDims : public mlir::OpRewritePattern<mlir::gpu::LaunchOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::LaunchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    const mlir::Value dimArgs[] = {op.gridSizeX(),  op.gridSizeY(),
                                   op.gridSizeZ(),  op.blockSizeX(),
                                   op.blockSizeY(), op.blockSizeZ()};
    llvm::SmallVector<std::pair<mlir::OpOperand *, unsigned>> uses;
    for (auto it : llvm::enumerate(dimArgs)) {
      auto i = static_cast<unsigned>(it.index());
      auto addUse = [&](mlir::OpOperand &use) {
        if (op->isProperAncestor(use.getOwner()))
          uses.emplace_back(&use, i);
      };
      auto val = it.value();
      for (auto &use : val.getUses())
        addUse(use);

      if (auto cast = val.getDefiningOp<mlir::arith::IndexCastOp>())
        for (auto &use : cast.getIn().getUses())
          addUse(use);
    }

    if (uses.empty())
      return mlir::failure();

    std::array<mlir::Value, 6> dims = {}; // TODO: static vector

    auto loc = op->getLoc();
    rewriter.setInsertionPointToStart(&op.body().front());
    auto getDim = [&](unsigned i, mlir::Type type) -> mlir::Value {
      assert(i < dims.size());
      auto dim = dims[i];
      if (!dim) {
        if (i < 3) {
          dim = rewriter.create<mlir::gpu::GridDimOp>(
              loc, static_cast<mlir::gpu::Dimension>(i));
        } else {
          dim = rewriter.create<mlir::gpu::BlockDimOp>(
              loc, static_cast<mlir::gpu::Dimension>(i - 3));
        }
        dims[i] = dim;
      }

      if (type != dim.getType())
        dim = rewriter.create<mlir::arith::IndexCastOp>(loc, type, dim);

      return dim;
    };

    for (auto it : uses) {
      auto *use = it.first;
      auto dim = it.second;
      auto owner = use->getOwner();
      rewriter.updateRootInPlace(owner, [&]() {
        auto type = use->get().getType();
        auto newVal = getDim(dim, type);
        use->set(newVal);
      });
    }

    return mlir::success();
  }
};

struct SinkGpuDimsPass : public plier::RewriteWrapperPass<SinkGpuDimsPass, void,
                                                          void, SinkGpuDims> {};

static void commonOptPasses(mlir::OpPassManager &pm) {
  pm.addPass(plier::createCommonOptsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(plier::createCommonOptsPass());
}

static void populateLowerToGPUPipelineHigh(mlir::OpPassManager &pm) {
  pm.addNestedPass<mlir::FuncOp>(std::make_unique<MarkGpuArraysInputs>());
  pm.addPass(std::make_unique<ConvertGpuArrays>());
  pm.addPass(std::make_unique<LowerGpuRangePass>());
  pm.addPass(std::make_unique<LowerGpuBuiltinsPass>());
  commonOptPasses(pm);
  pm.addPass(mlir::createSymbolDCEPass());
}

static void populateLowerToGPUPipelineLow(mlir::OpPassManager &pm) {
  auto &funcPM = pm.nest<mlir::FuncOp>();
  funcPM.addPass(std::make_unique<PrepareForGPUPass>());
  funcPM.addPass(mlir::createCanonicalizerPass());
  funcPM.addPass(std::make_unique<RemoveNestedParallelPass>());
  funcPM.addPass(std::make_unique<ParallelLoopGPUMappingPass>());
  funcPM.addPass(mlir::createParallelLoopToGpuPass());
  funcPM.addPass(std::make_unique<RemoveKernelMarkerPass>());
  funcPM.addPass(mlir::createCanonicalizerPass());
  funcPM.addPass(std::make_unique<InsertGPUAllocs>());
  funcPM.addPass(mlir::createCanonicalizerPass());
  funcPM.addPass(std::make_unique<UnstrideMemrefsPass>());
  funcPM.addPass(mlir::createLowerAffinePass());

  commonOptPasses(funcPM);
  funcPM.addPass(std::make_unique<KernelMemrefOpsMovementPass>());
  funcPM.addPass(std::make_unique<GpuLaunchSinkOpsPass>());
  funcPM.addPass(std::make_unique<SinkGpuDimsPass>());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(mlir::createSymbolDCEPass());

  pm.addNestedPass<mlir::FuncOp>(std::make_unique<GPULowerDefaultLocalSize>());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolDCEPass());

  auto &gpuFuncPM =
      pm.nest<mlir::gpu::GPUModuleOp>().nest<mlir::gpu::GPUFuncOp>();
  gpuFuncPM.addPass(mlir::arith::createArithmeticExpandOpsPass());
  gpuFuncPM.addPass(std::make_unique<FlattenScfPass>());
  commonOptPasses(gpuFuncPM);

  pm.addNestedPass<mlir::gpu::GPUModuleOp>(std::make_unique<AbiAttrsPass>());
  pm.addPass(std::make_unique<SetSPIRVCapabilitiesPass>());
  pm.addPass(std::make_unique<GPUToSpirvPass>());
  commonOptPasses(pm);

  auto &modulePM = pm.nest<mlir::spirv::ModuleOp>();
  modulePM.addPass(mlir::spirv::createLowerABIAttributesPass());
  modulePM.addPass(mlir::spirv::createUpdateVersionCapabilityExtensionPass());
  pm.addPass(std::make_unique<SerializeSPIRVPass>());
  pm.addNestedPass<mlir::FuncOp>(std::make_unique<GPUExPass>());
  commonOptPasses(pm);
  pm.addNestedPass<mlir::FuncOp>(std::make_unique<GPUExDeallocPass>());
  pm.addPass(std::make_unique<OutlineInitPass>());
  pm.addNestedPass<mlir::FuncOp>(
      std::make_unique<GenerateOutlineContextPass>());
  pm.addPass(std::make_unique<EnumerateEventsPass>());
  pm.addPass(std::make_unique<GPUToLLVMPass>());
  commonOptPasses(pm);
}
} // namespace

void registerLowerToGPUPipeline(plier::PipelineRegistry &registry) {
  registry.registerPipeline([](auto sink) {
    auto highStage = getHighLoweringStage();
    sink(lowerToGPUPipelineNameHigh(),
         {highStage.begin, plierToStdPipelineName(),
          plierToLinalgGenPipelineName()},
         {highStage.end, untuplePipelineName()}, {plierToStdPipelineName()},
         &populateLowerToGPUPipelineHigh);

    auto lowStage = getLowerLoweringStage();
    sink(lowerToGPUPipelineNameLow(), {lowStage.begin, untuplePipelineName()},
         {lowStage.end, lowerToLLVMPipelineName()}, {},
         &populateLowerToGPUPipelineLow);
  });
}

llvm::StringRef lowerToGPUPipelineNameHigh() { return "lower_to_gpu_high"; }
llvm::StringRef lowerToGPUPipelineNameLow() { return "lower_to_gpu_low"; }
