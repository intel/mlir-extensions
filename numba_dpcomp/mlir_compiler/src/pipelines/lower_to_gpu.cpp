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
#include <mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h>
#include <mlir/Conversion/GPUCommon/GPUCommonPass.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/SCFToGPU/SCFToGPUPass.h>
#include <mlir/Conversion/StandardToSPIRV/StandardToSPIRV.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/GPU/ParallelLoopMapper.h>
#include <mlir/Dialect/GPU/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/SPIRV/Serialization.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "base_pipeline.hpp"
#include "pipelines/lower_to_llvm.hpp"

#include "plier/compiler/pipeline_registry.hpp"
#include "plier/dialect.hpp"
#include "plier/transforms/func_utils.hpp"

namespace {
struct ParallelLoopGPUMappingPass
    : public mlir::PassWrapper<ParallelLoopGPUMappingPass, mlir::FunctionPass> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnFunction() override {
    mlir::greedilyMapParallelSCFToGPU(getFunction().getBody());
  }
};

static const char *kGpuAllocShared = "gpu.alloc_shared";

struct InsertGPUAllocs
    : public mlir::PassWrapper<InsertGPUAllocs, mlir::FunctionPass> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
  }

  void runOnFunction() override {
    auto func = getFunction();
    auto &funcBody = func.getBody();
    if (funcBody.empty()) {
      return;
    } else if (!llvm::hasSingleElement(funcBody)) {
      func.emitError("Function must have exactly one block");
      signalPassFailure();
      return;
    }

    llvm::SmallMapVector<mlir::Operation *, bool, 8> gpuBufferAllocs;
    llvm::SmallMapVector<unsigned, bool, 8> gpuBufferParams;
    auto &aliases = getAnalysis<mlir::BufferViewFlowAnalysis>();

    auto getMemref = [](mlir::Operation *op) -> mlir::Value {
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
        return load.memref();
      } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
        return store.memref();
      } else {
        op->emitError("Uhhandled mem op in gpu region");
        return nullptr;
      }
    };

    auto scfDialect = getContext().getOrLoadDialect<mlir::scf::SCFDialect>();

    if (func.walk([&](mlir::Operation *op) {
              auto memInterface =
                  mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
              if (!memInterface)
                return mlir::WalkResult::advance();

              if (!memInterface.hasEffect<mlir::MemoryEffects::Read>() &&
                  !memInterface.hasEffect<mlir::MemoryEffects::Write>())
                return mlir::WalkResult::advance();

              if (!op->getParentOfType<mlir::gpu::LaunchOp>())
                return mlir::WalkResult::advance();

              auto memref = getMemref(op);
              if (!memref)
                return mlir::WalkResult::interrupt();

              for (auto mem : aliases.resolve(memref)) {
                auto op = mem.getDefiningOp();
                if (op) {
                  if (op->getDialect() == scfDialect ||
                      mlir::isa<mlir::ViewLikeOpInterface>(op))
                    continue;

                  auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op);
                  if (!allocOp) {
                    op->emitError("Unhandled memref producer");
                    return mlir::WalkResult::interrupt();
                  }

                  gpuBufferAllocs.insert({allocOp, false});
                } else {
                  auto block = mem.getParentBlock();
                  auto blockArgs = block->getArguments();
                  auto it = llvm::find(blockArgs, mem);
                  assert(it != blockArgs.end());
                  auto index = it - blockArgs.begin();
                  gpuBufferParams.insert({static_cast<unsigned>(index), false});
                }
              }

              return mlir::WalkResult::advance();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }

    auto hasHostAccess = [&](mlir::Value memref) {
      for (auto mem : aliases.resolve(memref)) {
        for (auto user : mem.getUsers()) {
          if (mlir::isa<mlir::ReturnOp>(user))
            return true;

          if (auto memInterface =
                  mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            if (memInterface.hasEffect<mlir::MemoryEffects::Read>() &&
                memInterface.hasEffect<mlir::MemoryEffects::Write>())
              return true;
          }
        }
      }
      return false;
    };

    for (auto &it : gpuBufferAllocs) {
      auto alloc = mlir::cast<mlir::memref::AllocOp>(it.first);
      if (!it.second && hasHostAccess(alloc))
        it.second = true;
    }

    auto &block = funcBody.front();
    for (auto &it : gpuBufferParams) {
      // TODO: need a way to initial copy data from host to device-only memory.
      // Until than, threat all input buffers as shared
      it.second = true;
      //      auto param = block.getArgument(it.first);
      //      if (!it.second && hasHostAccess(param))
      //        it.second = true;
    }

    mlir::OpBuilder builder(func);
    for (auto it : gpuBufferAllocs) {
      auto alloc = mlir::cast<mlir::memref::AllocOp>(it.first);
      bool shared = it.second;
      auto loc = alloc.getLoc();
      builder.setInsertionPoint(alloc);
      auto gpuAlloc = builder.create<mlir::gpu::AllocOp>(
          loc, alloc.getType(), /*asyncToken*/ nullptr,
          /*asyncDependencies*/ llvm::None, alloc.dynamicSizes(),
          alloc.symbolOperands());
      alloc->replaceAllUsesWith(gpuAlloc);
      alloc.erase();
      if (shared)
        gpuAlloc->setAttr(kGpuAllocShared, builder.getUnitAttr());
    }

    auto term = block.getTerminator();
    assert(term);

    llvm::SmallVector<mlir::Value> dims;
    llvm::SmallPtrSet<mlir::Operation *, 8> filter;
    for (auto it : gpuBufferParams) {
      auto param = block.getArgument(it.first);
      bool shared = it.second;
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
      auto allocType = mlir::MemRefType::get(memrefType.getShape(),
                                             memrefType.getElementType(), {},
                                             memrefType.getMemorySpace());
      auto gpuAlloc = builder.create<mlir::gpu::AllocOp>(
          loc, allocType, /*asyncToken*/ nullptr,
          /*asyncDependencies*/ llvm::None, dims,
          /*symbolOperands*/ llvm::None);
      auto allocResult = gpuAlloc.getResult(0);

      if (shared) {
        gpuAlloc->setAttr(kGpuAllocShared, builder.getUnitAttr());
        auto copy =
            builder.create<mlir::memref::CopyOp>(loc, param, allocResult);
        filter.insert(copy);
      }

      if (allocType != memrefType) {
        allocResult =
            builder.create<mlir::memref::CastOp>(loc, allocResult, memrefType);
      }

      param.replaceAllUsesExcept(allocResult, filter);
      builder.setInsertionPoint(term);
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
  assert(memrefType.getAffineMaps().size() <= 1);
  if (memrefType.getAffineMaps().empty()) {
    auto shape = memrefType.getShape();
    auto expr =
        mlir::makeCanonicalStridedLayoutExpr(shape, builder.getContext());
    llvm::SmallVector<mlir::Value> applyOperands;
    applyOperands.reserve(rank * 2);
    applyOperands.assign(indices.begin(), indices.end());
    if (rank != 0) {
      mlir::OpBuilder::InsertionGuard g(builder);
      setInsertionPointToStart(builder, memref);
      for (auto i : llvm::seq(0u, rank - 1)) {
        if (shape[i] == mlir::ShapedType::kDynamicSize) {
          auto dim = builder.createOrFold<mlir::memref::DimOp>(loc, memref, i);
          applyOperands.emplace_back(dim);
        }
      }
    }
    auto affineMap = mlir::AffineMap::get(
        rank, static_cast<unsigned>(applyOperands.size()) - rank, expr);
    return builder.createOrFold<mlir::AffineApplyOp>(loc, affineMap,
                                                     applyOperands);
  } else {
    llvm::SmallVector<mlir::Value> applyOperands(rank * 2 + 1);
    {
      mlir::OpBuilder::InsertionGuard g(builder);
      setInsertionPointToStart(builder, memref);
      llvm::copy(indices, applyOperands.begin());
      applyOperands[rank] =
          builder.createOrFold<plier::ExtractMemrefMetadataOp>(loc, memref);
      auto strides = llvm::MutableArrayRef<mlir::Value>(applyOperands)
                         .drop_front(rank + 1);
      for (auto i : llvm::seq(0u, rank)) {
        strides[i] = builder.createOrFold<plier::ExtractMemrefMetadataOp>(
            loc, memref, i);
      }
    }
    auto affineMap = memrefType.getAffineMaps()[0];
    return builder.createOrFold<mlir::AffineApplyOp>(loc, affineMap,
                                                     applyOperands);
  }
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
  return !type.getAffineMaps().empty() ||
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

struct UnstrideMemrefsPass
    : public mlir::PassWrapper<UnstrideMemrefsPass, mlir::FunctionPass> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
  }

  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns(&getContext());

    patterns.insert<FlattenLoad, FlattenStore>(&getContext());

    (void)mlir::applyPatternsAndFoldGreedily(getFunction(),
                                             std::move(patterns));
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
    auto attrName = mlir::spirv::getEntryPointABIAttrName();
    const int32_t sizes[] = {1, 1, 1};
    auto abi = mlir::spirv::getEntryPointABIAttr(sizes, context);
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
        // clang-format on
    };
    //    spirv::Extension exts[] = {};
    auto triple = spirv::VerCapExtAttr::get(spirv::Version::V_1_0, caps,
                                            /*exts*/ {}, context);
    auto attr = spirv::TargetEnvAttr::get(
        triple, spirv::Vendor::Unknown, spirv::DeviceType::Unknown,
        spirv::TargetEnvAttr::kUnknownDeviceID,
        spirv::getDefaultResourceLimits(context));
    auto module = getOperation();
    module->setAttr(spirv::getTargetEnvAttrName(), attr);
  }
};

template <typename Op> static Op getOp(mlir::Region &reg) {
  auto ops = reg.getOps<Op>();
  if (llvm::hasSingleElement(ops))
    return *std::begin(ops);

  return {};
}

struct SerializeSPIRVPass
    : public mlir::PassWrapper<SerializeSPIRVPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto mod = getOperation();

    namespace gpu = mlir::gpu;
    auto gpuMod = getOp<gpu::GPUModuleOp>(mod.getRegion());
    if (!gpuMod) {
      mod.emitError() << "Invalid gpu module";
      signalPassFailure();
      return;
    }

    namespace spirv = mlir::spirv;
    auto spvMod = getOp<spirv::ModuleOp>(mod.getRegion());
    if (!spvMod) {
      mod.emitError() << "Invalid spir-v module";
      signalPassFailure();
      return;
    }

    llvm::SmallVector<uint32_t, 0> spvBinary;
    if (mlir::failed(
            spirv::serialize(spvMod, spvBinary, /*emitDebugInfo*/ false))) {
      mod.emitError() << "Failed to serialize spir-v module";
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

class ConvertLoadOp : public mlir::OpConversionPattern<mlir::memref::LoadOp> {
public:
  using mlir::OpConversionPattern<mlir::memref::LoadOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memrefType = op.memref().getType().cast<mlir::MemRefType>();
    if (!memrefType.hasRank() || memrefType.getRank() != 1)
      return mlir::failure();

    mlir::memref::LoadOp::Adaptor adaptor(operands);

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
                  llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memrefType = op.memref().getType().cast<mlir::MemRefType>();
    if (!memrefType.hasRank() || memrefType.getRank() != 1)
      return mlir::failure();

    mlir::memref::StoreOp::Adaptor adaptor(operands);

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

    mlir::SPIRVTypeConverter typeConverter(targetAttr);
    mlir::RewritePatternSet patterns(context);

    typeConverter.addConversion(
        [](mlir::MemRefType type) -> llvm::Optional<mlir::Type> {
          if (type.hasRank() && type.getRank() == 1 &&
              type.getElementType().isIntOrFloat())
            return mlir::spirv::PointerType::get(
                type.getElementType(),
                mlir::spirv::StorageClass::CrossWorkgroup);
          return mlir::Type(nullptr);
        });

    mlir::populateGPUToSPIRVPatterns(typeConverter, patterns);
    mlir::populateStandardToSPIRVPatterns(typeConverter, patterns);

    patterns.insert<ConvertLoadOp, ConvertStoreOp>(typeConverter, context);

    if (failed(
            applyFullConversion(kernelModules, *target, std::move(patterns))))
      return signalPassFailure();
  }
};

struct ExpandLaunchOp : public mlir::OpRewritePattern<mlir::gpu::LaunchFuncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::LaunchFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto gpuMod =
        mod.lookupSymbol<mlir::gpu::GPUModuleOp>(op.getKernelModuleName());
    if (!gpuMod)
      return mlir::failure();

    auto gpuKernel =
        gpuMod.lookupSymbol<mlir::gpu::GPUFuncOp>(op.getKernelName());
    if (!gpuKernel)
      return mlir::failure();

    auto stream = getGpuStream(rewriter, op);
    if (!stream)
      return mlir::failure();

    auto loc = op.getLoc();
    auto module = rewriter.create<plier::LoadGpuModuleOp>(loc, *stream, gpuMod);
    auto kernel =
        rewriter.create<plier::GetGpuKernelOp>(loc, module, gpuKernel);
    auto launch = rewriter.create<plier::LaunchGpuKernelOp>(
        loc, *stream, kernel, op.getGridSizeOperandValues(),
        op.getBlockSizeOperandValues(), op.operands());
    rewriter.create<plier::DestroyGpuKernelOp>(loc, kernel);
    rewriter.create<plier::DestroyGpuModuleOp>(loc, module);
    rewriter.replaceOp(op, launch.getResults());
    return mlir::success();
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

    mlir::Type token = op.asyncToken() ? op.asyncToken().getType() : nullptr;
    rewriter.replaceOpWithNewOp<plier::GPUAllocOp>(
        op, op.getType(), token, op.asyncDependencies(), *stream,
        op.dynamicSizes(), op.symbolOperands());
    return mlir::success();
  }
};

struct GPUExPass : public mlir::PassWrapper<GPUExPass, mlir::FunctionPass> {

  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns(&getContext());

    patterns.insert<ExpandLaunchOp, ExpandAllocOp>(&getContext());

    (void)mlir::applyPatternsAndFoldGreedily(getFunction(),
                                             std::move(patterns));
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
  mlir::Type llvmRangeType = mlir::LLVM::LLVMStructType::getLiteral(
      context, {llvmPointerType, llvmIndexType});
  mlir::Type llvmRangePointerType =
      mlir::LLVM::LLVMPointerType::get(llvmRangeType);

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
};

static const char *kEventCountAttrName = "gpu.event_count";
static const char *kEventIndexAttrName = "gpu.event_index";

class ConvertGpuStreamCreatePattern
    : public ConvertOpToGpuRuntimeCallPattern<plier::CreateGpuStreamOp> {
public:
  ConvertGpuStreamCreatePattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<plier::CreateGpuStreamOp>(converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(plier::CreateGpuStreamOp op,
                  mlir::ArrayRef<mlir::Value> /*operands*/,
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
                  mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    plier::DestroyGpuStreamOp::Adaptor adaptor(operands);
    auto loc = op.getLoc();
    auto res = streamDestroyCallBuilder.create(loc, rewriter, adaptor.source());
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuModuleLoadPattern
    : public ConvertOpToGpuRuntimeCallPattern<plier::LoadGpuModuleOp> {
public:
  ConvertGpuModuleLoadPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<plier::LoadGpuModuleOp>(converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(plier::LoadGpuModuleOp op,
                  mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    plier::LoadGpuModuleOp::Adaptor adaptor(operands);
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
    auto data = mlir::LLVM::createGlobalString(loc, rewriter, "gpu_blob", blob,
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
                  mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    plier::DestroyGpuModuleOp::Adaptor adaptor(operands);
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
                  mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    plier::GetGpuKernelOp::Adaptor adaptor(operands);
    auto loc = op.getLoc();
    llvm::SmallString<64> name = op.kernel().getLeafReference();

    auto varName = llvm::formatv("{0}_kernel_name", name).str();
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
                  mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    plier::DestroyGpuKernelOp::Adaptor adaptor(operands);
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
                  mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    plier::LaunchGpuKernelOp::Adaptor adaptor(operands,
                                              op->getAttrDictionary());
    auto eventIndex = [&]() -> int64_t {
      auto value = mlir::getConstantIntValue(op->getAttr(kEventIndexAttrName));
      if (!value)
        return -1;

      return *value;
    }();

    auto loc = op.getLoc();
    auto deps = adaptor.asyncDependencies();

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
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(1));
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

    auto eventsIndexVar = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, eventIndex));

    auto depsArrayVoidPtr = rewriter.create<mlir::LLVM::BitcastOp>(
        loc, llvmPointerPointerType, depsArrayPtr);
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
        depsArrayVoidPtr,
        paramsArrayVoidPtr,
        eventsIndexVar,
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
        plier::LaunchGpuKernelOp
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
        ConvertGpuKernelLaunchPattern
        // clang-format on
        >(converter);

    auto mod = getOperation();
    if (mlir::failed(
            mlir::applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};

static void populateLowerToGPUPipeline(mlir::OpPassManager &pm) {
  pm.addNestedPass<mlir::FuncOp>(
      std::make_unique<ParallelLoopGPUMappingPass>());
  pm.addNestedPass<mlir::FuncOp>(mlir::createParallelLoopToGpuPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::FuncOp>(std::make_unique<InsertGPUAllocs>());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::FuncOp>(std::make_unique<UnstrideMemrefsPass>());
  pm.addNestedPass<mlir::FuncOp>(mlir::createLowerAffinePass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(std::make_unique<AbiAttrsPass>());
  pm.addPass(std::make_unique<SetSPIRVCapabilitiesPass>());
  pm.addPass(std::make_unique<GPUToSpirvPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  auto &modulePM = pm.nest<mlir::spirv::ModuleOp>();
  modulePM.addPass(mlir::spirv::createLowerABIAttributesPass());
  modulePM.addPass(mlir::spirv::createUpdateVersionCapabilityExtensionPass());
  pm.addPass(std::make_unique<SerializeSPIRVPass>());
  pm.addNestedPass<mlir::FuncOp>(std::make_unique<GPUExPass>());
  pm.addPass(std::make_unique<EnumerateEventsPass>());
  pm.addPass(std::make_unique<GPUToLLVMPass>());
  pm.addPass(mlir::createCanonicalizerPass());
}
} // namespace

void registerLowerToGPUPipeline(plier::PipelineRegistry &registry) {
  registry.register_pipeline([](auto sink) {
    auto stage = getLowerLoweringStage();
    sink(lowerToGPUPipelineName(), {stage.begin},
         {stage.end, lowerToLLVMPipelineName()}, {},
         &populateLowerToGPUPipeline);
  });
}

llvm::StringRef lowerToGPUPipelineName() { return "lower_to_gpu"; }
