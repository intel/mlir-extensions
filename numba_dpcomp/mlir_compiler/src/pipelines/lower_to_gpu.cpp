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

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h>
#include <mlir/Conversion/GPUCommon/GPUCommonPass.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/SCFToGPU/SCFToGPUPass.h>
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
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/SPIRV/Serialization.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "base_pipeline.hpp"
#include "pipelines/lower_to_llvm.hpp"

#include "plier/compiler/pipeline_registry.hpp"
#include "plier/dialect.hpp"

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
    auto caps = {spirv::Capability::Shader};
    auto exts = {spirv::Extension::SPV_KHR_storage_buffer_storage_class};
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
    spvBinary.insert(spvBinary.begin(),
                     static_cast<uint32_t>(spvBinary.size())); // size

    auto spvData =
        llvm::StringRef(reinterpret_cast<const char *>(spvBinary.data()),
                        spvBinary.size() * sizeof(uint32_t));
    auto spvAttr = mlir::StringAttr::get(&getContext(), spvData);
    gpuMod->setAttr(gpu::getDefaultGpuBinaryAnnotation(), spvAttr);
    spvMod->erase();
  }
};

struct GPUToLLVMPass
    : public mlir::PassWrapper<GPUToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    mlir::LLVMTypeConverter converter(&getContext());
    mlir::RewritePatternSet patterns(&getContext());
    mlir::LLVMConversionTarget target(getContext());

    target.addIllegalDialect<mlir::gpu::GPUDialect>();

    mlir::populateAsyncStructuralTypeConversionsAndLegality(converter, patterns,
                                                            target);
    mlir::populateGpuToLLVMConversionPatterns(
        converter, patterns, mlir::gpu::getDefaultGpuBinaryAnnotation());

    auto mod = getOperation();
    if (mlir::failed(
            mlir::applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();

    // TODO: populateGpuToLLVMConversionPatterns can generate call with invalid
    // types, fix upstream.
    mlir::OpBuilder builder(&getContext());
    llvm::SmallVector<mlir::Value> args;
    if (mod.walk([&](mlir::LLVM::CallOp call) {
             auto funcName = call.callee();
             if (!funcName) {
               mod.emitError() << "Failed to get llvm function";
               return mlir::WalkResult::interrupt();
             }

             auto func = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(*funcName);
             if (!func) {
               mod.emitError() << "Failed to get llvm function";
               return mlir::WalkResult::interrupt();
             }

             auto funcType = func.getType();

             if (call.getNumOperands() != funcType.getNumParams()) {
               mod.emitError() << "Invalid LLVM::CallOp operands count";
               return mlir::WalkResult::interrupt();
             }

             bool needFix = false;
             args.resize(funcType.getNumParams());
             builder.setInsertionPoint(call);

             for (auto it : llvm::enumerate(
                      llvm::zip(call.getOperands(), funcType.getParams()))) {
               auto i = it.index();
               auto operand = std::get<0>(it.value());
               auto type = std::get<1>(it.value());
               if (operand.getType() != type) {
                 args[i] = builder
                               .create<mlir::UnrealizedConversionCastOp>(
                                   call.getLoc(), type, operand)
                               .getResult(0);
                 needFix = true;
               } else {
                 args[i] = operand;
               }
             }

             if (needFix)
               call->setOperands(args);

             return mlir::WalkResult::advance();
           })
            .wasInterrupted())
      signalPassFailure();
  }
};

static void populateLowerToGPUPipeline(mlir::OpPassManager &pm) {
  pm.addNestedPass<mlir::FuncOp>(
      std::make_unique<ParallelLoopGPUMappingPass>());
  pm.addNestedPass<mlir::FuncOp>(mlir::createParallelLoopToGpuPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::FuncOp>(std::make_unique<UnstrideMemrefsPass>());
  pm.addNestedPass<mlir::FuncOp>(mlir::createLowerAffinePass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(std::make_unique<AbiAttrsPass>());
  pm.addPass(std::make_unique<SetSPIRVCapabilitiesPass>());
  pm.addPass(mlir::createConvertGPUToSPIRVPass());
  pm.addPass(mlir::createCanonicalizerPass());
  auto &modulePM = pm.nest<mlir::spirv::ModuleOp>();
  modulePM.addPass(mlir::spirv::createLowerABIAttributesPass());
  modulePM.addPass(mlir::spirv::createUpdateVersionCapabilityExtensionPass());
  pm.addPass(std::make_unique<SerializeSPIRVPass>());
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
