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

static const char *kGpuAllocShared = "gpu.alloc_shared";

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

static llvm::Optional<mlir::Value> getGpuStream(mlir::OpBuilder &builder,
                                                mlir::Operation *op) {
  assert(op);
  auto func = op->getParentOfType<mlir::FuncOp>();
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

    auto shared = op->hasAttr(kGpuAllocShared);

    mlir::Type token = op.asyncToken() ? op.asyncToken().getType() : nullptr;
    auto res = rewriter.replaceOpWithNewOp<gpu_runtime::GPUAllocOp>(
        op, op.getType(), token, op.asyncDependencies(), *stream,
        op.dynamicSizes(), op.symbolOperands());

    if (shared)
      res->setAttr(kGpuAllocShared, rewriter.getUnitAttr());

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

struct GPUExPass : public mlir::PassWrapper<GPUExPass, mlir::FunctionPass> {

  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns(&getContext());

    patterns.insert<ExpandLaunchOp, ExpandAllocOp, ExpandSuggestBlockSizeOp>(
        &getContext());

    (void)mlir::applyPatternsAndFoldGreedily(getFunction(),
                                             std::move(patterns));
  }
};

static void commonOptPasses(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

static void populateLowerToGPURuntimePipelineLow(mlir::OpPassManager &pm) {
  pm.addNestedPass<mlir::FuncOp>(std::make_unique<GPUExPass>());
  commonOptPasses(pm);
}

// TODO(nbpatel): Check if lowerToLLVMPipelineName is required for this pass
// since we are not lowering all the way down to llvm in this pass.
// TODO(nbpatel) : Check if a new pipeline registry is required for
// GpuRuntimeDialect
void registerLowerToGPURuntimePipeline(plier::PipelineRegistry &registry) {
  registry.registerPipeline([](auto sink) {
    auto lowStage = getLowerLoweringStage();
    sink(lowerToGPURuntimePipelineNameLow(), {lowStage.begin},
         {lowStage.end, lowerToLLVMPipelineName()}, {},
         &populateLowerToGPURuntimePipelineLow);
  });
}

llvm::StringRef lowerToGPURuntimePipelineNameLow() {
  return "lower_to_gpu_runtime_low";
}
