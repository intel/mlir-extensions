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
    : public mlir::PassWrapper<GPUExDeallocPass, mlir::FunctionPass> {

  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns(&getContext());

    patterns.insert<CreateDeallocOp<gpu_runtime::LoadGpuModuleOp,
                                    gpu_runtime::DestroyGpuModuleOp>,
                    CreateDeallocOp<gpu_runtime::GetGpuKernelOp,
                                    gpu_runtime::DestroyGpuKernelOp>>(
        &getContext());

    (void)mlir::applyPatternsAndFoldGreedily(getFunction(),
                                             std::move(patterns));
  }
};

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

struct GPUExPass : public mlir::PassWrapper<GPUExPass, mlir::FunctionPass> {

  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns(&getContext());

    patterns.insert<ExpandLaunchOp, ExpandAllocOp, ExpandSuggestBlockSizeOp>(
        &getContext());

    (void)mlir::applyPatternsAndFoldGreedily(getFunction(),
                                             std::move(patterns));
  }
};

// Expose the passes to the outside world
std::unique_ptr<mlir::Pass> gpu_runtime::runSerializeSPIRVPass() {
  return std::make_unique<SerializeSPIRVPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::runGPUExPass() {
  return std::make_unique<GPUExPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::runGPUExDeallocPass() {
  return std::make_unique<GPUExDeallocPass>();
}