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

#include "pipelines/LowerToGpu.hpp"

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h>
#include <mlir/Conversion/GPUCommon/GPUCommonPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/SCFToGPU/SCFToGPUPass.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/Transforms/Passes.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/GPU/Transforms/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include "BasePipeline.hpp"
#include "CheckGpuCaps.hpp"
#include "LoopUtils.hpp"
#include "PyLinalgResolver.hpp"
#include "pipelines/LowerToLlvm.hpp"
#include "pipelines/PlierToLinalg.hpp"
#include "pipelines/PlierToStd.hpp"
#include "pipelines/PreLowSimplifications.hpp"

#include "imex/Compiler/PipelineRegistry.hpp"
#include "imex/Conversion/GpuRuntimeToLlvm.hpp"
#include "imex/Conversion/GpuToGpuRuntime.hpp"
#include "imex/Conversion/UtilConversion.hpp"
#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOps.hpp"
#include "imex/Dialect/gpu_runtime/Transforms/MakeBarriersUniform.hpp"
#include "imex/Dialect/imex_util/Dialect.hpp"
#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"
#include "imex/Transforms/CallLowering.hpp"
#include "imex/Transforms/CastUtils.hpp"
#include "imex/Transforms/CommonOpts.hpp"
#include "imex/Transforms/PipelineUtils.hpp"
#include "imex/Transforms/RewriteWrapper.hpp"
#include "imex/Transforms/TypeConversion.hpp"

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

static bool isGpuRegion(imex::util::EnvironmentRegionOp op) {
  return op.getEnvironment().isa<gpu_runtime::GPURegionDescAttr>();
}

struct PrepareForGPUPass
    : public mlir::PassWrapper<PrepareForGPUPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareForGPUPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<gpu_runtime::GpuRuntimeDialect>();
    registry.insert<imex::util::ImexUtilDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    getOperation()->walk([](imex::util::EnvironmentRegionOp envOp) {
      if (!isGpuRegion(envOp))
        return;

      for (auto &op : envOp.getRegion().front()) {
        if (auto parallel = mlir::dyn_cast<mlir::scf::ParallelOp>(op))
          moveOpsIntoParallel(parallel);
      }
    });
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
    : public imex::RewriteWrapperPass<RemoveNestedParallelPass, void, void,
                                      RemoveNestedParallel> {};

struct RemoveGpuRegion
    : public mlir::OpRewritePattern<imex::util::EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::EnvironmentRegionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isGpuRegion(op))
      return mlir::failure();

    imex::util::EnvironmentRegionOp::inlineIntoParent(rewriter, op);
    return mlir::success();
  }
};

struct RemoveGpuRegionPass
    : public imex::RewriteWrapperPass<RemoveGpuRegionPass, void, void,
                                      RemoveGpuRegion> {};

struct KernelMemrefOpsMovementPass
    : public mlir::PassWrapper<KernelMemrefOpsMovementPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KernelMemrefOpsMovementPass)

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
      launch.getBody().walk([&](mlir::Operation *op) {
        if (!mlir::isa<mlir::memref::DimOp,
                       imex::util::ExtractMemrefMetadataOp>(op))
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

struct AssumeGpuIdRangePass
    : public mlir::PassWrapper<AssumeGpuIdRangePass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssumeGpuIdRangePass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::cf::ControlFlowDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto op = getOperation();

    mlir::OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(&op->getRegion(0).front());
    auto maxInt =
        builder
            .create<mlir::arith::ConstantIndexOp>(
                builder.getUnknownLoc(),
                static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1)
            .getResult();

    op->walk([&](mlir::Operation *nestedOp) {
      if (!mlir::isa<mlir::gpu::ThreadIdOp, mlir::gpu::BlockIdOp,
                     mlir::gpu::GlobalIdOp>(nestedOp))
        return;

      assert(nestedOp->getNumResults() == 1);
      auto res = nestedOp->getResult(0);
      assert(res.getType().isa<mlir::IndexType>());
      builder.setInsertionPointAfter(nestedOp);
      auto loc = op->getLoc();
      auto cmp = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::slt, res, maxInt);
      builder.create<mlir::cf::AssertOp>(loc, cmp, "Invalid gpu id range");
    });
  }
};

struct GPULowerDefaultLocalSize
    : public mlir::PassWrapper<GPULowerDefaultLocalSize,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPULowerDefaultLocalSize)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<imex::util::ImexUtilDialect>();
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

    llvm::StringRef funcName("set_default_local_size");
    mlir::func::CallOp setDefSize;
    for (auto op : region.front().getOps<mlir::func::CallOp>()) {
      if (op.getCallee() == funcName && op->getNumOperands() == 3) {
        setDefSize = op;
        break;
      }
    }

    mlir::DominanceInfo dom;
    mlir::OpBuilder builder(&getContext());
    func.walk([&](mlir::gpu::LaunchFuncOp op) {
      auto bx = op.getBlockSizeX();
      if (auto call = bx.getDefiningOp<gpu_runtime::GPUSuggestBlockSizeOp>()) {
        if (call.getKernel())
          return;

        auto loc = call.getLoc();
        auto kernel = op.getKernel();
        builder.setInsertionPoint(call);

        mlir::ValueRange operands = call.getGridSize();
        if (setDefSize && dom.properlyDominates(setDefSize, call))
          operands = setDefSize.getArgOperands();

        auto count = static_cast<unsigned>(operands.size());
        llvm::SmallVector<mlir::Value, 3> globalSize(count);
        for (auto i : llvm::seq(0u, count))
          globalSize[i] = imex::indexCast(builder, loc, operands[i]);

        auto res = builder
                       .create<gpu_runtime::GPUSuggestBlockSizeOp>(
                           loc, /*stream*/ llvm::None, globalSize, kernel)
                       .getResults();

        for (auto i : llvm::seq(0u, count)) {
          auto castedRes = imex::indexCast(builder, loc, res[i],
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
        getContext()->getOrLoadDialect<mlir::arith::ArithDialect>();
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

struct FlattenScfPass : public imex::RewriteWrapperPass<FlattenScfPass, void,
                                                        void, FlattenScfIf> {};

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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUExDeallocPass)

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<CreateDeallocOp<gpu_runtime::LoadGpuModuleOp,
                                    gpu_runtime::DestroyGpuModuleOp>,
                    CreateDeallocOp<gpu_runtime::GetGpuKernelOp,
                                    gpu_runtime::DestroyGpuKernelOp>>(ctx);

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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OutlineInitPass)

  void runOnOperation() override {
    auto mod = getOperation();

    using outline_func_t =
        bool (*)(mlir::Operation &, llvm::SmallVectorImpl<mlir::Operation *> &);
    const outline_func_t outlineHandlers[] = {
        &outlineOp<gpu_runtime::CreateGpuStreamOp,
                   gpu_runtime::DestroyGpuStreamOp>,
        &outlineOp<gpu_runtime::LoadGpuModuleOp,
                   gpu_runtime::DestroyGpuModuleOp>,
        &outlineOp<gpu_runtime::GetGpuKernelOp,
                   gpu_runtime::DestroyGpuKernelOp>,
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
    for (auto func : mod.getOps<mlir::func::FuncOp>()) {
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
        auto func = builder.create<mlir::func::FuncOp>(
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
        auto func = builder.create<mlir::func::FuncOp>(
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
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateOutlineContextPass)

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
        if (call->getNumResults() != 0) {
          call.emitError("deinit function mus have zero results");
          signalPassFailure();
          return;
        }

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
    auto takeCtx = builder.create<imex::util::TakeContextOp>(
        init->getLoc(), initSym, deinitSym, init.getResultTypes());
    auto ctx = takeCtx.getContext();
    auto resValues = takeCtx.getResults();
    init->replaceAllUsesWith(resValues);
    init->erase();

    if (deinit) {
      builder.setInsertionPoint(deinit);
      builder.create<imex::util::ReleaseContextOp>(deinit->getLoc(), ctx);
      deinit->erase();
    } else {
      builder.setInsertionPoint(body.front().getTerminator());
      builder.create<imex::util::ReleaseContextOp>(builder.getUnknownLoc(),
                                                   ctx);
    }
  }
};

static void rerunStdPipeline(mlir::Operation *op) {
  assert(nullptr != op);
  auto marker =
      mlir::StringAttr::get(op->getContext(), plierToStdPipelineName());
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  assert(nullptr != mod);
  imex::addPipelineJumpMarker(mod, marker);
}

static mlir::FailureOr<mlir::StringAttr>
getDeviceDescFromFunc(mlir::MLIRContext *context, mlir::TypeRange argTypes) {
  mlir::StringAttr res;
  for (auto arg : argTypes) {
    auto tensor = arg.dyn_cast<imex::ntensor::NTensorType>();
    if (!tensor)
      continue;

    auto env = tensor.getEnvironment()
                   .dyn_cast_or_null<gpu_runtime::GPURegionDescAttr>();
    if (!env)
      continue;

    auto name = env.getDevice();
    assert(name && "Invalid device name");
    if (!res) {
      res = name;
    } else if (res != name) {
      return mlir::failure();
    }
  }

  // TODO: remove default device.
  if (!res)
    if (auto dev = getDefaultDevice())
      res = mlir::StringAttr::get(context, *dev);

  return res;
}

struct LowerGpuRange final : public imex::CallOpLowering {
  using CallOpLowering::CallOpLowering;

protected:
  virtual mlir::LogicalResult
  resolveCall(plier::PyCallOp op, mlir::StringRef name, mlir::Location /*loc*/,
              mlir::PatternRewriter &rewriter, mlir::ValueRange args,
              KWargs kwargs) const override {
    if (name != "_gpu_range")
      return mlir::failure();

    auto parent = op->getParentOfType<mlir::FunctionOpInterface>();
    if (!parent)
      return mlir::failure();

    auto device =
        getDeviceDescFromFunc(op->getContext(), parent.getArgumentTypes());

    if (mlir::failed(device))
      return mlir::failure();

    llvm::SmallVector<mlir::scf::ForOp> newOps;
    auto setAttr = [&](mlir::scf::ForOp op) {
      auto unitAttr = mlir::UnitAttr::get(op->getContext());
      op->setAttr(imex::util::attributes::getParallelName(), unitAttr);
      newOps.emplace_back(op);
    };
    if (mlir::failed(imex::lowerRange(op, args, kwargs, rewriter, setAttr)))
      return mlir::failure();

    mlir::OpBuilder::InsertionGuard g(rewriter);
    auto envAttr = gpu_runtime::GPURegionDescAttr::get(getContext(), *device);
    for (auto op : newOps) {
      auto bodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc) {
        auto newOp = builder.clone(*op);
        builder.create<imex::util::EnvironmentRegionYieldOp>(
            loc, newOp->getResults());
      };
      auto opResults = op.getResults();
      rewriter.setInsertionPoint(op);
      auto newOp = rewriter.create<imex::util::EnvironmentRegionOp>(
          op->getLoc(), envAttr, /*args*/ llvm::None, opResults.getTypes(),
          bodyBuilder);
      rewriter.replaceOp(op, newOp->getResults());
    }

    rerunStdPipeline(parent);
    return mlir::success();
  }
};

static bool isGpuArray(mlir::Type type) {
  auto tensor = type.dyn_cast<imex::ntensor::NTensorType>();
  if (!tensor)
    return false;

  auto env = tensor.getEnvironment();
  if (!env)
    return false;

  return env.isa<gpu_runtime::GPURegionDescAttr>();
}

struct MarkGpuArraysInputs
    : public mlir::PassWrapper<MarkGpuArraysInputs,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkGpuArraysInputs)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<gpu_runtime::GpuRuntimeDialect>();
    registry.insert<imex::ntensor::NTensorDialect>();
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
  auto funcType = func.getFunctionType();

  mlir::OpBuilder builder(&getContext());
  auto attrStr = builder.getStringAttr(gpu_runtime::getGpuAccessibleAttrName());
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

  for (auto type : (func.getFunctionType().getInputs()))
    visitTypeRecursive(type, visitor);

  if (needAttr)
    func->setAttr(attrStr, builder.getBoolArrayAttr(result));

  markAllAnalysesPreserved();
}

struct LowerGpuRangePass
    : public mlir::PassWrapper<LowerGpuRangePass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGpuRangePass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<gpu_runtime::GpuRuntimeDialect>();
  }

  void runOnOperation() override {
    auto context = &getContext();
    mlir::RewritePatternSet patterns(context);

    patterns.insert<LowerGpuRange>(context);

    imex::util::EnvironmentRegionOp::getCanonicalizationPatterns(patterns,
                                                                 context);

    auto op = getOperation();
    (void)mlir::applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

struct LowerPlierCalls final : public imex::CallOpLowering {
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

    auto results = std::move(res).value();
    assert(results.size() == op->getNumResults());
    for (auto it : llvm::enumerate(results)) {
      auto i = it.index();
      auto r = it.value();
      auto dstType = op->getResultTypes()[i];
      if (dstType != r.getType())
        results[i] = rewriter.create<plier::CastOp>(loc, dstType, r);
    }

    rerunStdPipeline(op);
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
    res = builder.createOrFold<mlir::arith::IndexCastOp>(loc, resType, res);

  builder.replaceOp(op, res);
  return mlir::success();
}

static mlir::LogicalResult
lowerGetLocallId(mlir::func::CallOp op, mlir::ValueRange /*globalSizes*/,
                 mlir::ValueRange /*localSizes*/, mlir::ValueRange /*gridArgs*/,
                 mlir::ValueRange blockArgs, mlir::PatternRewriter &builder,
                 unsigned index) {
  auto loc = op.getLoc();
  auto res = blockArgs[index];
  auto resType = op.getResult(0).getType();
  if (res.getType() != resType)
    res = builder.createOrFold<mlir::arith::IndexCastOp>(loc, resType, res);

  builder.replaceOp(op, res);
  return mlir::success();
}

static mlir::LogicalResult
lowerGetGlobalSize(mlir::func::CallOp op, mlir::ValueRange globalSizes,
                   mlir::ValueRange localSizes, mlir::ValueRange /*gridArgs*/,
                   mlir::ValueRange /*blockArgs*/,
                   mlir::PatternRewriter &builder, unsigned index) {
  auto loc = op.getLoc();
  auto indexType = builder.getIndexType();
  auto indexCast = [&](mlir::Value val) -> mlir::Value {
    if (val.getType() != indexType)
      return builder.createOrFold<mlir::arith::IndexCastOp>(loc, indexType,
                                                            val);
    return val;
  };
  mlir::Value global = indexCast(globalSizes[index]);
  mlir::Value local = indexCast(localSizes[index]);
  mlir::Value res = builder.create<mlir::arith::MulIOp>(loc, global, local);
  auto resType = op.getResult(0).getType();
  if (res.getType() != resType)
    res = builder.createOrFold<mlir::arith::IndexCastOp>(loc, resType, res);

  builder.replaceOp(op, res);
  return mlir::success();
}

static mlir::LogicalResult
lowerGetLocalSize(mlir::func::CallOp op, mlir::ValueRange /*globalSizes*/,
                  mlir::ValueRange localSizes, mlir::ValueRange /*gridArgs*/,
                  mlir::ValueRange /*blockArgs*/,
                  mlir::PatternRewriter &builder, unsigned index) {
  auto loc = op.getLoc();
  auto indexType = builder.getIndexType();
  auto indexCast = [&](mlir::Value val) -> mlir::Value {
    if (val.getType() != indexType)
      return builder.createOrFold<mlir::arith::IndexCastOp>(loc, indexType,
                                                            val);
    return val;
  };
  mlir::Value res = indexCast(localSizes[index]);
  auto resType = op.getResult(0).getType();
  if (res.getType() != resType)
    res = builder.createOrFold<mlir::arith::IndexCastOp>(loc, resType, res);

  builder.replaceOp(op, res);
  return mlir::success();
}

static std::array<mlir::Value, 3>
dim3ToArray(const mlir::gpu::KernelDim3 &val) {
  return {val.x, val.y, val.z};
}

struct LowerBuiltinCalls : public mlir::OpRewritePattern<mlir::func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    using handler_func_t = mlir::LogicalResult (*)(
        mlir::func::CallOp, mlir::ValueRange, mlir::ValueRange,
        mlir::ValueRange, mlir::ValueRange, mlir::PatternRewriter &, unsigned);
    auto launch = op->getParentOfType<mlir::gpu::LaunchOp>();
    if (!launch)
      return mlir::failure();

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

    auto indAttr = mlir::getConstantIntValue(op.operands()[0]);
    if (!indAttr)
      return mlir::failure();

    auto ind = *indAttr;
    if (ind < 0 || ind >= 3)
      return mlir::failure();

    auto globalSize = dim3ToArray(launch.getGridSize());
    auto localSize = dim3ToArray(launch.getBlockSize());

    auto globalArgs = dim3ToArray(launch.getBlockIds());
    auto localArgs = dim3ToArray(launch.getThreadIds());

    auto uind = static_cast<unsigned>(ind);
    return handler(op, globalSize, localSize, globalArgs, localArgs, rewriter,
                   uind);
  }
};

struct LowerGpuBuiltinsPass
    : public imex::RewriteWrapperPass<LowerGpuBuiltinsPass, void, void,
                                      LowerPlierCalls> {};

static llvm::Optional<gpu_runtime::FenceFlags>
getFenceFlags(mlir::OpFoldResult arg) {
  auto val = mlir::getConstantIntValue(arg);
  if (!val)
    return llvm::None;

  auto v = *val;
  if (v == 1)
    return gpu_runtime::FenceFlags::local;

  if (v == 2)
    return gpu_runtime::FenceFlags::global;

  return llvm::None;
}

template <typename Op>
static void genBarrierOp(mlir::Operation *srcOp,
                         mlir::PatternRewriter &rewriter,
                         gpu_runtime::FenceFlags flags) {
  rewriter.create<Op>(srcOp->getLoc(), flags);

  // TODO: remove
  assert(srcOp->getNumResults() == 1);
  auto retType = srcOp->getResult(0).getType();
  rewriter.replaceOpWithNewOp<imex::util::UndefOp>(srcOp, retType);
}

class ConvertBarrierOps : public mlir::OpRewritePattern<mlir::func::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<mlir::gpu::LaunchOp>())
      return mlir::failure();

    auto operands = op.operands();
    if (operands.size() != 1)
      return mlir::failure();

    if (op.getNumResults() != 1)
      return mlir::failure();

    auto fenceFlags = getFenceFlags(operands[0]);
    if (!fenceFlags)
      return mlir::failure();

    using funcptr_t = void (*)(mlir::Operation *, mlir::PatternRewriter &,
                               gpu_runtime::FenceFlags);
    const std::pair<llvm::StringRef, funcptr_t> handlers[] = {
        {"kernel_barrier", &genBarrierOp<gpu_runtime::GPUBarrierOp>},
        {"kernel_mem_fence", &genBarrierOp<gpu_runtime::GPUMemFenceOp>},
    };

    auto funcName = op.getCallee();
    for (auto &h : handlers) {
      if (h.first == funcName) {
        h.second(op, rewriter, *fenceFlags);
        return mlir::success();
      }
    }

    return mlir::failure();
  }
};

template <mlir::gpu::AllReduceOperation ReduceType>
static void genGroupOp(mlir::Operation *srcOp, mlir::PatternRewriter &rewriter,
                       mlir::Value arg) {
  auto ctx = srcOp->getContext();
  auto reduceAttr = mlir::gpu::AllReduceOperationAttr::get(ctx, ReduceType);
  rewriter.replaceOpWithNewOp<mlir::gpu::AllReduceOp>(srcOp, arg, reduceAttr);
}

class ConvertGroupOps : public mlir::OpRewritePattern<mlir::func::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<mlir::gpu::LaunchOp>())
      return mlir::failure();

    auto operands = op.operands();
    if (operands.size() != 1)
      return mlir::failure();

    if (op.getNumResults() != 1)
      return mlir::failure();

    auto src = operands[0];
    auto srcType = src.getType();

    if (srcType != op.getResult(0).getType())
      return mlir::failure();

    auto funcName = op.getCallee();
    if (!funcName.consume_front("group_"))
      return mlir::failure();

    using funcptr_t =
        void (*)(mlir::Operation *, mlir::PatternRewriter &, mlir::Value);
    const std::pair<llvm::StringRef, funcptr_t> handlers[] = {
        {"reduce_add", &genGroupOp<mlir::gpu::AllReduceOperation::ADD>},
    };

    for (auto &h : handlers) {
      if (funcName.startswith(h.first)) {
        h.second(op, rewriter, src);
        return mlir::success();
      }
    }

    return mlir::failure();
  }
};

template <typename SpvOp>
static mlir::Value reduceOp(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value val1, mlir::Value val2) {
  return builder.create<SpvOp>(loc, val1, val2);
}

using ReduceFuncType = mlir::Value (*)(mlir::OpBuilder &, mlir::Location,
                                       mlir::Value, mlir::Value);
static ReduceFuncType getReduceFunc(mlir::gpu::AllReduceOperation op,
                                    bool isFloat) {
  using ReduceOp = mlir::gpu::AllReduceOperation;
  using HandlerType = std::tuple<ReduceOp, ReduceFuncType, ReduceFuncType>;
  const HandlerType handers[] = {{ReduceOp::ADD, &reduceOp<mlir::arith::AddIOp>,
                                  &reduceOp<mlir::arith::AddFOp>}};
  for (auto handler : handers) {
    if (std::get<0>(handler) == op)
      return isFloat ? std::get<2>(handler) : std::get<1>(handler);
  }
  return nullptr;
}

class ConvertGroupOpsToSubgroup
    : public mlir::OpRewritePattern<mlir::gpu::AllReduceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::AllReduceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto launchOp = op->getParentOfType<mlir::gpu::LaunchOp>();
    if (!launchOp)
      return mlir::failure();

    if (!op.getOp())
      return mlir::failure();

    if (!op.getType().isIntOrFloat())
      return mlir::failure();

    auto isFloat = op.getType().isa<mlir::FloatType>();
    auto reduceFunc = getReduceFunc(*op.getOp(), isFloat);

    if (!reduceFunc)
      return mlir::failure();

    mlir::Value groupBuffer;
    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(launchOp);
      auto loc = launchOp->getLoc();
      mlir::Value size = launchOp.getBlockSizeX();
      size = rewriter.create<mlir::arith::MulIOp>(loc, size,
                                                  launchOp.getBlockSizeY());
      size = rewriter.create<mlir::arith::MulIOp>(loc, size,
                                                  launchOp.getBlockSizeZ());

      // TODO: Subgroup size is hardcoded for now.
      mlir::Value subgroupSize =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, 8);

      mlir::Value numSubgroups =
          rewriter.create<mlir::arith::CeilDivSIOp>(loc, size, subgroupSize);

      auto elemType = op.getType();

      // TODO: Fix storage class handling upstream
      //      auto storageClass = gpu_runtime::StorageClassAttr::get(
      //          getContext(), gpu_runtime::StorageClass::local);
      auto storageClass = rewriter.getI64IntegerAttr(
          mlir::gpu::GPUDialect::getPrivateAddressSpace());
      auto memrefType = mlir::MemRefType::get(mlir::ShapedType::kDynamicSize,
                                              elemType, nullptr, storageClass);
      groupBuffer = rewriter
                        .create<mlir::gpu::AllocOp>(
                            loc, memrefType, /*asyncToken*/ mlir::Type(),
                            /*asyncDependencies*/ llvm::None, numSubgroups,
                            /*symbolOperands*/ llvm::None)
                        .getMemref();
      rewriter.setInsertionPointAfter(launchOp);
      rewriter.create<mlir::gpu::DeallocOp>(loc, /*asyncToken*/ mlir::Type(),
                                            /*asyncDependencies*/ llvm::None,
                                            groupBuffer);
    }

    mlir::Value subgroupId = [&]() {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&launchOp.getBody().front());
      return rewriter.create<mlir::gpu::SubgroupIdOp>(rewriter.getUnknownLoc());
    }();

    auto loc = op->getLoc();
    auto reduceType = *op.getOp();
    mlir::Value sgResult = rewriter.create<mlir::gpu::SubgroupReduceOp>(
        loc, op.getValue(), reduceType);
    rewriter.create<mlir::memref::StoreOp>(loc, sgResult, groupBuffer,
                                           subgroupId);

    rewriter.create<gpu_runtime::GPUBarrierOp>(loc,
                                               gpu_runtime::FenceFlags::local);

    mlir::Value numSubgroups = [&]() {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&launchOp.getBody().front());
      return rewriter.create<mlir::gpu::NumSubgroupsOp>(
          rewriter.getUnknownLoc());
    }();

    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value isFirstSg = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, subgroupId, zero);

    auto ifBodyBuilder = [&](mlir::OpBuilder &ifBuilder, mlir::Location ifLoc) {
      mlir::Value init =
          ifBuilder.create<mlir::memref::LoadOp>(ifLoc, groupBuffer, zero);

      auto forBodyBuilder = [&](mlir::OpBuilder &forBuilder,
                                mlir::Location forLoc, mlir::Value i,
                                mlir::ValueRange args) {
        assert(args.size() == 1);
        auto prev = args.front();
        mlir::Value val =
            forBuilder.create<mlir::memref::LoadOp>(forLoc, groupBuffer, i);

        mlir::Value res = reduceFunc(forBuilder, forLoc, prev, val);
        forBuilder.create<mlir::scf::YieldOp>(forLoc, res);
      };

      mlir::Value res = ifBuilder
                            .create<mlir::scf::ForOp>(ifLoc, one, numSubgroups,
                                                      one, init, forBodyBuilder)
                            .getResult(0);
      mlir::Value isSingleSg = ifBuilder.create<mlir::arith::CmpIOp>(
          ifLoc, mlir::arith::CmpIPredicate::eq, numSubgroups, one);
      res =
          ifBuilder.create<mlir::arith::SelectOp>(ifLoc, isSingleSg, init, res);
      ifBuilder.create<mlir::memref::StoreOp>(ifLoc, res, groupBuffer, zero);
      ifBuilder.create<mlir::scf::YieldOp>(ifLoc);
    };

    rewriter.create<mlir::scf::IfOp>(loc, /*resultTypes*/ llvm::None, isFirstSg,
                                     ifBodyBuilder);

    rewriter.create<gpu_runtime::GPUBarrierOp>(loc,
                                               gpu_runtime::FenceFlags::local);

    mlir::Value result =
        rewriter.create<mlir::memref::LoadOp>(loc, groupBuffer, zero);
    rewriter.replaceOp(op, result);
    return mlir::failure();
  }
};

struct LowerGpuBuiltins2Pass
    : public imex::RewriteWrapperPass<
          LowerGpuBuiltins2Pass, void, void, ConvertBarrierOps, ConvertGroupOps,
          ConvertGroupOpsToSubgroup, LowerBuiltinCalls> {};

class ConvertArrayAllocOps : public mlir::OpRewritePattern<mlir::func::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto name = op.getCallee();
    if (!name.startswith("local_array_"))
      return mlir::failure();

    if (op->getNumResults() != 1)
      return mlir::failure();

    auto mod = op->getParentOfType<mlir::gpu::GPUModuleOp>();
    if (!mod)
      return mlir::failure();

    auto oldType = op->getResult(0).getType().dyn_cast<mlir::MemRefType>();
    if (!oldType)
      return mlir::failure();

    auto operands = op.operands();
    auto operandsCount = static_cast<unsigned>(operands.size());
    if (operandsCount != static_cast<unsigned>(oldType.getRank()))
      return mlir::failure();

    llvm::SmallVector<int64_t> shape(operandsCount);
    for (auto i : llvm::seq(0u, operandsCount)) {
      auto val = mlir::getConstantIntValue(operands[i]);
      if (!val)
        return mlir::failure();

      shape[i] = *val;
    }

    auto type = mlir::MemRefType::get(shape, oldType.getElementType());

    // TODO: Fix storage class upstream
    //    auto storageClass = gpu_runtime::StorageClassAttr::get(
    //        getContext(), gpu_runtime::StorageClass::local);
    auto storageClass = rewriter.getI64IntegerAttr(
        mlir::gpu::GPUDialect::getPrivateAddressSpace());
    auto typeLocal = mlir::MemRefType::get(shape, type.getElementType(),
                                           nullptr, storageClass);

    auto global = [&]() -> mlir::StringRef {
      auto *block = mod.getBody();
      llvm::SmallString<64> name;
      for (unsigned i = 0;; ++i) {
        if (i == 0) {
          name = "__local_array";
        } else {
          name.clear();
          (llvm::Twine("__local_array") + llvm::Twine(i)).toVector(name);
        }
        if (!mod.lookupSymbol(name))
          break;
      }
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(block);
      auto loc = rewriter.getUnknownLoc();
      auto global = rewriter.create<mlir::memref::GlobalOp>(
          loc, name,
          /*sym_visibility=*/rewriter.getStringAttr("private"),
          /*type=*/typeLocal,
          /*initial_value=*/nullptr,
          /*constant=*/false,
          /*alignment=*/nullptr);
      return global.getSymName();
    }();

    auto loc = op->getLoc();
    mlir::Value newArray =
        rewriter.create<mlir::memref::GetGlobalOp>(loc, typeLocal, global);

    newArray = rewriter.create<imex::util::SignCastOp>(loc, type, newArray);

    if (type != oldType)
      newArray = rewriter.create<mlir::memref::CastOp>(loc, oldType, newArray);

    rewriter.replaceOp(op, newArray);
    return mlir::success();
  }
};

struct LowerGpuBuiltins3Pass
    : public imex::RewriteWrapperPass<LowerGpuBuiltins3Pass, void, void,
                                      ConvertArrayAllocOps> {};

class GpuLaunchSinkOpsPass
    : public mlir::PassWrapper<GpuLaunchSinkOpsPass,
                               mlir::OperationPass<void>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuLaunchSinkOpsPass)

  void runOnOperation() override {
    using namespace mlir;

    Operation *op = getOperation();
    if (op->walk([](gpu::LaunchOp launch) {
            auto isSinkingBeneficiary = [](mlir::Operation *op) -> bool {
              return isa<arith::ConstantOp, func::ConstantOp, arith::SelectOp,
                         arith::CmpIOp, arith::IndexCastOp, arith::MulIOp,
                         arith::SubIOp, arith::AddIOp, imex::util::UndefOp>(op);
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

static const constexpr llvm::StringLiteral
    kGpuModuleDeviceName("gpu_module_device");

class NameGpuModulesPass
    : public mlir::PassWrapper<NameGpuModulesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NameGpuModulesPass)

  void runOnOperation() override {
    auto mod = getOperation();
    mod->walk([&](mlir::gpu::LaunchFuncOp launch) {
      auto env = launch->getParentOfType<imex::util::EnvironmentRegionOp>();
      if (!env)
        return;

      auto gpuEnv =
          env.getEnvironment().dyn_cast<gpu_runtime::GPURegionDescAttr>();
      if (!gpuEnv)
        return;

      auto deviceName = gpuEnv.getDevice();

      auto kernel = launch.getKernel();
      auto gpuModName = kernel.getRootReference();
      auto gpuMod = mod.lookupSymbol<mlir::gpu::GPUModuleOp>(gpuModName);
      if (!gpuMod)
        return;

      auto gpuModAttr =
          gpuMod->getAttrOfType<mlir::StringAttr>(kGpuModuleDeviceName);
      if (gpuModAttr && gpuModAttr != deviceName) {
        gpuMod->emitError("Incompatible gpu module devices: ")
            << gpuModAttr.getValue() << " and " << deviceName;
        return signalPassFailure();
      }
      gpuMod->setAttr(kGpuModuleDeviceName, deviceName);
    });
  }
};

struct SinkGpuDims : public mlir::OpRewritePattern<mlir::gpu::LaunchOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::LaunchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    const mlir::Value dimArgs[] = {op.getGridSizeX(),  op.getGridSizeY(),
                                   op.getGridSizeZ(),  op.getBlockSizeX(),
                                   op.getBlockSizeY(), op.getBlockSizeZ()};
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
    rewriter.setInsertionPointToStart(&op.getBody().front());
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

struct SinkGpuDimsPass : public imex::RewriteWrapperPass<SinkGpuDimsPass, void,
                                                         void, SinkGpuDims> {};

struct GPUToLLVMPass
    : public mlir::PassWrapper<GPUToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUToLLVMPass)

  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
    mlir::LLVMTypeConverter converter(&context);
    mlir::RewritePatternSet patterns(&context);
    mlir::LLVMConversionTarget target(context);

    mlir::populateAsyncStructuralTypeConversionsAndLegality(converter, patterns,
                                                            target);
    mlir::populateGpuToLLVMConversionPatterns(
        converter, patterns, mlir::gpu::getDefaultGpuBinaryAnnotation());

    imex::populateControlFlowTypeConversionRewritesAndTarget(converter,
                                                             patterns, target);

    gpu_runtime::populateGpuToLLVMPatternsAndLegality(converter, patterns,
                                                      target);
    imex::populateUtilConversionPatterns(context, converter, patterns, target);

    auto mod = getOperation();
    if (mlir::failed(
            mlir::applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};

static llvm::Optional<mlir::spirv::Version> mapSpirvVersion(uint16_t major,
                                                            uint16_t minor) {
  if (major == 1) {
    const mlir::spirv::Version mapping[] = {
        mlir::spirv::Version::V_1_0, mlir::spirv::Version::V_1_1,
        mlir::spirv::Version::V_1_2, mlir::spirv::Version::V_1_3,
        mlir::spirv::Version::V_1_4, mlir::spirv::Version::V_1_5,
        mlir::spirv::Version::V_1_6,
    };
    if (minor < std::size(mapping))
      return mapping[minor];
  }
  return llvm::None;
}

static mlir::spirv::TargetEnvAttr deviceCapsMapper(mlir::gpu::GPUModuleOp op) {
  auto deviceAttr = op->getAttrOfType<mlir::StringAttr>(kGpuModuleDeviceName);
  if (!deviceAttr)
    return {};

  auto deviceCapsRet = getOffloadDeviceCapabilities();
  if (!deviceCapsRet)
    return nullptr;

  auto deviceCaps = *deviceCapsRet;

  auto spirvVersionRet = mapSpirvVersion(deviceCaps.spirvMajorVersion,
                                         deviceCaps.spirvMinorVersion);
  if (!spirvVersionRet)
    return nullptr;

  auto spirvVersion = *spirvVersionRet;

  // Pretend we are supporting 1.3 for non-uniform ops.
  if (spirvVersion == mlir::spirv::Version::V_1_2)
    spirvVersion = mlir::spirv::Version::V_1_3;

  auto context = op.getContext();
  namespace spirv = mlir::spirv;
  spirv::Capability fixedCaps[] = {
      // clang-format off
      spirv::Capability::Addresses,
      spirv::Capability::AtomicFloat32AddEXT,
      spirv::Capability::ExpectAssumeKHR,
      spirv::Capability::GenericPointer,
      spirv::Capability::GroupNonUniformArithmetic,
      spirv::Capability::Groups,
      spirv::Capability::Int16,
      spirv::Capability::Int64,
      spirv::Capability::Int8,
      spirv::Capability::Kernel,
      spirv::Capability::Linkage,
      spirv::Capability::Vector16,
      // clang-format on
  };
  spirv::Extension exts[] = {spirv::Extension::SPV_EXT_shader_atomic_float_add,
                             spirv::Extension::SPV_KHR_expect_assume};

  llvm::SmallVector<spirv::Capability, 0> caps(std::begin(fixedCaps),
                                               std::end(fixedCaps));

  if (deviceCaps.hasFP16) {
    caps.emplace_back(spirv::Capability::Float16);
    caps.emplace_back(spirv::Capability::Float16Buffer);
  }

  if (deviceCaps.hasFP64)
    caps.emplace_back(spirv::Capability::Float64);

  llvm::sort(caps);
  llvm::sort(exts);

  auto triple = spirv::VerCapExtAttr::get(spirvVersion, caps, exts, context);
  auto attr = spirv::TargetEnvAttr::get(
      triple, spirv::Vendor::Unknown, spirv::DeviceType::Unknown,
      spirv::TargetEnvAttr::kUnknownDeviceID,
      spirv::getDefaultResourceLimits(context));
  return attr;
}

static void commonOptPasses(mlir::OpPassManager &pm) {
  pm.addPass(imex::createCommonOptsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(imex::createCommonOptsPass());
}

static void populateLowerToGPUPipelineHigh(mlir::OpPassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<MarkGpuArraysInputs>());
  pm.addPass(std::make_unique<LowerGpuRangePass>());
  pm.addPass(std::make_unique<LowerGpuBuiltinsPass>());
  commonOptPasses(pm);
  pm.addPass(mlir::createSymbolDCEPass());
}

static void populateLowerToGPUPipelineLow(mlir::OpPassManager &pm) {
  auto &funcPM = pm.nest<mlir::func::FuncOp>();
  funcPM.addPass(std::make_unique<PrepareForGPUPass>());
  funcPM.addPass(mlir::createCanonicalizerPass());
  funcPM.addPass(std::make_unique<RemoveNestedParallelPass>());
  funcPM.addPass(gpu_runtime::createTileParallelLoopsForGPUPass());
  funcPM.addPass(gpu_runtime::createParallelLoopGPUMappingPass());
  funcPM.addPass(mlir::createParallelLoopToGpuPass());
  funcPM.addPass(mlir::createCanonicalizerPass());
  funcPM.addPass(gpu_runtime::createInsertGPUAllocsPass());
  funcPM.addPass(mlir::createCanonicalizerPass());
  funcPM.addPass(gpu_runtime::createUnstrideMemrefsPass());
  funcPM.addPass(mlir::createLowerAffinePass());

  funcPM.addPass(std::make_unique<LowerGpuBuiltins2Pass>());
  commonOptPasses(funcPM);
  funcPM.addPass(std::make_unique<KernelMemrefOpsMovementPass>());
  funcPM.addPass(gpu_runtime::createMakeBarriersUniformPass());
  funcPM.addPass(std::make_unique<SinkGpuDimsPass>());
  funcPM.addPass(std::make_unique<GpuLaunchSinkOpsPass>());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(std::make_unique<NameGpuModulesPass>());
  pm.addPass(mlir::createSymbolDCEPass());

  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<GPULowerDefaultLocalSize>());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolDCEPass());

  auto &gpuFuncPM =
      pm.nest<mlir::gpu::GPUModuleOp>().nest<mlir::gpu::GPUFuncOp>();
  gpuFuncPM.addPass(mlir::arith::createArithExpandOpsPass());
  gpuFuncPM.addPass(std::make_unique<FlattenScfPass>());
  gpuFuncPM.addPass(std::make_unique<LowerGpuBuiltins3Pass>());
  commonOptPasses(gpuFuncPM);
  gpuFuncPM.addPass(std::make_unique<AssumeGpuIdRangePass>());

  pm.addNestedPass<mlir::gpu::GPUModuleOp>(gpu_runtime::createAbiAttrsPass());
  pm.addPass(gpu_runtime::createSetSPIRVCapabilitiesPass(&deviceCapsMapper));
  pm.addPass(gpu_runtime::createGPUToSpirvPass());
  commonOptPasses(pm);

  auto &modulePM = pm.nest<mlir::spirv::ModuleOp>();
  modulePM.addPass(mlir::spirv::createLowerABIAttributesPass());
  modulePM.addPass(mlir::spirv::createUpdateVersionCapabilityExtensionPass());
  pm.addPass(gpu_runtime::createSerializeSPIRVPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      gpu_runtime::createConvertGPUDeallocsPass());
  pm.addNestedPass<mlir::func::FuncOp>(gpu_runtime::createGPUExPass());
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<RemoveGpuRegionPass>());
  commonOptPasses(pm);
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<GPUExDeallocPass>());
  pm.addPass(std::make_unique<OutlineInitPass>());
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<GenerateOutlineContextPass>());
  pm.addPass(gpu_runtime::createEnumerateEventsPass());
  pm.addPass(std::make_unique<GPUToLLVMPass>());
  commonOptPasses(pm);
}
} // namespace

void registerLowerToGPUPipeline(imex::PipelineRegistry &registry) {
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
