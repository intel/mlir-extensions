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
#include <mlir/Conversion/SCFToGPU/SCFToGPUPass.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/Transforms/Passes.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/Passes.h>
#include <mlir/Dialect/GPU/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include "base_pipeline.hpp"
#include "loop_utils.hpp"
#include "pipelines/lower_to_llvm.hpp"
#include "pipelines/plier_to_linalg.hpp"
#include "pipelines/plier_to_std.hpp"
#include "pipelines/pre_low_simplifications.hpp"
#include "py_linalg_resolver.hpp"

#include "mlir-extensions/Conversion/gpu_runtime_to_llvm.hpp"
#include "mlir-extensions/Conversion/gpu_to_gpu_runtime.hpp"
#include "mlir-extensions/Dialect/gpu_runtime/IR/gpu_runtime_ops.hpp"
#include "mlir-extensions/Dialect/plier_util/dialect.hpp"
#include "mlir-extensions/Transforms/call_lowering.hpp"
#include "mlir-extensions/Transforms/cast_utils.hpp"
#include "mlir-extensions/Transforms/common_opts.hpp"
#include "mlir-extensions/Transforms/pipeline_utils.hpp"
#include "mlir-extensions/Transforms/rewrite_wrapper.hpp"
#include "mlir-extensions/Transforms/type_conversion.hpp"
#include "mlir-extensions/compiler/pipeline_registry.hpp"

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
                               mlir::OperationPass<mlir::func::FuncOp>> {
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

struct KernelMemrefOpsMovementPass
    : public mlir::PassWrapper<KernelMemrefOpsMovementPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
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

struct AssumeGpuIdRangePass
    : public mlir::PassWrapper<AssumeGpuIdRangePass,
                               mlir::OperationPass<void>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithmeticDialect>();
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
          globalSize[i] = plier::indexCast(builder, loc, operands[i]);

        auto res = builder
                       .create<gpu_runtime::GPUSuggestBlockSizeOp>(
                           loc, /*stream*/ llvm::None, kernel, globalSize)
                       .getResults();

        for (auto i : llvm::seq(0u, count)) {
          auto castedRes = plier::indexCast(builder, loc, res[i],
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

void rerun_std_pipeline(mlir::Operation *op) {
  assert(nullptr != op);
  auto marker =
      mlir::StringAttr::get(op->getContext(), plierToStdPipelineName());
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  assert(nullptr != mod);
  plier::addPipelineJumpMarker(mod, marker);
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
                               mlir::OperationPass<mlir::func::FuncOp>> {
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
    auto func = op->getParentOfType<mlir::func::FuncOp>();
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
          return cast.getInputs()[0];

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
  rewriter.replaceOpWithNewOp<plier::UndefOp>(srcOp, retType);
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

struct LowerGpuBuiltins2Pass
    : public plier::RewriteWrapperPass<LowerGpuBuiltins2Pass, void, void,
                                       ConvertBarrierOps> {};

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
    auto storageClass = gpu_runtime::StorageClassAttr::get(
        getContext(), gpu_runtime::StorageClass::local);
    auto typeLocal = mlir::MemRefType::get(shape, type.getElementType(),
                                           nullptr, storageClass);

    auto global = [&]() -> mlir::StringRef {
      auto *block = &mod.body().front();
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
      return global.sym_name();
    }();

    auto loc = op->getLoc();
    mlir::Value newArray =
        rewriter.create<mlir::memref::GetGlobalOp>(loc, typeLocal, global);

    newArray = rewriter.create<plier::SignCastOp>(loc, type, newArray);

    if (type != oldType)
      newArray = rewriter.create<mlir::memref::CastOp>(loc, oldType, newArray);

    rewriter.replaceOp(op, newArray);
    return mlir::success();
  }
};

struct LowerGpuBuiltins3Pass
    : public plier::RewriteWrapperPass<LowerGpuBuiltins3Pass, void, void,
                                       ConvertArrayAllocOps> {};

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
                         arith::SubIOp, arith::AddIOp, plier::UndefOp>(op);
            };

            // Pull in instructions that can be sunk
            if (failed(
                    sinkOperationsIntoLaunchOp(launch, isSinkingBeneficiary)))
              return WalkResult::interrupt();

            return WalkResult::advance();
          })
            .wasInterrupted())
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
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<MarkGpuArraysInputs>());
  pm.addPass(std::make_unique<ConvertGpuArrays>());
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
  funcPM.addPass(gpu_runtime::createParallelLoopGPUMappingPass());
  funcPM.addPass(mlir::createParallelLoopToGpuPass());
  funcPM.addPass(std::make_unique<RemoveKernelMarkerPass>());
  funcPM.addPass(mlir::createCanonicalizerPass());
  funcPM.addPass(
      gpu_runtime::createInsertGPUAllocsPass(/*useGpuDealloc*/ false));
  funcPM.addPass(mlir::createCanonicalizerPass());
  funcPM.addPass(gpu_runtime::createUnstrideMemrefsPass());
  funcPM.addPass(mlir::createLowerAffinePass());

  commonOptPasses(funcPM);
  funcPM.addPass(std::make_unique<KernelMemrefOpsMovementPass>());
  funcPM.addPass(std::make_unique<LowerGpuBuiltins2Pass>());
  funcPM.addPass(std::make_unique<SinkGpuDimsPass>());
  funcPM.addPass(std::make_unique<GpuLaunchSinkOpsPass>());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(mlir::createSymbolDCEPass());

  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<GPULowerDefaultLocalSize>());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolDCEPass());

  auto &gpuFuncPM =
      pm.nest<mlir::gpu::GPUModuleOp>().nest<mlir::gpu::GPUFuncOp>();
  gpuFuncPM.addPass(mlir::arith::createArithmeticExpandOpsPass());
  gpuFuncPM.addPass(std::make_unique<FlattenScfPass>());
  gpuFuncPM.addPass(std::make_unique<LowerGpuBuiltins3Pass>());
  commonOptPasses(gpuFuncPM);
  gpuFuncPM.addPass(std::make_unique<AssumeGpuIdRangePass>());

  pm.addNestedPass<mlir::gpu::GPUModuleOp>(gpu_runtime::createAbiAttrsPass());
  pm.addPass(gpu_runtime::createSetSPIRVCapabilitiesPass());
  pm.addPass(gpu_runtime::createGPUToSpirvPass());
  commonOptPasses(pm);

  auto &modulePM = pm.nest<mlir::spirv::ModuleOp>();
  modulePM.addPass(mlir::spirv::createLowerABIAttributesPass());
  modulePM.addPass(mlir::spirv::createUpdateVersionCapabilityExtensionPass());
  pm.addPass(gpu_runtime::createSerializeSPIRVPass());
  pm.addNestedPass<mlir::func::FuncOp>(gpu_runtime::createGPUExPass());
  commonOptPasses(pm);
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<GPUExDeallocPass>());
  pm.addPass(std::make_unique<OutlineInitPass>());
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<GenerateOutlineContextPass>());
  pm.addPass(gpu_runtime::createEnumerateEventsPass());
  pm.addPass(gpu_runtime::createGPUToLLVMPass());
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
