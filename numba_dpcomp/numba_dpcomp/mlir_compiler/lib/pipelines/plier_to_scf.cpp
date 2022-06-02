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

#include "pipelines/plier_to_scf.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "mlir-extensions/Dialect/plier/dialect.hpp"
#include "mlir-extensions/Dialect/plier_util/dialect.hpp"
#include "mlir-extensions/Transforms/arg_lowering.hpp"
#include "mlir-extensions/Transforms/common_opts.hpp"

#include "base_pipeline.hpp"
#include "mlir-extensions/compiler/pipeline_registry.hpp"

namespace {
static mlir::Block *getNextBlock(mlir::Block *block) {
  assert(nullptr != block);
  if (auto br =
          mlir::dyn_cast_or_null<mlir::cf::BranchOp>(block->getTerminator())) {
    return br.getDest();
  }
  return nullptr;
};

static void eraseBlocks(mlir::PatternRewriter &rewriter,
                        llvm::ArrayRef<mlir::Block *> blocks) {
  for (auto block : blocks) {
    assert(nullptr != block);
    block->dropAllDefinedValueUses();
  }
  for (auto block : blocks) {
    rewriter.eraseBlock(block);
  }
}

static bool isBlocksDifferent(llvm::ArrayRef<mlir::Block *> blocks) {
  for (auto it : llvm::enumerate(blocks)) {
    auto block1 = it.value();
    assert(nullptr != block1);
    for (auto block2 : blocks.drop_front(it.index() + 1)) {
      assert(nullptr != block2);
      if (block1 == block2)
        return false;
    }
  }
  return true;
}

struct ScfIfRewriteOneExit
    : public mlir::OpRewritePattern<mlir::cf::CondBranchOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.getTrueDest() || !op.getFalseDest())
      return mlir::failure();

    auto getDest = [&](bool trueDest) {
      return trueDest ? op.getTrueDest() : op.getFalseDest();
    };
    auto getOperands = [&](bool trueDest) {
      return trueDest ? op.getTrueOperands() : op.getFalseOperands();
    };
    auto loc = op.getLoc();
    auto returnBlock = reinterpret_cast<mlir::Block *>(1); // Fake block
    for (bool reverse : {false, true}) {
      auto trueBlock = getDest(!reverse);
      auto getNextBlock = [&](mlir::Block *block) -> mlir::Block * {
        assert(nullptr != block);
        auto term = block->getTerminator();
        if (auto br = mlir::dyn_cast_or_null<mlir::cf::BranchOp>(term))
          return br.getDest();

        if (auto ret = mlir::dyn_cast_or_null<mlir::func::ReturnOp>(term))
          return returnBlock;

        return nullptr;
      };
      auto postBlock = getNextBlock(trueBlock);
      if (nullptr == postBlock)
        continue;

      auto falseBlock = getDest(reverse);
      if (falseBlock != postBlock && getNextBlock(falseBlock) != postBlock)
        continue;

      auto startBlock = op.getOperation()->getBlock();
      if (!isBlocksDifferent({startBlock, trueBlock, postBlock}))
        continue;

      mlir::Value cond = op.getCondition();
      if (reverse) {
        auto i1 = mlir::IntegerType::get(op.getContext(), 1);
        auto one = rewriter.create<mlir::arith::ConstantOp>(
            loc, mlir::IntegerAttr::get(i1, 1));
        cond = rewriter.create<mlir::arith::XOrIOp>(loc, cond, one);
      }

      mlir::BlockAndValueMapping mapper;
      llvm::SmallVector<mlir::Value> yieldVals;
      auto copyBlock = [&](mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Block &block, mlir::ValueRange args) {
        assert(args.size() == block.getNumArguments());
        mapper.clear();
        mapper.map(block.getArguments(), args);
        for (auto &op : block.without_terminator())
          builder.clone(op, mapper);

        auto operands = [&]() {
          auto term = block.getTerminator();
          if (postBlock == returnBlock) {
            return mlir::cast<mlir::func::ReturnOp>(term).operands();
          } else {
            return mlir::cast<mlir::cf::BranchOp>(term).getDestOperands();
          }
        }();
        yieldVals.clear();
        yieldVals.reserve(operands.size());
        for (auto op : operands)
          yieldVals.emplace_back(mapper.lookupOrDefault(op));

        builder.create<mlir::scf::YieldOp>(loc, yieldVals);
      };

      auto trueBody = [&](mlir::OpBuilder &builder, mlir::Location loc) {
        copyBlock(builder, loc, *trueBlock, getOperands(!reverse));
      };

      bool hasElse = (falseBlock != postBlock);
      auto resTypes = [&]() {
        auto term = trueBlock->getTerminator();
        if (postBlock == returnBlock) {
          return mlir::cast<mlir::func::ReturnOp>(term).operands().getTypes();
        } else {
          return mlir::cast<mlir::cf::BranchOp>(term)
              .getDestOperands()
              .getTypes();
        }
      }();
      mlir::scf::IfOp ifOp;
      if (hasElse) {
        auto falseBody = [&](mlir::OpBuilder &builder, mlir::Location loc) {
          copyBlock(builder, loc, *falseBlock, getOperands(reverse));
        };
        ifOp = rewriter.create<mlir::scf::IfOp>(loc, resTypes, cond, trueBody,
                                                falseBody);
      } else {
        if (resTypes.empty()) {
          ifOp =
              rewriter.create<mlir::scf::IfOp>(loc, resTypes, cond, trueBody);
        } else {
          auto falseBody = [&](mlir::OpBuilder &builder, mlir::Location loc) {
            auto res = getOperands(reverse);
            yieldVals.clear();
            yieldVals.reserve(res.size());
            for (auto op : res) {
              yieldVals.emplace_back(mapper.lookupOrDefault(op));
            }
            builder.create<mlir::scf::YieldOp>(loc, yieldVals);
          };
          ifOp = rewriter.create<mlir::scf::IfOp>(loc, resTypes, cond, trueBody,
                                                  falseBody);
        }
      }

      if (postBlock == returnBlock) {
        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op,
                                                          ifOp.getResults());
      } else {
        rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, postBlock,
                                                        ifOp.getResults());
      }

      if (trueBlock->use_empty())
        eraseBlocks(rewriter, trueBlock);

      if (falseBlock->use_empty())
        eraseBlocks(rewriter, falseBlock);

      return mlir::success();
    }
    return mlir::failure();
  }
};

struct ScfIfRewriteTwoExits
    : public mlir::OpRewritePattern<mlir::cf::CondBranchOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    assert(op.getTrueDest());
    assert(op.getFalseDest());

    auto thisBlock = op->getBlock();
    for (bool reverse : {false, true}) {
      auto getDest = [&](bool reverse) {
        return reverse ? op.getTrueDest() : op.getFalseDest();
      };
      auto thenBlock = getDest(!reverse);
      auto exitBlock = getDest(reverse);
      auto exitOps = (reverse ? op.getTrueOperands() : op.getFalseOperands());
      if (thenBlock == thisBlock || exitBlock == thisBlock)
        continue;

      auto thenBr =
          mlir::dyn_cast<mlir::cf::CondBranchOp>(thenBlock->getTerminator());
      if (!thenBr)
        continue;

      auto exitBlock1 = thenBr.getTrueDest();
      auto exitBlock2 = thenBr.getFalseDest();
      auto ops1 = thenBr.getTrueOperands();
      auto ops2 = thenBr.getFalseOperands();
      bool reverseExitCond = false;
      if (exitBlock2 == exitBlock) {
        // nothing
      } else if (exitBlock1 == exitBlock) {
        std::swap(exitBlock1, exitBlock2);
        std::swap(ops1, ops2);
        reverseExitCond = true;
      } else {
        continue;
      }

      if (exitBlock1->getNumArguments() != 0)
        continue;

      if (thenBlock->getNumArguments() != 0)
        continue;

      llvm::SmallVector<mlir::Value> thenValsUsers;
      for (auto &op : thenBlock->without_terminator()) {
        for (auto res : op.getResults()) {
          if (res.isUsedOutsideOfBlock(thenBlock)) {
            thenValsUsers.emplace_back(res);
          }
        }
      }

      auto trueBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc) {
        mlir::BlockAndValueMapping mapper;
        for (auto &op : thenBlock->without_terminator())
          builder.clone(op, mapper);

        auto cond = mapper.lookupOrDefault(thenBr.getCondition());
        if (reverseExitCond) {
          auto one =
              builder.create<mlir::arith::ConstantIntOp>(loc, /*value*/ 1,
                                                         /*width*/ 1);
          cond = builder.create<mlir::arith::XOrIOp>(loc, one, cond);
        }

        llvm::SmallVector<mlir::Value> ret;
        ret.emplace_back(cond);
        for (auto op : ops2)
          ret.emplace_back(mapper.lookupOrDefault(op));

        for (auto user : thenValsUsers)
          ret.emplace_back(mapper.lookupOrDefault(user));

        builder.create<mlir::scf::YieldOp>(loc, ret);
      };

      auto falseBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc) {
        mlir::Value cond = rewriter.create<mlir::arith::ConstantIntOp>(
            loc, /*value*/ 0, /*width*/ 1);
        llvm::SmallVector<mlir::Value> ret;
        ret.emplace_back(cond);
        llvm::copy(exitOps, std::back_inserter(ret));
        for (auto user : thenValsUsers) {
          auto val = builder.create<plier::UndefOp>(loc, user.getType());
          ret.emplace_back(val);
        }
        builder.create<mlir::scf::YieldOp>(loc, ret);
      };

      auto cond = op.getCondition();
      auto loc = op->getLoc();
      if (reverse) {
        auto one = rewriter.create<mlir::arith::ConstantIntOp>(loc, /*value*/ 1,
                                                               /*width*/ 1);
        cond = rewriter.create<mlir::arith::XOrIOp>(loc, one, cond);
      }

      auto ifRetType = rewriter.getIntegerType(1);

      llvm::SmallVector<mlir::Type> retTypes;
      retTypes.emplace_back(ifRetType);
      llvm::copy(exitOps.getTypes(), std::back_inserter(retTypes));
      for (auto user : thenValsUsers)
        retTypes.emplace_back(user.getType());

      auto ifResults = rewriter
                           .create<mlir::scf::IfOp>(loc, retTypes, cond,
                                                    trueBuilder, falseBuilder)
                           .getResults();
      cond = rewriter.create<mlir::arith::AndIOp>(loc, cond, ifResults[0]);
      ifResults = ifResults.drop_front();
      rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
          op, cond, exitBlock1, ops1, exitBlock2,
          ifResults.take_front(exitOps.size()));
      for (auto it : llvm::zip(thenValsUsers,
                               ifResults.take_back(thenValsUsers.size()))) {
        auto oldUser = std::get<0>(it);
        auto newUser = std::get<1>(it);
        oldUser.replaceAllUsesWith(newUser);
      }
      return mlir::success();
    }
    return mlir::failure();
  }
};

mlir::scf::WhileOp
createWhile(mlir::OpBuilder &builder, mlir::Location loc,
            mlir::ValueRange iterArgs,
            llvm::function_ref<void(mlir::OpBuilder &, mlir::Location,
                                    mlir::ValueRange)>
                beforeBuilder,
            llvm::function_ref<void(mlir::OpBuilder &, mlir::Location,
                                    mlir::ValueRange)>
                afterBuilder) {
  mlir::OperationState state(loc, mlir::scf::WhileOp::getOperationName());
  state.addOperands(iterArgs);

  {
    mlir::OpBuilder::InsertionGuard g(builder);
    auto addRegion = [&](mlir::ValueRange args) -> mlir::Block * {
      auto reg = state.addRegion();
      auto block = builder.createBlock(reg);
      auto loc = builder.getUnknownLoc();
      for (auto arg : args)
        block->addArgument(arg.getType(), loc);

      return block;
    };

    auto beforeBlock = addRegion(iterArgs);
    beforeBuilder(builder, state.location, beforeBlock->getArguments());
    auto cond =
        mlir::cast<mlir::scf::ConditionOp>(beforeBlock->getTerminator());
    state.addTypes(cond.getArgs().getTypes());

    auto afterblock = addRegion(cond.getArgs());
    afterBuilder(builder, state.location, afterblock->getArguments());
  }
  return mlir::cast<mlir::scf::WhileOp>(builder.create(state));
}

bool isInsideBlock(mlir::Operation *op, mlir::Block *block) {
  assert(nullptr != op);
  assert(nullptr != block);
  do {
    if (op->getBlock() == block)
      return true;
  } while ((op = op->getParentOp()));
  return false;
}

struct ScfWhileRewrite : public mlir::OpRewritePattern<mlir::cf::BranchOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::BranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto beforeBlock = op.getDest();
    auto beforeTerm =
        mlir::dyn_cast<mlir::cf::CondBranchOp>(beforeBlock->getTerminator());
    if (!beforeTerm)
      return mlir::failure();

    auto startBlock = op.getOperation()->getBlock();
    auto afterBlock = beforeTerm.getTrueDest();
    auto postBlock = beforeTerm.getFalseDest();
    if (getNextBlock(afterBlock) != beforeBlock ||
        !isBlocksDifferent({startBlock, beforeBlock, afterBlock, postBlock}))
      return mlir::failure();

    auto checkOutsideVals = [&](mlir::Operation *op) -> mlir::WalkResult {
      for (auto user : op->getUsers())
        if (!isInsideBlock(user, beforeBlock) &&
            !isInsideBlock(user, afterBlock))
          return mlir::WalkResult::interrupt();

      return mlir::WalkResult::advance();
    };

    if (afterBlock->walk(checkOutsideVals).wasInterrupted())
      return mlir::failure();

    mlir::BlockAndValueMapping mapper;
    llvm::SmallVector<mlir::Value> yieldVars;
    auto beforeBlockArgs = beforeBlock->getArguments();
    llvm::SmallVector<mlir::Value> origVars(beforeBlockArgs.begin(),
                                            beforeBlockArgs.end());

    auto beforeBody = [&](mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::ValueRange iterargs) {
      mapper.map(beforeBlockArgs, iterargs);
      yieldVars.resize(beforeBlockArgs.size());
      for (auto &op : beforeBlock->without_terminator()) {
        auto newOp = builder.clone(op, mapper);
        for (auto user : op.getUsers()) {
          if (!isInsideBlock(user, beforeBlock)) {
            for (auto it : llvm::zip(op.getResults(), newOp->getResults())) {
              origVars.emplace_back(std::get<0>(it));
              yieldVars.emplace_back(std::get<1>(it));
            }
            break;
          }
        }
      }

      llvm::transform(beforeBlockArgs, yieldVars.begin(), [&](mlir::Value val) {
        return mapper.lookupOrDefault(val);
      });

      auto falseArgs = beforeTerm.getFalseDestOperands();
      for (auto arg : falseArgs) {
        origVars.emplace_back(arg);
        yieldVars.emplace_back(mapper.lookupOrDefault(arg));
      }

      auto cond = mapper.lookupOrDefault(beforeTerm.getCondition());
      builder.create<mlir::scf::ConditionOp>(loc, cond, yieldVars);
    };
    auto afterBody = [&](mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ValueRange iterargs) {
      mapper.clear();
      assert(origVars.size() == iterargs.size());
      mapper.map(origVars, iterargs);
      for (auto &op : afterBlock->without_terminator())
        builder.clone(op, mapper);

      yieldVars.clear();
      auto term = mlir::cast<mlir::cf::BranchOp>(afterBlock->getTerminator());
      for (auto arg : term.getOperands())
        yieldVars.emplace_back(mapper.lookupOrDefault(arg));

      builder.create<mlir::scf::YieldOp>(loc, yieldVars);
    };

    auto whileOp = createWhile(rewriter, op.getLoc(), op.getOperands(),
                               beforeBody, afterBody);

    assert(origVars.size() == whileOp.getNumResults());
    for (auto arg : llvm::zip(origVars, whileOp.getResults())) {
      auto origVal = std::get<0>(arg);
      for (auto &use : llvm::make_early_inc_range(origVal.getUses())) {
        auto *owner = use.getOwner();
        auto *block = owner->getBlock();
        if (block != &whileOp.getBefore().front() &&
            block != &whileOp.getAfter().front()) {
          auto newVal = std::get<1>(arg);
          rewriter.updateRootInPlace(owner, [&]() { use.set(newVal); });
        }
      }
    }

    auto results =
        whileOp.getResults().take_back(beforeTerm.getNumFalseOperands());
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, postBlock, results);
    return mlir::success();
  }
};

struct BreakRewrite : public mlir::OpRewritePattern<mlir::cf::CondBranchOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto bodyBlock = op->getBlock();
    auto exitBlock = op.getTrueDest();
    auto conditionBlock = op.getFalseDest();
    assert(exitBlock);
    assert(conditionBlock);

    auto conditionBr =
        mlir::dyn_cast<mlir::cf::CondBranchOp>(conditionBlock->getTerminator());
    if (!conditionBr)
      return mlir::failure();

    if (conditionBr.getTrueDest() != bodyBlock ||
        conditionBr.getFalseDest() != exitBlock)
      return mlir::failure();

    auto loc = rewriter.getUnknownLoc();

    auto type = rewriter.getIntegerType(1);
    auto condVal = rewriter.getIntegerAttr(type, 1);

    conditionBlock->addArgument(op.getCondition().getType(),
                                rewriter.getUnknownLoc());
    mlir::OpBuilder::InsertionGuard g(rewriter);
    for (auto user : llvm::make_early_inc_range(conditionBlock->getUsers())) {
      if (user != op) {
        rewriter.setInsertionPoint(user);
        auto condConst = rewriter.create<mlir::arith::ConstantOp>(loc, condVal);
        if (auto br = mlir::dyn_cast<mlir::cf::BranchOp>(user)) {
          llvm::SmallVector<mlir::Value> params(br.getDestOperands());
          params.emplace_back(condConst);
          rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(br, conditionBlock,
                                                          params);
        } else if (auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(user)) {
          llvm_unreachable("not implemented");
        } else {
          llvm_unreachable("Unknown terminator type");
        }
      }
    }

    rewriter.setInsertionPoint(op);
    llvm::SmallVector<mlir::Value> params(op.getFalseOperands());
    auto one = rewriter.create<mlir::arith::ConstantOp>(loc, condVal);
    auto invertedCond =
        rewriter.create<mlir::arith::XOrIOp>(loc, one, op.getCondition());
    params.push_back(invertedCond);
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, conditionBlock, params);

    rewriter.setInsertionPoint(conditionBr);
    auto oldCond = conditionBr.getCondition();
    mlir::Value newCond = conditionBlock->getArguments().back();
    one = rewriter.create<mlir::arith::ConstantOp>(loc, condVal);
    newCond = rewriter.create<mlir::arith::AndIOp>(loc, newCond, oldCond);
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        conditionBr, newCond, conditionBr.getTrueDest(),
        conditionBr.getTrueOperands(), conditionBr.getFalseDest(),
        conditionBr.getFalseOperands());
    return mlir::success();
  }
};

struct CondBranchSameTargetRewrite
    : public mlir::OpRewritePattern<mlir::cf::CondBranchOp> {
  // Set higher benefit than if rewrites
  CondBranchSameTargetRewrite(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::cf::CondBranchOp>(context,
                                                       /*benefit*/ 10) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto trueDest = op.getTrueDest();
    assert(trueDest);
    auto falseDest = op.getFalseDest();
    assert(falseDest);
    if (trueDest != falseDest)
      return mlir::failure();

    assert(op.getTrueOperands().size() == op.getFalseOperands().size());

    auto loc = op.getLoc();
    auto condition = op.getCondition();
    auto count = static_cast<unsigned>(op.getTrueOperands().size());
    llvm::SmallVector<mlir::Value> newOperands(count);
    for (auto i : llvm::seq(0u, count)) {
      auto trueArg = op.getTrueOperand(i);
      assert(trueArg);
      auto falseArg = op.getFalseOperand(i);
      assert(falseArg);
      if (trueArg == falseArg) {
        newOperands[i] = trueArg;
      } else {
        newOperands[i] = rewriter.create<mlir::arith::SelectOp>(
            loc, condition, trueArg, falseArg);
      }
    }

    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, trueDest, newOperands);
    return mlir::success();
  }
};

struct PlierToScfPass
    : public mlir::PassWrapper<PlierToScfPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<plier::PlierDialect>();
    registry.insert<plier::PlierUtilDialect>();
  }

  void runOnOperation() override;
};

void PlierToScfPass::runOnOperation() {
  auto context = &getContext();

  mlir::RewritePatternSet patterns(context);

  patterns.insert<
      // clang-format off
      plier::ArgOpLowering,
      BreakRewrite,
      ScfIfRewriteOneExit,
      ScfIfRewriteTwoExits,
      ScfWhileRewrite,
      CondBranchSameTargetRewrite
      // clang-format on
      >(context);

  plier::populateCanonicalizationPatterns(*context, patterns);

  auto op = getOperation();
  (void)mlir::applyPatternsAndFoldGreedily(op, std::move(patterns));

  op.walk([&](mlir::Operation *o) -> mlir::WalkResult {
    if (mlir::isa<mlir::cf::BranchOp, mlir::cf::CondBranchOp>(o)) {
      o->emitError("Unable to convert CFG to SCF");
      signalPassFailure();
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
}

static void populatePlierToScfPipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<PlierToScfPass>());
  pm.addPass(mlir::createCanonicalizerPass());
}
} // namespace

void registerPlierToScfPipeline(plier::PipelineRegistry &registry) {
  registry.registerPipeline([](auto sink) {
    auto stage = getHighLoweringStage();
    sink(plierToScfPipelineName(), {stage.begin}, {stage.end}, {},
         &populatePlierToScfPipeline);
  });
}

llvm::StringRef plierToScfPipelineName() { return "plier_to_scf"; }
