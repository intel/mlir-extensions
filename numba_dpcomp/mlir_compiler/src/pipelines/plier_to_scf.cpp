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
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "plier/dialect.hpp"
#include "plier/rewrites/arg_lowering.hpp"

#include "base_pipeline.hpp"
#include "plier/compiler/pipeline_registry.hpp"

namespace {
struct CondBrOpLowering : public mlir::OpRewritePattern<mlir::CondBranchOp> {
  CondBrOpLowering(mlir::MLIRContext *context) : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::CondBranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto operands = op.getOperands();
    assert(!operands.empty());
    auto cond = operands.front();
    operands = operands.drop_front();
    bool changed = false;

    auto process_operand = [&](mlir::Block &block, auto &ret) {
      for (auto arg : block.getArguments()) {
        assert(!operands.empty());
        auto val = operands.front();
        operands = operands.drop_front();
        auto src_type = val.getType();
        auto dst_type = arg.getType();
        if (src_type != dst_type) {
          ret.push_back(
              rewriter.create<plier::CastOp>(op.getLoc(), dst_type, val));
          changed = true;
        } else {
          ret.push_back(val);
        }
      }
    };

    llvm::SmallVector<mlir::Value, 4> true_vals;
    llvm::SmallVector<mlir::Value, 4> false_vals;
    auto true_dest = op.getTrueDest();
    auto false_dest = op.getFalseDest();
    process_operand(*true_dest, true_vals);
    process_operand(*false_dest, false_vals);
    if (changed) {
      rewriter.create<mlir::CondBranchOp>(op.getLoc(), cond, true_dest,
                                          true_vals, false_dest, false_vals);
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return mlir::failure();
  }
};

mlir::Block *getNextBlock(mlir::Block *block) {
  assert(nullptr != block);
  if (auto br =
          mlir::dyn_cast_or_null<mlir::BranchOp>(block->getTerminator())) {
    return br.dest();
  }
  return nullptr;
};

void erase_blocks(mlir::PatternRewriter &rewriter,
                  llvm::ArrayRef<mlir::Block *> blocks) {
  for (auto block : blocks) {
    assert(nullptr != block);
    block->dropAllDefinedValueUses();
  }
  for (auto block : blocks) {
    rewriter.eraseBlock(block);
  }
}

bool is_blocks_different(llvm::ArrayRef<mlir::Block *> blocks) {
  for (auto it : llvm::enumerate(blocks)) {
    auto block1 = it.value();
    assert(nullptr != block1);
    for (auto block2 : blocks.drop_front(it.index() + 1)) {
      assert(nullptr != block2);
      if (block1 == block2) {
        return false;
      }
    }
  }
  return true;
}

struct ScfIfRewriteOneExit : public mlir::OpRewritePattern<mlir::CondBranchOp> {
  ScfIfRewriteOneExit(mlir::MLIRContext *context) : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::CondBranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto getDest = [&](bool true_dest) {
      return true_dest ? op.getTrueDest() : op.getFalseDest();
    };
    auto getOperands = [&](bool true_dest) {
      return true_dest ? op.getTrueOperands() : op.getFalseOperands();
    };
    auto loc = op.getLoc();
    auto returnBlock = reinterpret_cast<mlir::Block *>(1); // Fake block
    for (bool reverse : {false, true}) {
      auto trueBlock = getDest(!reverse);
      auto getNextBlock = [&](mlir::Block *block) -> mlir::Block * {
        assert(nullptr != block);
        auto term = block->getTerminator();
        if (auto br = mlir::dyn_cast_or_null<mlir::BranchOp>(term)) {
          return br.dest();
        }
        if (auto ret = mlir::dyn_cast_or_null<mlir::ReturnOp>(term)) {
          return returnBlock;
        }
        return nullptr;
      };
      auto postBlock = getNextBlock(trueBlock);
      if (nullptr == postBlock) {
        continue;
      }
      auto falseBlock = getDest(reverse);
      if (falseBlock != postBlock && getNextBlock(falseBlock) != postBlock) {
        continue;
      }

      auto startBlock = op.getOperation()->getBlock();
      if (!is_blocks_different({startBlock, trueBlock, postBlock})) {
        continue;
      }
      mlir::Value cond = op.condition();
      if (reverse) {
        auto i1 = mlir::IntegerType::get(op.getContext(), 1);
        auto one = rewriter.create<mlir::ConstantOp>(
            loc, mlir::IntegerAttr::get(i1, 1));
        cond = rewriter.create<mlir::arith::XOrIOp>(loc, cond, one);
      }

      mlir::BlockAndValueMapping mapper;
      llvm::SmallVector<mlir::Value> yieldVals;
      auto copyBlock = [&](mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Block &block) {
        mapper.clear();
        for (auto &op : block.without_terminator()) {
          builder.clone(op, mapper);
        }
        auto operands = [&]() {
          auto term = block.getTerminator();
          if (postBlock == returnBlock) {
            return mlir::cast<mlir::ReturnOp>(term).operands();
          } else {
            return mlir::cast<mlir::BranchOp>(term).destOperands();
          }
        }();
        yieldVals.clear();
        yieldVals.reserve(operands.size());
        for (auto op : operands) {
          yieldVals.emplace_back(mapper.lookupOrDefault(op));
        }
        builder.create<mlir::scf::YieldOp>(loc, yieldVals);
      };

      auto trueBody = [&](mlir::OpBuilder &builder, mlir::Location loc) {
        copyBlock(builder, loc, *trueBlock);
      };

      bool hasElse = (falseBlock != postBlock);
      auto resTypes = [&]() {
        auto term = trueBlock->getTerminator();
        if (postBlock == returnBlock) {
          return mlir::cast<mlir::ReturnOp>(term).operands().getTypes();
        } else {
          return mlir::cast<mlir::BranchOp>(term).destOperands().getTypes();
        }
      }();
      mlir::scf::IfOp ifOp;
      if (hasElse) {
        auto falseBody = [&](mlir::OpBuilder &builder, mlir::Location loc) {
          copyBlock(builder, loc, *falseBlock);
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
        rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, ifOp.getResults());
      } else {
        rewriter.replaceOpWithNewOp<mlir::BranchOp>(op, postBlock,
                                                    ifOp.getResults());
      }

      if (trueBlock->getUsers().empty()) {
        erase_blocks(rewriter, trueBlock);
      }
      if (falseBlock->getUsers().empty()) {
        erase_blocks(rewriter, falseBlock);
      }
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct ScfIfRewriteTwoExits
    : public mlir::OpRewritePattern<mlir::CondBranchOp> {
  ScfIfRewriteTwoExits(mlir::MLIRContext *context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::CondBranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto thisBlock = op->getBlock();
    for (bool reverse : {false, true}) {
      auto getDest = [&](bool reverse) {
        return reverse ? op.getTrueDest() : op.getFalseDest();
      };
      auto thenBlock = getDest(!reverse);
      auto exitBlock = getDest(reverse);
      auto exitOps = (reverse ? op.getTrueOperands() : op.getFalseOperands());
      if (thenBlock == thisBlock || exitBlock == thisBlock) {
        continue;
      }
      auto thenBr =
          mlir::dyn_cast<mlir::CondBranchOp>(thenBlock->getTerminator());
      if (!thenBr) {
        continue;
      }
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

      if (exitBlock1->getNumArguments() != 0) {
        continue;
      }

      if (thenBlock->getNumArguments() != 0) {
        continue;
      }

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
        for (auto &op : thenBlock->without_terminator()) {
          builder.clone(op, mapper);
        }

        auto cond = mapper.lookupOrDefault(thenBr.condition());
        if (reverseExitCond) {
          auto one =
              builder.create<mlir::arith::ConstantIntOp>(loc, /*value*/ 1,
                                                         /*width*/ 1);
          cond = builder.create<mlir::arith::SubIOp>(loc, one, cond);
        }

        llvm::SmallVector<mlir::Value> ret;
        ret.emplace_back(cond);
        for (auto op : ops2) {
          ret.emplace_back(mapper.lookupOrDefault(op));
        }

        for (auto user : thenValsUsers) {
          ret.emplace_back(mapper.lookupOrDefault(user));
        }

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
        cond = rewriter.create<mlir::arith::SubIOp>(loc, one, cond);
      }

      auto ifRetType = rewriter.getIntegerType(1);

      llvm::SmallVector<mlir::Type> retTypes;
      retTypes.emplace_back(ifRetType);
      llvm::copy(exitOps.getTypes(), std::back_inserter(retTypes));
      for (auto user : thenValsUsers) {
        retTypes.emplace_back(user.getType());
      }

      auto ifResults = rewriter
                           .create<mlir::scf::IfOp>(loc, retTypes, cond,
                                                    trueBuilder, falseBuilder)
                           .getResults();
      cond = rewriter.create<mlir::arith::AndIOp>(loc, cond, ifResults[0]);
      ifResults = ifResults.drop_front();
      rewriter.replaceOpWithNewOp<mlir::CondBranchOp>(
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
create_while(mlir::OpBuilder &builder, mlir::Location loc,
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
    auto add_region = [&](mlir::ValueRange args) -> mlir::Block * {
      auto reg = state.addRegion();
      auto block = builder.createBlock(reg);
      for (auto arg : args) {
        block->addArgument(arg.getType());
      }
      return block;
    };

    auto beforeBlock = add_region(iterArgs);
    beforeBuilder(builder, state.location, beforeBlock->getArguments());
    auto cond =
        mlir::cast<mlir::scf::ConditionOp>(beforeBlock->getTerminator());
    state.addTypes(cond.args().getTypes());

    auto afterblock = add_region(cond.args());
    afterBuilder(builder, state.location, afterblock->getArguments());
  }
  return mlir::cast<mlir::scf::WhileOp>(builder.createOperation(state));
}

bool is_inside_block(mlir::Operation *op, mlir::Block *block) {
  assert(nullptr != op);
  assert(nullptr != block);
  do {
    if (op->getBlock() == block) {
      return true;
    }
  } while ((op = op->getParentOp()));
  return false;
}

struct ScfWhileRewrite : public mlir::OpRewritePattern<mlir::BranchOp> {
  ScfWhileRewrite(mlir::MLIRContext *context) : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::BranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto before_block = op.dest();
    auto before_term =
        mlir::dyn_cast<mlir::CondBranchOp>(before_block->getTerminator());
    if (!before_term) {
      return mlir::failure();
    }
    auto start_block = op.getOperation()->getBlock();
    auto after_block = before_term.trueDest();
    auto post_block = before_term.falseDest();
    if (getNextBlock(after_block) != before_block ||
        !is_blocks_different(
            {start_block, before_block, after_block, post_block})) {
      return mlir::failure();
    }

    auto check_outside_vals = [&](mlir::Operation *op) -> mlir::WalkResult {
      for (auto user : op->getUsers()) {
        if (!is_inside_block(user, before_block) &&
            !is_inside_block(user, after_block)) {
          return mlir::WalkResult::interrupt();
        }
      }
      return mlir::WalkResult::advance();
    };

    if (after_block->walk(check_outside_vals).wasInterrupted()) {
      return mlir::failure();
    }

    mlir::BlockAndValueMapping mapper;
    llvm::SmallVector<mlir::Value> yield_vars;
    auto before_block_args = before_block->getArguments();
    llvm::SmallVector<mlir::Value> orig_vars(before_block_args.begin(),
                                             before_block_args.end());

    auto before_body = [&](mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::ValueRange iterargs) {
      mapper.map(before_block_args, iterargs);
      yield_vars.resize(before_block_args.size());
      for (auto &op : before_block->without_terminator()) {
        auto new_op = builder.clone(op, mapper);
        for (auto user : op.getUsers()) {
          if (!is_inside_block(user, before_block)) {
            for (auto it : llvm::zip(op.getResults(), new_op->getResults())) {
              orig_vars.emplace_back(std::get<0>(it));
              yield_vars.emplace_back(std::get<1>(it));
            }
            break;
          }
        }
      }

      llvm::transform(
          before_block->getArguments(), yield_vars.begin(),
          [&](mlir::Value val) { return mapper.lookupOrDefault(val); });

      auto term = mlir::cast<mlir::CondBranchOp>(before_block->getTerminator());
      for (auto arg : term.falseDestOperands()) {
        orig_vars.emplace_back(arg);
        yield_vars.emplace_back(mapper.lookupOrDefault(arg));
      }
      auto cond = mapper.lookupOrDefault(term.condition());
      builder.create<mlir::scf::ConditionOp>(loc, cond, yield_vars);
    };
    auto after_body = [&](mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::ValueRange iterargs) {
      mapper.clear();
      assert(orig_vars.size() == iterargs.size());
      mapper.map(orig_vars, iterargs);
      for (auto &op : after_block->without_terminator()) {
        builder.clone(op, mapper);
      }
      yield_vars.clear();
      auto term = mlir::cast<mlir::BranchOp>(after_block->getTerminator());
      for (auto arg : term.getOperands()) {
        yield_vars.emplace_back(mapper.lookupOrDefault(arg));
      }
      builder.create<mlir::scf::YieldOp>(loc, yield_vars);
    };

    auto while_op = create_while(rewriter, op.getLoc(), op.getOperands(),
                                 before_body, after_body);

    assert(orig_vars.size() == while_op.getNumResults());
    for (auto arg : llvm::zip(orig_vars, while_op.getResults())) {
      std::get<0>(arg).replaceAllUsesWith(std::get<1>(arg));
    }

    rewriter.create<mlir::BranchOp>(op.getLoc(), post_block,
                                    before_term.falseDestOperands());
    rewriter.eraseOp(op);
    erase_blocks(rewriter, {before_block, after_block});

    return mlir::success();
  }
};

struct BreakRewrite : public mlir::OpRewritePattern<mlir::CondBranchOp> {
  BreakRewrite(mlir::MLIRContext *context) : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::CondBranchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto bodyBlock = op->getBlock();
    auto exitBlock = op.getTrueDest();
    auto conditionBlock = op.getFalseDest();
    auto conditionBr =
        mlir::dyn_cast<mlir::CondBranchOp>(conditionBlock->getTerminator());
    if (!conditionBr) {
      return mlir::failure();
    }

    if (conditionBr.getTrueDest() != bodyBlock ||
        conditionBr.getFalseDest() != exitBlock) {
      return mlir::failure();
    }

    auto loc = rewriter.getUnknownLoc();

    auto type = rewriter.getIntegerType(1);
    auto condVal = rewriter.getIntegerAttr(type, 1);

    conditionBlock->addArgument(op.getCondition().getType());
    for (auto user : llvm::make_early_inc_range(conditionBlock->getUsers())) {
      if (user != op) {
        rewriter.setInsertionPoint(user);
        auto condConst = rewriter.create<mlir::ConstantOp>(loc, condVal);
        if (auto br = mlir::dyn_cast<mlir::BranchOp>(user)) {
          llvm::SmallVector<mlir::Value> params(br.destOperands());
          params.emplace_back(condConst);
          rewriter.create<mlir::BranchOp>(br.getLoc(), conditionBlock, params);
          rewriter.eraseOp(br);
        } else if (auto condBr = mlir::dyn_cast<mlir::CondBranchOp>(user)) {
          llvm_unreachable("not implemented");
        } else {
          llvm_unreachable("Unknown terminator type");
        }
      }
    }

    rewriter.setInsertionPoint(op);
    llvm::SmallVector<mlir::Value> params(op.getFalseOperands());
    auto one = rewriter.create<mlir::ConstantOp>(loc, condVal);
    auto invertedCond =
        rewriter.create<mlir::arith::SubIOp>(loc, one, op.condition());
    params.push_back(invertedCond);
    rewriter.create<mlir::BranchOp>(op.getLoc(), conditionBlock, params);
    rewriter.eraseOp(op);

    rewriter.setInsertionPoint(conditionBr);
    auto oldCond = conditionBr.getCondition();
    mlir::Value newCond = conditionBlock->getArguments().back();
    one = rewriter.create<mlir::ConstantOp>(loc, condVal);
    newCond = rewriter.create<mlir::arith::AndIOp>(loc, newCond, oldCond);
    rewriter.create<mlir::CondBranchOp>(
        conditionBr.getLoc(), newCond, conditionBr.getTrueDest(),
        conditionBr.getTrueOperands(), conditionBr.getFalseDest(),
        conditionBr.getFalseOperands());
    rewriter.eraseOp(conditionBr);
    return mlir::success();
  }
};

struct PlierToScfPass
    : public mlir::PassWrapper<PlierToScfPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    // registry.insert<plier::PlierDialect>();
    // registry.insert<mlir::StandardOpsDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override;
};

void PlierToScfPass::runOnOperation() {
  auto context = &getContext();

  mlir::OwningRewritePatternList patterns(context);

  patterns.insert<
      // clang-format off
      plier::ArgOpLowering,
      CondBrOpLowering,
      BreakRewrite,
      ScfIfRewriteOneExit,
      ScfIfRewriteTwoExits,
      ScfWhileRewrite
      // clang-format on
      >(context);

  for (auto *op : context->getRegisteredOperations())
    op->getCanonicalizationPatterns(patterns, context);

  (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

void populate_plier_to_scf_pipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<PlierToScfPass>());
  pm.addPass(mlir::createCanonicalizerPass());
}
} // namespace

void registerPlierToScfPipeline(plier::PipelineRegistry &registry) {
  registry.register_pipeline([](auto sink) {
    auto stage = getHighLoweringStage();
    sink(plierToScfPipelineName(), {stage.begin}, {stage.end}, {},
         &populate_plier_to_scf_pipeline);
  });
}

llvm::StringRef plierToScfPipelineName() { return "plier_to_scf"; }
