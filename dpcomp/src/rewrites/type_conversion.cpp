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

#include "plier/rewrites/type_conversion.hpp"

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SCF/Transforms.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/StandardOps/Transforms/FuncConversions.h>
#include <mlir/Transforms/DialectConversion.h>

#include "plier/dialect.hpp"

namespace {
mlir::LogicalResult
setBlockSig(mlir::Block &block, mlir::OpBuilder &builder,
            const mlir::TypeConverter::SignatureConversion &conversion) {
  llvm::SmallVector<unsigned> argsToErase;
  for (auto it : llvm::enumerate(block.getArguments())) {
    auto arg = it.value();
    auto i = static_cast<unsigned>(it.index());
    auto input = conversion.getInputMapping(i);
    bool hasInput = static_cast<bool>(input);
    auto getNewType = [&]() {
      assert(hasInput);
      return conversion.getConvertedTypes()[input->inputNo];
    };
    if (!hasInput || arg.getType() != getNewType()) {
      auto type = (hasInput ? getNewType() : arg.getType());
      if (!arg.getUses().empty()) {
        builder.setInsertionPointToStart(&block);
        auto loc = builder.getUnknownLoc();
        if (hasInput) {
          auto res = builder.create<plier::CastOp>(loc, arg.getType(), arg);
          arg.replaceAllUsesExcept(res, res);
        } else {
          auto res = builder.create<plier::UndefOp>(loc, arg.getType());
          arg.replaceAllUsesWith(res);
        }
      }
      if (!hasInput) {
        argsToErase.emplace_back(i);
      }
      for (auto &use : block.getUses()) {
        auto op = use.getOwner();
        builder.setInsertionPoint(op);
        if (auto br = mlir::dyn_cast<mlir::BranchOp>(op)) {
          assert(&block == br.dest());
          auto src = br.destOperands()[i];
          auto new_op = builder.create<plier::CastOp>(op->getLoc(), type, src);
          br.destOperandsMutable().slice(i, 1).assign(new_op);
        } else if (auto cond_br = mlir::dyn_cast<mlir::CondBranchOp>(op)) {
          if (&block == cond_br.trueDest()) {
            auto src = cond_br.trueDestOperands()[i];
            auto new_op =
                builder.create<plier::CastOp>(op->getLoc(), type, src);
            cond_br.trueDestOperandsMutable().slice(i, 1).assign(new_op);
          }
          if (&block == cond_br.falseDest()) {
            auto src = cond_br.falseDestOperands()[i];
            auto new_op =
                builder.create<plier::CastOp>(op->getLoc(), type, src);
            cond_br.falseDestOperandsMutable().slice(i, 1).assign(new_op);
          }
        } else {
          llvm_unreachable("setBlockSig: unknown operation type");
        }
      }
      arg.setType(type);
    }
  }
  block.eraseArguments(argsToErase);
  return mlir::success();
}

mlir::LogicalResult convertRegionTypes(mlir::Region *region,
                                       mlir::TypeConverter &converter,
                                       bool apply) {
  assert(nullptr != region);
  if (region->empty()) {
    return mlir::failure();
  }

  mlir::OpBuilder builder(region->getContext());

  // Convert the arguments of each block within the region.
  auto sig = converter.convertBlockSignature(&region->front());
  assert(static_cast<bool>(sig));
  if (apply) {
    auto res = setBlockSig(region->front(), builder, *sig);
    if (mlir::failed(res)) {
      return mlir::failure();
    }
  }
  for (auto &block : llvm::make_early_inc_range(llvm::drop_begin(*region, 1))) {
    sig = converter.convertBlockSignature(&block);
    if (!sig) {
      return mlir::failure();
    }
    if (apply) {
      if (mlir::failed(setBlockSig(block, builder, *sig))) {
        return mlir::failure();
      }
    }
  }
  return mlir::success();
}
} // namespace

plier::FuncOpSignatureConversion::FuncOpSignatureConversion(
    mlir::TypeConverter &conv, mlir::MLIRContext *ctx)
    : OpRewritePattern(ctx), converter(conv) {}

mlir::LogicalResult plier::FuncOpSignatureConversion::matchAndRewrite(
    mlir::FuncOp funcOp, mlir::PatternRewriter &rewriter) const {
  auto oldFuncType = funcOp.getType();

  // Convert the original function types.
  mlir::TypeConverter::SignatureConversion result(oldFuncType.getNumInputs());
  llvm::SmallVector<mlir::Type, 1> newResults;
  if (mlir::failed(
          converter.convertSignatureArgs(oldFuncType.getInputs(), result)) ||
      mlir::failed(
          converter.convertTypes(oldFuncType.getResults(), newResults)) ||
      mlir::failed(convertRegionTypes(&funcOp.getBody(), converter, false))) {
    return mlir::failure();
  }

  auto newFuncType = mlir::FunctionType::get(
      funcOp.getContext(), result.getConvertedTypes(), newResults);
  if (newFuncType == oldFuncType) {
    return mlir::failure();
  }

  bool ret_type_changed = (newResults != oldFuncType.getResults());
  // Update the function signature in-place.
  rewriter.startRootUpdate(funcOp);
  funcOp.setType(newFuncType);
  auto res = convertRegionTypes(&funcOp.getBody(), converter, true);
  if (mlir::failed(res)) {
    funcOp.setType(oldFuncType);
    rewriter.cancelRootUpdate(funcOp);
    return mlir::failure();
  }

  if (ret_type_changed) {
    auto ret_types = funcOp.getType().getResults();
    funcOp.walk([&](mlir::ReturnOp ret) {
      if (ret->getParentOp() == funcOp) {
        mlir::OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(ret);
        for (auto it :
             llvm::enumerate(llvm::zip(ret.getOperandTypes(), ret_types))) {
          auto prev_type = std::get<0>(it.value());
          auto new_type = std::get<1>(it.value());
          if (prev_type != new_type) {
            auto index = static_cast<unsigned>(it.index());
            auto cast = rewriter.create<plier::CastOp>(ret.getLoc(), new_type,
                                                       ret.getOperand(index));
            rewriter.updateRootInPlace(ret,
                                       [&]() { ret.setOperand(index, cast); });
          }
        }
      }
    });
    auto mod = funcOp->getParentOfType<mlir::ModuleOp>();
    auto uses = funcOp.getSymbolUses(mod);
    if (uses) {
      for (auto use : *uses) {
        if (auto call = mlir::dyn_cast<mlir::CallOp>(use.getUser())) {
          rewriter.updateRootInPlace(call, [&]() {
            for (auto it : llvm::zip(call.getResults(), ret_types)) {
              auto res = std::get<0>(it);
              auto type = std::get<1>(it);
              res.setType(type);
            }
          });
        }
      }
    }
  }
  rewriter.finalizeRootUpdate(funcOp);
  return mlir::success();
}

plier::FixupIfYieldTypes::FixupIfYieldTypes(mlir::TypeConverter &typeConverter,
                                            mlir::MLIRContext *context)
    : OpRewritePattern(context), converter(typeConverter) {}

mlir::LogicalResult plier::FixupIfYieldTypes::matchAndRewrite(
    mlir::scf::IfOp op, mlir::PatternRewriter &rewriter) const {
  if (op->getNumResults() == 0) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Type> newTypes;
  newTypes.reserve(op->getNumResults());
  auto trueYield =
      mlir::cast<mlir::scf::YieldOp>(op.thenBlock()->getTerminator());
  auto falseYield =
      mlir::cast<mlir::scf::YieldOp>(op.elseBlock()->getTerminator());
  bool updateTrueYeild = false;
  bool updateFalseYeild = false;

  assert(trueYield->getNumOperands() == falseYield->getNumOperands());
  llvm::SmallVector<mlir::Type> newTrueTypes(trueYield->getNumOperands());
  llvm::SmallVector<mlir::Type> newFalseTypes(falseYield->getNumOperands());
  for (auto it : llvm::enumerate((llvm::zip(trueYield->getOperandTypes(),
                                            falseYield->getOperandTypes())))) {
    auto index = it.index();
    auto types = it.value();
    auto trueType = std::get<0>(types);
    auto falseType = std::get<1>(types);

    if (trueType != falseType) {
      assert((converter.convertType(trueType) != falseType) ||
             (converter.convertType(falseType) != trueType));
      if (converter.convertType(trueType) == falseType) {
        trueType = falseType;
        updateTrueYeild = true;
      } else if (converter.convertType(falseType) == trueType) {
        falseType = trueType;
        updateFalseYeild = true;
      } else {
        return mlir::failure();
      }
    }

    newTrueTypes[index] = trueType;
    newFalseTypes[index] = falseType;
  }

  auto updateYield = [](mlir::PatternRewriter &rewriter, mlir::scf::YieldOp op,
                        const llvm::SmallVector<mlir::Type> &newTypes) {
    rewriter.setInsertionPoint(op);
    llvm::SmallVector<mlir::Value> newValues(op->getNumOperands());
    for (auto it : llvm::enumerate(llvm::zip(op.results(), newTypes))) {
      auto index = it.index();
      auto value = it.value();
      auto result = std::get<0>(value);
      auto newType = std::get<1>(value);

      auto loc = rewriter.getUnknownLoc();

      if (result.getType() != newType) {
        result =
            rewriter.create<plier::CastOp>(loc, newType, result).getResult();
      }
      newValues[index] = result;
    }

    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, newValues);
  };

  if (updateTrueYeild)
    updateYield(rewriter, trueYield, newTrueTypes);

  if (updateFalseYeild)
    updateYield(rewriter, falseYield, newFalseTypes);

  return mlir::success(updateTrueYeild || updateFalseYeild);
}

plier::FixupIfTypes::FixupIfTypes(mlir::TypeConverter &typeConverter,
                                  mlir::MLIRContext *context)
    : OpRewritePattern(context), converter(typeConverter) {}

mlir::LogicalResult
plier::FixupIfTypes::matchAndRewrite(mlir::scf::IfOp op,
                                     mlir::PatternRewriter &rewriter) const {
  if (op->getNumResults() == 0) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Type> newTypes;
  newTypes.reserve(op->getNumResults());
  auto trueYield =
      mlir::cast<mlir::scf::YieldOp>(op.thenBlock()->getTerminator());
  auto falseYield =
      mlir::cast<mlir::scf::YieldOp>(op.elseBlock()->getTerminator());
  bool needUpdate = false;
  for (auto it : llvm::zip(op->getResultTypes(), trueYield->getOperandTypes(),
                           falseYield->getOperandTypes())) {
    auto retType = std::get<0>(it);
    auto trueType = std::get<1>(it);
    auto falseType = std::get<2>(it);
    if (trueType != falseType) {
      return mlir::failure();
    }

    if (retType == trueType) {
      newTypes.emplace_back(trueType);
    } else if (converter.convertType(retType) == trueType) {
      newTypes.emplace_back(trueType);
      needUpdate = true;
    } else {
      return mlir::failure();
    }
  }

  if (needUpdate) {
    rewriter.updateRootInPlace(op, [&]() {
      for (auto it : llvm::enumerate(op->getResults())) {
        it.value().setType(newTypes[it.index()]);
      }
    });
    return mlir::success();
  }
  return mlir::failure();
}

plier::FixCallOmittedArgs::FixCallOmittedArgs(
    mlir::TypeConverter &typeConverter, mlir::MLIRContext *context)
    : OpRewritePattern(context), converter(typeConverter) {}

mlir::LogicalResult plier::FixCallOmittedArgs::matchAndRewrite(
    mlir::CallOp op, mlir::PatternRewriter &rewriter) const {
  bool needChanges = false;
  llvm::SmallVector<bool> newResultsMask(op->getNumResults());
  llvm::SmallVector<mlir::Type> newResultsTypes;
  llvm::SmallVector<mlir::Type, 1> argTypes;
  for (auto it : llvm::enumerate(op.getResults())) {
    auto res = it.value();
    if (mlir::failed(converter.convertType(res.getType(), argTypes)) ||
        argTypes.size() > 1) {
      return mlir::failure();
    }
    if (!argTypes.empty()) {
      newResultsTypes.emplace_back(res.getType());
    } else {
      needChanges = true;
    }

    newResultsMask[it.index()] = argTypes.empty();
  }
  llvm::SmallVector<mlir::Value> newArgs;
  for (auto arg : op.operands()) {
    argTypes.clear();
    if (mlir::failed(converter.convertType(arg.getType(), argTypes)) ||
        argTypes.size() > 1) {
      return mlir::failure();
    }
    if (argTypes.empty()) {
      needChanges = true;
    } else {
      newArgs.emplace_back(arg);
    }
  }
  if (!needChanges) {
    return mlir::failure();
  }

  auto newOp = rewriter.create<mlir::CallOp>(op.getLoc(), op.getCallee(),
                                             newResultsTypes, newArgs);
  auto newOpResults = newOp.getResults();
  auto filterResults = [&]() {
    while (!newOpResults.empty()) {
      auto type = newOpResults.front().getType();
      argTypes.clear();
      auto res = converter.convertType(type, argTypes);
      assert(mlir::succeeded(res));
      (void)res;
      assert(argTypes.size() < 2);
      if (!argTypes.empty()) {
        break;
      }
      newOpResults = newOpResults.drop_front();
    }
  };
  mlir::Value undef;
  llvm::SmallVector<mlir::Value> newResults(op->getNumResults());
  filterResults();
  for (auto it : llvm::enumerate(newResultsMask)) {
    auto omitted = it.value();
    auto ind = it.index();
    if (omitted) {
      if (!undef) {
        undef = rewriter.create<plier::UndefOp>(op.getLoc(),
                                                op->getResultTypes()[0]);
      }
      newResults[ind] = undef;
    } else {
      newResults[ind] = newOpResults.front();
      newOpResults = newOpResults.drop_front();
      filterResults();
    }
  }
  assert((filterResults(), newOpResults.empty()));
  rewriter.replaceOp(op, newResults);
  return mlir::success();
}

namespace {
class ConvertSelectOp : public mlir::OpConversionPattern<mlir::SelectOp> {
public:
  using mlir::OpConversionPattern<mlir::SelectOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::SelectOp op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::SelectOp::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<mlir::SelectOp>(
        op, adaptor.condition(), adaptor.true_value(), adaptor.false_value());
    return mlir::success();
  }
};
} // namespace

void plier::populateControlFlowTypeConversionRewritesAndTarget(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {
  mlir::populateFuncOpTypeConversionPattern(patterns, typeConverter);
  target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getType()) &&
           typeConverter.isLegal(&op.getBody());
  });
  mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
  target.addDynamicallyLegalOp<mlir::CallOp>(
      [&](mlir::CallOp op) { return typeConverter.isLegal(op); });

  mlir::populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
  mlir::scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                             patterns, target);

  patterns.insert<ConvertSelectOp>(typeConverter, patterns.getContext());
  target.addDynamicallyLegalOp<mlir::SelectOp>(
      [&typeConverter](mlir::SelectOp op) {
        return typeConverter.isLegal(op);
      });

  target.markUnknownOpDynamicallyLegal([&](mlir::Operation *op) {
    return mlir::isNotBranchOpInterfaceOrReturnLikeOp(op) ||
           mlir::isLegalForBranchOpInterfaceTypeConversionPattern(
               op, typeConverter) ||
           mlir::isLegalForReturnOpTypeConversionPattern(op, typeConverter);
  });
}

namespace {
class BuildTupleConversionPattern
    : public mlir::OpConversionPattern<plier::BuildTupleOp> {
public:
  using OpConversionPattern<plier::BuildTupleOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::BuildTupleOp op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    plier::BuildTupleOp::Adaptor transformed(operands);
    auto retType =
        mlir::TupleType::get(op.getContext(), transformed.args().getTypes());
    rewriter.replaceOpWithNewOp<plier::BuildTupleOp>(op, retType,
                                                     transformed.args());
    return mlir::success();
  }
};

class GetItemTupleConversionPattern
    : public mlir::OpConversionPattern<plier::GetItemOp> {
public:
  using OpConversionPattern<plier::GetItemOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::GetItemOp op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    plier::GetItemOp::Adaptor transformed(operands);
    auto container = transformed.value();
    if (!container.getType().isa<mlir::TupleType>())
      return mlir::failure();

    auto &converter = *getTypeConverter();

    auto retType = converter.convertType(op.getType());
    if (!retType)
      return mlir::failure();

    auto index = transformed.index();

    rewriter.replaceOpWithNewOp<plier::GetItemOp>(op, retType, container,
                                                  index);
    return mlir::success();
  }
};
} // namespace

void plier::populateTupleTypeConversionRewritesAndTarget(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {
  patterns.insert<BuildTupleConversionPattern, GetItemTupleConversionPattern>(
      typeConverter, patterns.getContext());
  target.addDynamicallyLegalOp<plier::BuildTupleOp>(
      [&typeConverter](plier::BuildTupleOp op) {
        return typeConverter.isLegal(op.getResult().getType());
      });
}
