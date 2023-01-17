// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/ExpandTuple.hpp"

#include "imex/Dialect/imex_util/Dialect.hpp"
#include "imex/Transforms/TypeConversion.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

namespace {
static void flattenTuple(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ValueRange values,
                         llvm::SmallVectorImpl<mlir::Value> &ret) {
  for (auto arg : values) {
    if (auto tupleType = arg.getType().dyn_cast<mlir::TupleType>()) {
      for (auto [i, argType] : llvm::enumerate(tupleType.getTypes())) {
        auto ind = builder.createOrFold<mlir::arith::ConstantIndexOp>(loc, i);
        auto res = builder.createOrFold<imex::util::TupleExtractOp>(
            loc, argType, arg, ind);
        flattenTuple(builder, loc, res, ret);
      }
    } else {
      ret.emplace_back(arg);
    }
  }
}

struct ExpandTupleReturn
    : public mlir::OpConversionPattern<mlir::func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::ReturnOp op,
                  mlir::func::ReturnOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> newOperands;
    auto loc = op.getLoc();
    flattenTuple(rewriter, loc, adaptor.getOperands(), newOperands);
    auto *operation = op.getOperation();
    rewriter.updateRootInPlace(op,
                               [&]() { operation->setOperands(newOperands); });
    return mlir::success();
  }
};

class ExpandEnvRegionYield
    : public mlir::OpConversionPattern<imex::util::EnvironmentRegionYieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::EnvironmentRegionYieldOp op,
                  imex::util::EnvironmentRegionYieldOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> newOperands;
    auto loc = op.getLoc();
    flattenTuple(rewriter, loc, adaptor.getResults(), newOperands);

    rewriter.replaceOpWithNewOp<imex::util::EnvironmentRegionYieldOp>(
        op, newOperands);
    return mlir::success();
  }
};

static mlir::Value reconstructTuple(mlir::OpBuilder &builder,
                                    mlir::Location loc,
                                    mlir::TupleType tupleType,
                                    mlir::ValueRange &values) {
  llvm::SmallVector<mlir::Value, 4> vals(tupleType.size());
  for (auto [i, type] : llvm::enumerate(tupleType.getTypes())) {
    if (auto innerTuple = type.dyn_cast<mlir::TupleType>()) {
      vals[i] = reconstructTuple(builder, loc, innerTuple, values);
    } else {
      if (values.empty())
        return {};

      vals[i] = values.front();
      values = values.drop_front();
    }
  }
  return builder.create<imex::util::BuildTupleOp>(loc, tupleType, vals);
}

struct ExpandTuplePass
    : public mlir::PassWrapper<ExpandTuplePass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandTuplePass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::util::ImexUtilDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    mlir::TypeConverter typeConverter;
    // Convert unknown types to itself
    typeConverter.addConversion([](mlir::Type type) { return type; });
    typeConverter.addConversion(
        [&typeConverter](mlir::TupleType type,
                         llvm::SmallVectorImpl<mlir::Type> &ret)
            -> llvm::Optional<mlir::LogicalResult> {
          if (mlir::failed(typeConverter.convertTypes(type.getTypes(), ret)))
            return std::nullopt;
          return mlir::success();
        });

    auto materializeTupleCast =
        [](mlir::OpBuilder &builder, mlir::TupleType type,
           mlir::ValueRange inputs,
           mlir::Location loc) -> llvm::Optional<mlir::Value> {
      if (auto ret = reconstructTuple(builder, loc, type, inputs))
        return ret;

      return std::nullopt;
    };
    typeConverter.addArgumentMaterialization(materializeTupleCast);
    typeConverter.addSourceMaterialization(materializeTupleCast);
    typeConverter.addTargetMaterialization(materializeTupleCast);

    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);

    imex::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                             patterns, target);

    patterns.insert<ExpandTupleReturn, ExpandEnvRegionYield>(typeConverter,
                                                             context);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> imex::createExpandTuplePass() {
  return std::make_unique<ExpandTuplePass>();
}
