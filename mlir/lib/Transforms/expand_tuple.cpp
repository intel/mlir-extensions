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

#include "mlir-extensions/Transforms/expand_tuple.hpp"

#include "mlir-extensions/Dialect/imex_util/dialect.hpp"
#include "mlir-extensions/Transforms/type_conversion.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

namespace {
static void flattenTuple(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ValueRange values,
                         llvm::SmallVectorImpl<mlir::Value> &ret) {
  for (auto arg : values) {
    if (auto tupleType = arg.getType().dyn_cast<mlir::TupleType>()) {
      for (auto it : llvm::enumerate(tupleType.getTypes())) {
        auto i = it.index();
        auto argType = it.value();
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
    flattenTuple(rewriter, loc, adaptor.operands(), newOperands);
    auto *operation = op.getOperation();
    rewriter.updateRootInPlace(op,
                               [&]() { operation->setOperands(newOperands); });
    return mlir::success();
  }
};

static mlir::Value reconstructTuple(mlir::OpBuilder &builder,
                                    mlir::Location loc,
                                    mlir::TupleType tupleType,
                                    mlir::ValueRange &values) {
  llvm::SmallVector<mlir::Value, 4> vals(tupleType.size());
  for (auto it : llvm::enumerate(tupleType.getTypes())) {
    auto i = it.index();
    auto type = it.value();
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
            return llvm::None;
          return mlir::success();
        });

    auto materializeTupleCast =
        [](mlir::OpBuilder &builder, mlir::TupleType type,
           mlir::ValueRange inputs,
           mlir::Location loc) -> llvm::Optional<mlir::Value> {
      if (auto ret = reconstructTuple(builder, loc, type, inputs))
        return ret;

      return llvm::None;
    };
    typeConverter.addArgumentMaterialization(materializeTupleCast);
    typeConverter.addSourceMaterialization(materializeTupleCast);
    typeConverter.addTargetMaterialization(materializeTupleCast);

    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);

    imex::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                             patterns, target);

    patterns.insert<ExpandTupleReturn>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> imex::createExpandTuplePass() {
  return std::make_unique<ExpandTuplePass>();
}
