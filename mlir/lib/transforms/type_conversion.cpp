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

#include "plier/transforms/type_conversion.hpp"

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/SCF/Transforms.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/StandardOps/Transforms/FuncConversions.h>
#include <mlir/Transforms/DialectConversion.h>

#include "plier/dialect/plier/dialect.hpp"

namespace {
class ConvertSelectOp : public mlir::OpConversionPattern<mlir::SelectOp> {
public:
  using mlir::OpConversionPattern<mlir::SelectOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::SelectOp op, mlir::SelectOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::SelectOp>(op, adaptor.getCondition(),
                                                adaptor.getTrueValue(),
                                                adaptor.getFalseValue());
    return mlir::success();
  }
};
} // namespace

void plier::populateControlFlowTypeConversionRewritesAndTarget(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {
  mlir::populateFunctionLikeTypeConversionPattern<mlir::FuncOp>(patterns,
                                                                typeConverter);
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
  matchAndRewrite(plier::BuildTupleOp op, plier::BuildTupleOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto retType =
        mlir::TupleType::get(op.getContext(), adaptor.args().getTypes());
    rewriter.replaceOpWithNewOp<plier::BuildTupleOp>(op, retType,
                                                     adaptor.args());
    return mlir::success();
  }
};

class GetItemTupleConversionPattern
    : public mlir::OpConversionPattern<plier::GetItemOp> {
public:
  using OpConversionPattern<plier::GetItemOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::GetItemOp op, plier::GetItemOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto container = adaptor.value();
    if (!container.getType().isa<mlir::TupleType>())
      return mlir::failure();

    auto &converter = *getTypeConverter();

    auto retType = converter.convertType(op.getType());
    if (!retType)
      return mlir::failure();

    auto index = adaptor.index();

    rewriter.replaceOpWithNewOp<plier::GetItemOp>(op, retType, container,
                                                  index);
    return mlir::success();
  }
};

static bool isUniTuple(mlir::TupleType type) {
  auto count = type.size();
  if (count == 0)
    return false;

  auto elemType = type.getType(0);
  for (auto i : llvm::seq<size_t>(1, count)) {
    if (type.getType(i) != elemType)
      return false;
  }
  return true;
}
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

  target.addDynamicallyLegalOp<plier::GetItemOp>(
      [&typeConverter](plier::GetItemOp op) -> llvm::Optional<bool> {
        auto inputType = op.value().getType();
        if (auto tupleType = typeConverter.convertType(inputType)
                                 .dyn_cast_or_null<mlir::TupleType>()) {
          auto srcType = [&]() -> mlir::Type {
            if (auto index = mlir::getConstantIntValue(op.index())) {
              auto i = *index;
              auto size = static_cast<unsigned>(tupleType.size());
              if (i >= 0 && i < size)
                return tupleType.getType(static_cast<size_t>(i));
            } else if (isUniTuple(tupleType)) {
              return tupleType.getType(0);
            }
            return {};
          }();
          if (!srcType)
            return false;

          auto dstType = op.getType();
          return srcType == dstType &&
                 dstType == typeConverter.convertType(dstType);
        }

        return llvm::None;
      });
}
