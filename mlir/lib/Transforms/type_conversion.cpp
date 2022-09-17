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

#include "mlir-extensions/Transforms/type_conversion.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Transforms.h>
#include <mlir/Transforms/DialectConversion.h>

#include "mlir-extensions/Dialect/imex_util/dialect.hpp"

namespace {
class ConvertSelectOp
    : public mlir::OpConversionPattern<mlir::arith::SelectOp> {
public:
  using mlir::OpConversionPattern<mlir::arith::SelectOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op,
                  mlir::arith::SelectOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(
        op, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return mlir::success();
  }
};
} // namespace

void imex::populateControlFlowTypeConversionRewritesAndTarget(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {
  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
      patterns, typeConverter);
  target.addDynamicallyLegalOp<mlir::func::FuncOp>(
      [&](mlir::func::FuncOp op) -> llvm::Optional<bool> {
        if (typeConverter.isSignatureLegal(op.getFunctionType()) &&
            typeConverter.isLegal(&op.getBody()))
          return true;

        return llvm::None;
      });

  mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
  target.addDynamicallyLegalOp<mlir::arith::SelectOp, mlir::func::CallOp>(
      [&](mlir::Operation *op) -> llvm::Optional<bool> {
        if (typeConverter.isLegal(op))
          return true;

        return llvm::None;
      });

  mlir::populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
  mlir::scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                             patterns, target);

  patterns.insert<ConvertSelectOp>(typeConverter, patterns.getContext());

  target.markUnknownOpDynamicallyLegal(
      [&](mlir::Operation *op) -> llvm::Optional<bool> {
        if (mlir::isNotBranchOpInterfaceOrReturnLikeOp(op) ||
            mlir::isLegalForBranchOpInterfaceTypeConversionPattern(
                op, typeConverter) ||
            mlir::isLegalForReturnOpTypeConversionPattern(op, typeConverter))
          return true;

        return llvm::None;
      });
}

namespace {
struct BuildTupleConversionPattern
    : public mlir::OpConversionPattern<imex::util::BuildTupleOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::BuildTupleOp op,
                  imex::util::BuildTupleOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto retType =
        mlir::TupleType::get(op.getContext(), adaptor.getArgs().getTypes());
    rewriter.replaceOpWithNewOp<imex::util::BuildTupleOp>(op, retType,
                                                          adaptor.getArgs());
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

struct GetItemTupleConversionPattern
    : public mlir::OpConversionPattern<imex::util::TupleExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::TupleExtractOp op,
                  imex::util::TupleExtractOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto container = adaptor.getSource();
    auto containerType = container.getType().dyn_cast<mlir::TupleType>();
    if (!containerType || containerType.size() == 0)
      return mlir::failure();

    auto &converter = *getTypeConverter();

    auto retType = converter.convertType(op.getType());
    if (!retType)
      return mlir::failure();

    auto index = adaptor.getIndex();
    if (isUniTuple(containerType)) {
      if (retType != containerType.getType(0))
        return mlir::failure();
    } else {
      auto constIndex = mlir::getConstantIntValue(index);
      if (!constIndex)
        return mlir::failure();

      auto i = *constIndex;
      if (i < 0 || i >= static_cast<int64_t>(containerType.size()) ||
          containerType.getType(i) != retType)
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<imex::util::TupleExtractOp>(op, retType,
                                                            container, index);
    return mlir::success();
  }
};
} // namespace

void imex::populateTupleTypeConverter(mlir::MLIRContext & /*context*/,
                                      mlir::TypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&typeConverter](mlir::TupleType type) -> llvm::Optional<mlir::Type> {
        auto count = static_cast<unsigned>(type.size());
        llvm::SmallVector<mlir::Type> newTypes(count);
        bool changed = false;
        for (auto i : llvm::seq(0u, count)) {
          auto oldType = type.getType(i);
          auto newType = typeConverter.convertType(oldType);
          if (!newType)
            return llvm::None;

          changed = changed || (newType != oldType);
          newTypes[i] = newType;
        }
        if (!changed)
          return llvm::None;

        auto ret = mlir::TupleType::get(type.getContext(), newTypes);
        assert(ret != type);
        return ret;
      });
}

void imex::populateTupleTypeConversionRewritesAndTarget(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {
  patterns.insert<BuildTupleConversionPattern, GetItemTupleConversionPattern>(
      typeConverter, patterns.getContext());

  target.addDynamicallyLegalOp<imex::util::BuildTupleOp>(
      [&typeConverter](imex::util::BuildTupleOp op) {
        return typeConverter.isLegal(op.getResult().getType());
      });

  target.addDynamicallyLegalOp<imex::util::TupleExtractOp>(
      [&typeConverter](imex::util::TupleExtractOp op) -> llvm::Optional<bool> {
        auto inputType = op.getSource().getType();
        auto tupleType = typeConverter.convertType(inputType)
                             .dyn_cast_or_null<mlir::TupleType>();
        if (!tupleType)
          return llvm::None;

        auto srcType = [&]() -> mlir::Type {
          if (auto index = mlir::getConstantIntValue(op.getIndex())) {
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
          return llvm::None;

        auto dstType = op.getType();
        return inputType == tupleType && srcType == dstType &&
               dstType == typeConverter.convertType(dstType);
      });
}
