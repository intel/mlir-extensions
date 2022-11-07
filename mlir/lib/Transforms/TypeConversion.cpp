// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/TypeConversion.hpp"

#include "imex/Dialect/imex_util/Dialect.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Transforms.h>
#include <mlir/Transforms/DialectConversion.h>

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

using namespace mlir;
// Unpacks the single unrealized_conversion_cast using the list of inputs
// e.g., return [%b, %c, %d] for %a = unrealized_conversion_cast(%b, %c, %d)
static void unpackUnrealizedConversionCast(Value v,
                                           SmallVectorImpl<Value> &unpacked) {
  if (auto cast =
          dyn_cast_or_null<UnrealizedConversionCastOp>(v.getDefiningOp())) {
    if (cast.getInputs().size() != 1) {
      // 1 : N type conversion.
      unpacked.append(cast.getInputs().begin(), cast.getInputs().end());
      return;
    }
  }
  // 1 : 1 type conversion.
  unpacked.push_back(v);
}

// Need our own copy to workaround a bug in upstream
// https://github.com/llvm/llvm-project/issues/58742
class ConvertForOpTypes : public OpConversionPattern<scf::ForOp> {
public:
  ConvertForOpTypes(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<scf::ForOp>(typeConverter, context,
                                        /*benefit*/ 10) {}

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newResultTypes;
    SmallVector<unsigned> offsets;
    offsets.push_back(0);
    // Do the type conversion and record the offsets.
    for (Type type : op.getResultTypes()) {
      if (failed(typeConverter->convertTypes(type, newResultTypes)))
        return rewriter.notifyMatchFailure(op, "could not convert result");
      offsets.push_back(newResultTypes.size());
    }

    // Create a empty new op and inline the regions from the old op.
    //
    // This is a little bit tricky. We have two concerns here:
    //
    // 1. We cannot update the op in place because the dialect conversion
    // framework does not track type changes for ops updated in place, so it
    // won't insert appropriate materializations on the changed result types.
    // PR47938 tracks this issue, but it seems hard to fix. Instead, we need
    // to clone the op.
    //
    // 2. We need to resue the original region instead of cloning it, otherwise
    // the dialect conversion framework thinks that we just inserted all the
    // cloned child ops. But what we want is to "take" the child regions and let
    // the dialect conversion framework continue recursively into ops inside
    // those regions (which are already in its worklist; inlining them into the
    // new op's regions doesn't remove the child ops from the worklist).

    auto indexType = rewriter.getIndexType();
    TypeRange origBlockArgs = op.getLoopBody().front().getArgumentTypes();
    TypeConverter::SignatureConversion newSig(origBlockArgs.size());
    newSig.addInputs(0, indexType);
    if (failed(typeConverter->convertSignatureArgs(origBlockArgs.drop_front(),
                                                   newSig, 1)))
      return failure();

    // convertRegionTypes already takes care of 1:N conversion.
    if (failed(rewriter.convertRegionTypes(&op.getLoopBody(), *typeConverter,
                                           &newSig)))
      return failure();

    auto loc = op.getLoc();
    auto lBound = typeConverter->materializeSourceConversion(
        rewriter, loc, indexType, adaptor.getLowerBound());
    auto uBound = typeConverter->materializeSourceConversion(
        rewriter, loc, indexType, adaptor.getUpperBound());
    auto step = typeConverter->materializeSourceConversion(
        rewriter, loc, indexType, adaptor.getStep());
    if (!lBound || !uBound || !step)
      return failure();

    // Unpacked the iteration arguments.
    SmallVector<Value> flatArgs;
    for (Value arg : adaptor.getInitArgs())
      unpackUnrealizedConversionCast(arg, flatArgs);

    // We can not do clone as the number of result types after conversion might
    // be different.
    scf::ForOp newOp =
        rewriter.create<scf::ForOp>(loc, lBound, uBound, step, flatArgs);

    // Reserve whatever attributes in the original op.
    newOp->setAttrs(op->getAttrs());

    // We do not need the empty block created by rewriter.
    rewriter.eraseBlock(newOp.getBody(0));
    // Inline the type converted region from the original operation.
    rewriter.inlineRegionBefore(op.getLoopBody(), newOp.getLoopBody(),
                                newOp.getLoopBody().end());

    // Pack the return value.
    SmallVector<Value, 6> packedRets;
    for (unsigned i = 1, e = offsets.size(); i < e; i++) {
      unsigned start = offsets[i - 1], end = offsets[i];
      unsigned len = end - start;
      ValueRange mappedValue = newOp.getResults().slice(start, len);
      if (len != 1) {
        // 1 : N type conversion.
        Type origType = op.getResultTypes()[i - 1];
        Value mat = typeConverter->materializeSourceConversion(
            rewriter, loc, origType, mappedValue);
        if (!mat)
          return rewriter.notifyMatchFailure(
              op, "Failed to materialize 1:N type conversion");
        packedRets.push_back(mat);
      } else {
        // 1 : 1 type conversion.
        packedRets.push_back(mappedValue.front());
      }
    }

    rewriter.replaceOp(op, packedRets);
    return success();
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

  patterns.insert<ConvertSelectOp, ConvertForOpTypes>(typeConverter,
                                                      patterns.getContext());

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

void imex::populateTupleTypeConverter(mlir::TypeConverter &typeConverter) {
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
