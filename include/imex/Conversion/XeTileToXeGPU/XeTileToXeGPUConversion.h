//===- TypeConverter.h - XeTileToXeGPU conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the XeOneToNConversion, the base class for
/// XeTileToXeGPU conversion, XeOneToNTypeConverter, converting types used in
/// XeTile dialect to types used in XeGPU dialect, XeOneToNPatternRewriter a
/// wrapper around ConversionPatterRewriter providng interface for supporting
/// OneToN replace.
///
//===----------------------------------------------------------------------===//

#ifndef _XeTileToXeGPUConversion_H_INCLUDED_
#define _XeTileToXeGPUConversion_H_INCLUDED_

#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/XeGPU/IR/XeGPU.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/OneToNTypeConversion.h>

#include "imex/Dialect/XeTile/IR/XeTileOps.h"
#include "imex/Utils/DebugUtils.h"
#include "imex/Utils/PassWrapper.h"
#include "imex/Utils/XeCommon.h"

namespace imex {

class XeOneToNTypeConverter : public imex::XeTypeConverter {
public:
  XeOneToNTypeConverter(mlir::MLIRContext &context);

  std::optional<mlir::LogicalResult>
  convertTileType(xetile::TileType tileTy,
                  llvm::SmallVectorImpl<mlir::Type> &resultTypes) override;

  std::optional<mlir::LogicalResult>
  convertVectorType(mlir::VectorType vectorTy,
                    llvm::SmallVectorImpl<mlir::Type> &resultTypes) override;

  mlir::LogicalResult computeTypeMapping(mlir::ValueRange original,
                                         mlir::ValueRange converted,
                                         mlir::OneToNTypeMapping &resultMap);

private:
  mlir::Operation *targetOp;
};

class XeOneToNPatternRewriter : public mlir::PatternRewriter,
                                public mlir::RewriterBase::Listener {
public:
  explicit XeOneToNPatternRewriter(mlir::ConversionPatternRewriter &rewriter,
                                   XeOneToNTypeConverter &converter)
      : mlir::PatternRewriter(rewriter.getContext()), typeConverter(converter),
        rewriter(rewriter) {
    setListener(this);
  }

  mlir::Block *
  applySignatureConversion(mlir::Block *block,
                           mlir::TypeConverter::SignatureConversion &conversion,
                           const mlir::TypeConverter *converter = nullptr);

  template <typename OpTy, typename... Args>
  OpTy create(mlir::Location location, Args &&...args) {
    return rewriter.create<OpTy>(location, std::forward<Args>(args)...);
  }

  mlir::FailureOr<mlir::Block *> convertRegionTypes(
      mlir::Region *region, const mlir::TypeConverter &converter,
      mlir::TypeConverter::SignatureConversion *entryConversion = nullptr) {
    return rewriter.convertRegionTypes(region, converter, entryConversion);
  }

  void inlineRegionBefore(mlir::Region &region, mlir::Region &parent,
                          mlir::Region::iterator before) {
    rewriter.inlineRegionBefore(region, parent, before);
  }

  void replaceOp(mlir::Operation *op, mlir::Operation *newOp) override {
    assert(op && newOp && "expected non-null op");
    replaceOp(op, newOp->getResults());
  }

  void replaceOp(mlir::Operation *op, mlir::ValueRange newValues) override;

  void replaceOp(mlir::Operation *op, mlir::ValueRange newValues,
                 const mlir::OneToNTypeMapping &resultMapping);

  void eraseOp(mlir::Operation *op) override { rewriter.eraseOp(op); }

  void eraseBlock(mlir::Block *block) override { rewriter.eraseBlock(block); }

  template <typename CallableT>
  void modifyOpInPlace(mlir::Operation *root, CallableT &&callable) {
    rewriter.modifyOpInPlace(root, callable);
  }

  mlir::ConversionPatternRewriter &mlirConversionPatterRewriter() {
    return rewriter;
  };

private:
  XeOneToNTypeConverter &typeConverter;
  mlir::ConversionPatternRewriter &rewriter;
};

template <typename SourceOp>
class XeOneToNConversion : public XeConversionPattern<TileUsageAnalysis> {
public:
  XeOneToNConversion(mlir::MLIRContext *context,
                     XeOneToNTypeConverter &typeConverter,
                     TileUsageAnalysis &analysis,
                     mlir::PatternBenefit benefit = 1)
      : XeConversionPattern(typeConverter, analysis,
                            SourceOp::getOperationName(), benefit, context) {}

  using RangeT = llvm::ArrayRef<mlir::ValueRange>;
  using OpAdaptor = typename SourceOp::template GenericAdaptor<RangeT>;

  /*
   * This overwrites the RewritePattern::matchAndRewrite as it is the entry
   * point. It will set up the OpAdaptor such that it contains the converted
   * values, and wrap the ConversionPatternRewriter with
   * XeOneToNPatternRewriter to provide a clean interface for users.
   */
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const final {
    llvm::SmallVector<mlir::ValueRange> convertedValues;

    // converted into convertionPatternRewriter since applyPartialConversion
    // used it
    auto &convertionPatternRewriter =
        static_cast<mlir::ConversionPatternRewriter &>(rewriter);

    // One-To-One mapping provided by mlir::ConversionPatternRewriter.
    // remappedValues contains new values for each operand of the operation. It
    // is supposed to be a UnrealizedConversionCastOp (created by the replaceOp
    // of XeGPUXeOneToNPatternRewriter in form of cast newvalues to oldType) for
    // each operand that has One-to-N mapping.
    llvm::SmallVector<mlir::Value> remappedValues;
    if (mlir::failed(convertionPatternRewriter.getRemappedValues(
            op->getOperands(), remappedValues))) {
      return op->emitOpError("Failed to get remapped values.\n");
    }

    // retrive mapped values for each operand. If its type is not convereted
    // (convertedTypes.size() == 1) we will reuse the current value. Otherwise,
    // it has one-to-n mapping, and the new value should be an
    // UnrealizedConversionCastOp.
    for (size_t i = 0; i < remappedValues.size(); i++) {
      auto value = remappedValues[i];
      auto castOp = value.getDefiningOp<mlir::UnrealizedConversionCastOp>();
      auto valueTy = value.getType();
      if (castOp && valueTy == op->getOperand(i).getType())
        convertedValues.push_back(castOp.getInputs());
      else
        convertedValues.push_back(remappedValues[i]);
    }

    auto sourceOp = llvm::dyn_cast<SourceOp>(op);
    OpAdaptor adaptor(convertedValues, sourceOp);
    XeOneToNPatternRewriter OneToNRewriter(
        convertionPatternRewriter, getTypeConverter<XeOneToNTypeConverter>());
    return matchAndRewrite(sourceOp, adaptor, OneToNRewriter);
  }

  virtual mlir::LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const {
    llvm_unreachable("must override matchAndRewrite or a rewrite method");
  }
};

} // namespace imex

#endif
