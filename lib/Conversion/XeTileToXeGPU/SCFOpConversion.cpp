//===- SCFOpConversion.cpp - XeTileToXeGPU conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the Conversion Patter for SCFOps.
///
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/SCF/IR/SCF.h>

#include "imex/Conversion/XeTileToXeGPU/XeTileToXeGPUConversion.h"

namespace imex {

struct SgSCFForOpBlockPattern
    : public SgXeTileToXeGPUConversion<mlir::scf::ForOp> {
  using SgXeTileToXeGPUConversion<mlir::scf::ForOp>::SgXeTileToXeGPUConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op, OpAdaptor adaptor,
                  imex::XeGPUOneToNPatterRewriter &rewriter) const override {
    // OpAdaptor is defined with ValueRange, so it contains results after
    // One-to-N mapping
    llvm::SmallVector<mlir::Value> convertedArgs;
    for (auto &values : adaptor.getInitArgs())
      convertedArgs.append(values.begin(), values.end());

    auto newOp = rewriter.create<mlir::scf::ForOp>(
        op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(),
        convertedArgs);

    // compute the type mapping (from origial to convereted) between
    // orginal args and converted args. The standard typeconverter
    // way doesnot work because array_length value is set in-the-fly based on
    // whether tile is create for load or not, thus a TileType could be
    // lowered into many different types of TensorDescType (due to different
    // setting of array_length). But typeconverter has no knowledge about when
    // to use array_lenght and when not.
    auto typeConverter = getTypeConverter<XeGPUTypeConverter>();
    auto argTys = op.getRegion().getArgumentTypes();
    mlir::OneToNTypeMapping argumentMapping(argTys);
    llvm::ArrayRef<mlir::Value> args(op.getRegion().getArguments().begin(),
                                     op.getRegion().getArguments().end());
    llvm::ArrayRef<mlir::Value> newArgs(
        newOp.getRegion().getArguments().begin(),
        newOp.getRegion().getArguments().end());
    auto status =
        typeConverter.computeTypeMapping(args, newArgs, argumentMapping);
    if (mlir::failed(status)) {
      llvm_unreachable("It is an unexpected failure of computing "
                       "type mapping for SCF::ForOp arguments.");
    }

    // apply the signature convertion for SCFFor body arguments, an
    // UnrealizedConversionCastOp will be inserted by typeConverter
    rewriter.applySignatureConversion(&op.getRegion(), argumentMapping);

    newOp.getBody()->erase();
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
    rewriter.replaceOp(op, newOp.getResults());
    return mlir::success();
  }
};

struct SgSCFYieldOpPattern
    : public SgXeTileToXeGPUConversion<mlir::scf::YieldOp> {
  using SgXeTileToXeGPUConversion<
      mlir::scf::YieldOp>::SgXeTileToXeGPUConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op, OpAdaptor adaptor,
                  imex::XeGPUOneToNPatterRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> convertedResults;
    for (auto &values : adaptor.getResults())
      convertedResults.append(values.begin(), values.end());

    auto newOp =
        rewriter.create<mlir::scf::YieldOp>(op.getLoc(), convertedResults);

    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

bool isLegalSCFOp(mlir::Operation *op) {
  bool result = true;
  if (llvm::isa<mlir::scf::ForOp>(op)) {
    auto forOp = llvm::cast<mlir::scf::ForOp>(op);
    for (const auto &arg : forOp.getInitArgs()) {
      auto type = arg.getType();
      result &= !type.isa<imex::xetile::TileType>();

      if (type.isa<mlir::VectorType>())
        result &= (type.cast<mlir::VectorType>().getRank() != 4);
    }
  }

  if (llvm::isa<mlir::scf::YieldOp>(op)) {
    auto yieldOp = llvm::cast<mlir::scf::YieldOp>(op);
    for (const auto &arg : yieldOp.getResults()) {
      auto type = arg.getType();
      result &= !type.isa<imex::xetile::TileType>();
      if (type.isa<mlir::VectorType>())
        result &= (type.cast<mlir::VectorType>().getRank() != 4);
    }
  }
  return result;
}

void populateSCFOpConversionPatterns(imex::XeGPUTypeConverter &converter,
                                     mlir::RewritePatternSet &patterns) {
  patterns.add<SgSCFForOpBlockPattern, SgSCFYieldOpPattern>(
      patterns.getContext(), converter);
}

} // namespace imex
