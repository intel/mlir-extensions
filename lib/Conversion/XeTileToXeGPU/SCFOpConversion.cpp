//===- ArithOpConversion.cpp - XeTileToXeGPU conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the ArithOpConversionPattern, used in XeTileToXeGPU
/// conversion, converting the Arith Ops.
///
//===----------------------------------------------------------------------===//
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include "imex/Conversion/XeTileToXeGPU/XeTileToXeGPUConversion.h"

namespace imex {

struct SgSCFForOpBlockPattern
    : public SgXeTileToXeGPUConversion<mlir::scf::ForOp> {
  using SgXeTileToXeGPUConversion<mlir::scf::ForOp>::SgXeTileToXeGPUConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op, OpAdaptor adaptor,
                  imex::XeGPUOneToNPatterRewriter &rewriter) const override {
    auto loc = op.getLoc();

    llvm::SmallVector<mlir::Value> convertedArgs;
    // OpAdaptor is defined with ValueRange, so it contains results after
    // One-to-N mapping
    for (auto &values : adaptor.getInitArgs())
      convertedArgs.append(values.begin(), values.end());

    auto argumentTys = op.getRegion().getArgumentTypes();
    mlir::OneToNTypeMapping argumentMapping(argumentTys);
    // compute the type conversion (signature) for SCFFor body arguments.
    // argumentMapping is essentially a TypeConverter::SignatureConversion
    if (mlir::failed(
            typeConverter.computeTypeMapping(argumentTys, argumentMapping))) {
      op.emitOpError("Failed to compute the type mapping for arguments.\n");
      return mlir::failure();
    }

    // apply the signature convertion for SCFFor body arguments, an
    // UnrealizedConversionCastOp will be inserted by typeConverted by the
    // method registered in Materialization methods
    if (mlir::failed(rewriter.convertRegionTypes(&op.getRegion(), typeConverter,
                                                 &argumentMapping))) {
      op.emitOpError("Failed to convert region types.\n");
      return mlir::failure();
    }

    auto newOp = rewriter.create<mlir::scf::ForOp>(loc, op.getLowerBound(),
                                                   op.getUpperBound(),
                                                   op.getStep(), convertedArgs);

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
    for (auto arg : forOp.getInitArgs()) {
      auto type = arg.getType();
      result &= !type.isa<imex::xetile::TileType>();

      if (type.isa<mlir::VectorType>())
        result &= (type.cast<mlir::VectorType>().getRank() != 4);
    }
  }

  if (llvm::isa<mlir::scf::YieldOp>(op)) {
    auto yieldOp = llvm::cast<mlir::scf::YieldOp>(op);
    for (const auto arg : yieldOp.getResults()) {
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
