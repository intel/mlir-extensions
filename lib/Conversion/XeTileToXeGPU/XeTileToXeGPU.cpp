//===- XeTileToXeGPU.cpp - XeTileToXeGPU conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the XeTileToXeGPU conversion, converting the XeTile
/// dialect to the XeGPU dialect.
///
//===----------------------------------------------------------------------===//

#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/Passes.h>

#include "../PassDetail.h"
#include "ArithOpConversion.h"
#include "SCFOpConversion.h"
#include "XeTileOpConversion.h"

namespace imex {

class XeTileConversionTarget : public mlir::ConversionTarget {
public:
  explicit XeTileConversionTarget(mlir::MLIRContext &context)
      : mlir::ConversionTarget(context) {
    addIllegalOp<imex::xetile::InitTileOp>();

    addLegalOp<mlir::UnrealizedConversionCastOp>();

    addLegalDialect<imex::xegpu::XeGPUDialect>();

    addDynamicallyLegalDialect<mlir::arith::ArithDialect>(
        [&](mlir::Operation *op) { return isLegalArithOp(op); });

    addDynamicallyLegalDialect<mlir::scf::SCFDialect>(
        [&](mlir::Operation *op) { return isLegalSCFOp(op); });
  }
};

// Full Pass
struct ConvertXeTileToXeGPUPass // convert XeTile to XeGPU
    : public ::imex::ConvertXeTileToXeGPUBase<ConvertXeTileToXeGPUPass> {
  ConvertXeTileToXeGPUPass() = default;

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mlir::MLIRContext &context = getContext();

    // skip functions with XeTile.TileType inputs and outputs
    bool hasTileTyInFuncTy = false;
    mod.walk<mlir::WalkOrder::PreOrder>([&](mlir::func::FuncOp op) {
      auto funcTy = op.getFunctionType();
      hasTileTyInFuncTy |= std::any_of(
          funcTy.getInputs().begin(), funcTy.getInputs().end(),
          [](mlir::Type ty) { return llvm::isa<imex::xetile::TileType>(ty); });
      hasTileTyInFuncTy |= std::any_of(
          funcTy.getResults().begin(), funcTy.getInputs().end(),
          [](mlir::Type ty) { return llvm::isa<imex::xetile::TileType>(ty); });
    });

    if (hasTileTyInFuncTy) {
      mod.emitOpError(
          "Currently FunctionType with xetile.TileType is not supported.");
      return signalPassFailure();
    }

    imex::ValueAttributeMap map;
    mod.walk<mlir::WalkOrder::PreOrder>([&](imex::xetile::TileMMAOp op) {
      markDefChainValues(op.getA(), OperandType::DPASA, map);
      markDefChainValues(op.getB(), OperandType::DPASB, map);
      markDefChainValues(op.getC(), OperandType::DPASC, map);
      markUseChainValues(op.getOutput(), OperandType::DPASR, map);
    });

    XeGPUTypeConverter typeConverter(context, map);
    XeTileConversionTarget target(context);
    mlir::RewritePatternSet patterns(&context);

    populateXeTileToXeGPUConversionPatterns(typeConverter, patterns);

    if (mlir::failed(
            mlir::applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }
};

/// Populate the given list with patterns that convert XeTile to XeGPU
void populateXeTileToXeGPUConversionPatterns(
    imex::XeGPUTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  populateSCFOpConversionPatterns(converter, patterns);
  populateArithOpConversionPatterns(converter, patterns);
  populateXeTileOpConversionPatterns(converter, patterns);
}

/// Create a pass that convert XeTile to XeGPU
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertXeTileToXeGPUPass() {
  return std::make_unique<ConvertXeTileToXeGPUPass>();
}

} // namespace imex
