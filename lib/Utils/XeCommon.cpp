//===- XeCommon.cpp -  --------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements XeTypeConverter and some other
/// routines used by Xe related dialects.
///
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include "imex/Dialect/XeGPU/IR/XeGPU.h"
#include "imex/Dialect/XeTile/IR/XeTileOps.h"
#include "imex/Utils/DebugUtils.h"
#include "imex/Utils/XeCommon.h"

namespace imex {

int getOperandIndex(mlir::Operation *op, mlir::Value operand) {
  for (auto [i, value] : llvm::enumerate(op->getOperands())) {
    if (operand == value)
      return i;
  }
  return -1;
};

mlir::BlockArgument getArgForOperand(mlir::scf::ForOp &op,
                                     mlir::Value operand) {
  auto idx = getOperandIndex(op, operand);
  auto numControls = op.getNumControlOperands();
  assert(idx >= (int)numControls);
  return op.getRegionIterArg(idx - numControls);
};

bool isSupportedModule(mlir::gpu::GPUModuleOp mod) {
  bool hasTileTyInFuncTy = false;
  mod.walk<mlir::WalkOrder::PreOrder>([&](mlir::gpu::GPUFuncOp op) {
    auto funcTy = op.getFunctionType();
    hasTileTyInFuncTy |= std::any_of(
        funcTy.getInputs().begin(), funcTy.getInputs().end(),
        [](mlir::Type ty) { return llvm::isa<imex::xetile::TileType>(ty); });
    hasTileTyInFuncTy |= std::any_of(
        funcTy.getResults().begin(), funcTy.getResults().end(),
        [](mlir::Type ty) { return llvm::isa<imex::xetile::TileType>(ty); });
  });

  return hasTileTyInFuncTy == false;
}

mlir::ValueRange buildUnrealizedCast(mlir::OpBuilder &builder,
                                     mlir::TypeRange resultTypes,
                                     mlir::ValueRange inputs) {
  mlir::Location loc = builder.getUnknownLoc();
  if (!inputs.empty())
    loc = inputs.front().getLoc();
  auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(
      loc, resultTypes, inputs);
  return castOp->getResults();
}

} // namespace imex
