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
}

mlir::BlockArgument getArgForOperand(mlir::scf::ForOp &op,
                                     mlir::Value operand) {
  auto idx = getOperandIndex(op, operand);
  auto numControls = op.getNumControlOperands();
  assert(idx >= (int)numControls);
  return op.getRegionIterArg(idx - numControls);
}

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

std::pair<std::string, mlir::VectorType>
encodeVectorType(mlir::ConversionPatternRewriter &rewriter,
                 mlir::VectorType type, bool use64bitData,
                 bool enforceInteger) {
  auto elemType = type.getElementType();
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  int size = type.getNumElements() * bitWidth / 32;
  if (use64bitData) {
    size /= 2;
  }
  std::string str;
  switch (size) {
  case 16:
    str += "v16";
    break;
  case 32:
    str += "v32";
    break;
  case 64:
    str += "v64";
    break;
  case 128:
    str += "v128";
    break;
  case 256:
    str += "v256";
    break;
  case 512:
    str += "v512";
    break;
  default:
    assert(0 && "add more support");
    break;
  }
  if (use64bitData) {
    str += "i64";
    elemType = rewriter.getI64Type();
  } else if (enforceInteger) {
    str += "i32";
    elemType = rewriter.getI32Type();
  } else if (elemType == rewriter.getF32Type())
    str += "f32";
  else if (elemType == rewriter.getF16Type()) {
    str += "i32";
    elemType = rewriter.getI32Type();
  } else if (elemType == rewriter.getBF16Type()) {
    str += "i32";
    elemType = rewriter.getI32Type();
  } else if (elemType == rewriter.getI32Type()) {
    str += "i32";
  } else
    assert(0 && "add more support");
  auto newType = mlir::VectorType::get(size, elemType);
  return std::make_pair(str, newType);
}

/// @brief
/// We have to use i32 for intrinsic calls like llvm_genx_raw_send2_*, if we
/// want to get the original element type (e.g., f16) as the result of a load,
/// we have to encode the resulting i32 vector back to it.
mlir::VectorType encodeVectorTypeTo(mlir::VectorType currentVecType,
                                    mlir::Type toElemType) {
  auto elemType = currentVecType.getElementType();
  auto currentbitWidth = elemType.getIntOrFloatBitWidth();
  auto newBitwidth = toElemType.getIntOrFloatBitWidth();
  const int size =
      currentVecType.getNumElements() * currentbitWidth / newBitwidth;
  return mlir::VectorType::get(size, toElemType);
}

unsigned encodeDataum(mlir::Type type) {
  switch (type.getIntOrFloatBitWidth()) {
  case 8:
    return 1;
  case 16:
    return 2;
  case 32:
    return 3;
  case 64:
    return 4;
  default:
    assert(0 && "add more support");
    return 0;
  }
}

unsigned encodeOpcode(mlir::arith::AtomicRMWKind kind) {
  unsigned encode = 0;
  switch (kind) {
  case mlir::arith::AtomicRMWKind::addf:
    encode = 19;
    break;
  case mlir::arith::AtomicRMWKind::addi:
    encode = 12;
    break;
  case mlir::arith::AtomicRMWKind::assign:
    encode = 10;
    break;
  // case mlir::arith::AtomicRMWKind::maxf:
  //   encode = 22;
  //   break;
  case mlir::arith::AtomicRMWKind::maxs:
    encode = 15;
    break;
  case mlir::arith::AtomicRMWKind::maxu:
    encode = 17;
    break;
  // case mlir::arith::AtomicRMWKind::minf:
  //   encode = 21;
  //   break;
  case mlir::arith::AtomicRMWKind::mins:
    encode = 14;
    break;
  case mlir::arith::AtomicRMWKind::minu:
    encode = 16;
    break;
  case mlir::arith::AtomicRMWKind::ori:
    encode = 25;
    break;
  case mlir::arith::AtomicRMWKind::andi:
    encode = 24;
    break;
  case mlir::arith::AtomicRMWKind::mulf:
  case mlir::arith::AtomicRMWKind::muli:
    assert(0 && "Atomic operation not supported!");
    break;
  default:
    assert(0 && "to be supported");
    break;
  }
  return encode;
}

} // namespace imex
