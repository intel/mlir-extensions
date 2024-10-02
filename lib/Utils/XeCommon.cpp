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
#include "llvm/Support/FormatVariadic.h"

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

/// @brief
/// We have to use flattened i32 for intrinsic calls like llvm_genx_raw_send2_*,
/// hence we need to encode vectors with arbitrary datatypes as i32.
/// keepF16 = true: when the vector element type is f16, it disables flattening
/// it to i32.
std::pair<std::string, mlir::VectorType>
encodeVectorType(mlir::ConversionPatternRewriter &rewriter,
                 mlir::VectorType type, bool use64bitData, bool enforceInteger,
                 bool keepF16) {
  mlir::Type srcElemType = type.getElementType();
  assert((srcElemType.isF16() || srcElemType.isBF16() || srcElemType.isF32() ||
          srcElemType.isInteger(8) || srcElemType.isInteger(16) ||
          srcElemType.isInteger(32) || srcElemType.isInteger(64)) &&
         "Unsupported vector element type.");
  const uint32_t srcBitWidth = srcElemType.getIntOrFloatBitWidth();
  mlir::Type resElemType = rewriter.getI32Type();
  if (!enforceInteger) {
    if (srcElemType.isF32() || (keepF16 && srcElemType.isF16())) {
      resElemType = srcElemType;
    }
    if (use64bitData) {
      resElemType = rewriter.getI64Type();
    }
  }
  const uint32_t resBitWidth = resElemType.getIntOrFloatBitWidth();
  const uint32_t resVecSize =
      (type.getNumElements() * srcBitWidth) / resBitWidth;
  mlir::VectorType resVecType = mlir::VectorType::get(resVecSize, resElemType);
  std::string resStr =
      llvm::formatv("v{0}{1}{2}", resVecSize,
                    ((resElemType.isF32() || (keepF16 && resElemType.isF16()))
                         ? 'f'
                         : 'i'),
                    resBitWidth)
          .str();
  return {resStr, resVecType};
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

/// Creates the default strides for the given `shape`. Example:
///   input shape = 2x3x4x5
///   output strides = 60x20x5x1
llvm::SmallVector<int64_t> defaultStrides(llvm::ArrayRef<int64_t> shape) {
  int64_t stride = 1;
  llvm::SmallVector<int64_t> strides;
  for (int64_t size : llvm::reverse(shape)) {
    strides.push_back(stride);
    stride *= size;
  }
  std::reverse(strides.begin(), strides.end());
  return strides;
}

} // namespace imex
