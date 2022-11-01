// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/CastUtils.hpp"
#include "imex/Dialect/imex_util/Dialect.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>

namespace {
mlir::Type makeSignless(mlir::Type type) {
  if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    if (!intType.isSignless()) {
      return mlir::IntegerType::get(intType.getContext(), intType.getWidth());
    }
  }
  return type;
}
} // namespace

mlir::Value imex::indexCast(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value val, mlir::Type dstType) {
  auto srcType = val.getType();
  assert(srcType.isa<mlir::IndexType>() || dstType.isa<mlir::IndexType>());
  if (srcType == dstType)
    return val;

  auto newSrcType = makeSignless(srcType);
  if (newSrcType != srcType)
    val = builder.createOrFold<imex::util::SignCastOp>(loc, newSrcType, val);

  auto newDstType = makeSignless(dstType);
  val = builder.createOrFold<mlir::arith::IndexCastOp>(loc, newDstType, val);
  if (newDstType != dstType)
    val = builder.createOrFold<imex::util::SignCastOp>(loc, dstType, val);

  return val;
}

mlir::Value imex::indexCast(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value src) {
  return indexCast(builder, loc, src,
                   mlir::IndexType::get(builder.getContext()));
}

mlir::Type imex::makeSignlessType(mlir::Type type) {
  if (auto intType = type.dyn_cast<mlir::IntegerType>())
    return makeSignlessType(intType);

  return type;
}

mlir::IntegerType imex::makeSignlessType(mlir::IntegerType type) {
  if (!type.isSignless())
    return mlir::IntegerType::get(type.getContext(), type.getWidth());

  return type;
}
