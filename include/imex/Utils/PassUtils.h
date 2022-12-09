//===- PassUtils.h - Pass Utility Functions --------------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utility functions for writing passes.
//
//===----------------------------------------------------------------------===//

#ifndef _IMEX_PASSUTILS_H_
#define _IMEX_PASSUTILS_H_

#include <mlir/Dialect/Arith/IR/Arith.h>

namespace imex {

/// @return get ::mlir::FloatAttr with given Value and bitwidth W
template <int W = 64, typename T = double>
::mlir::FloatAttr getFloatAttr(::mlir::OpBuilder &builder, T val) {
  if (W == 64)
    return builder.getF64FloatAttr(val);
  if (W == 32)
    return builder.getF32FloatAttr(val);
  assert(!"only 32- and 64-bit floats supported");
}

/// @return new float ::mlir::Value with given Value and bitwidth W
template <int W = 64, typename T = double>
::mlir::Value createFloat(const ::mlir::Location &loc,
                          ::mlir::OpBuilder &builder, T val) {
  auto attr = getFloatAttr<W>(builder, val);
  return builder.create<::mlir::arith::ConstantOp>(loc, attr);
}

/// @return get ::mlir::IntegerAttr with given Value and bitwidth W
template <int W = 64>
::mlir::IntegerAttr getIntAttr(::mlir::OpBuilder &builder, int64_t val) {
  return builder.getIntegerAttr(builder.getIntegerType(W), val);
}

/// @return new integer ::mlir::Value with given Value and bitwidth W
template <int W = 64>
::mlir::Value createInt(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
                        int64_t val) {
  auto attr = getIntAttr<W>(builder, val);
  return builder.create<::mlir::arith::ConstantOp>(loc, attr);
}

/// @return new index ::mlir::Value with given Value
inline ::mlir::Value createIndex(const ::mlir::Location &loc,
                                 ::mlir::OpBuilder &builder, int64_t val) {
  auto attr = builder.getIndexAttr(val);
  return builder.create<::mlir::arith::ConstantOp>(loc, attr);
}

inline ::mlir::Value createIndexCast(const ::mlir::Location &loc,
                                     ::mlir::OpBuilder &builder,
                                     ::mlir::Value val,
                                     ::mlir::Type intTyp = ::mlir::Type()) {
  if (!intTyp)
    intTyp = builder.getIndexType();
  return val.getType() == intTyp
             ? val
             : builder.create<::mlir::arith::IndexCastOp>(loc, intTyp, val)
                   .getResult();
}

} // namespace imex
#endif // _IMEX_PASSUTILS_H_
