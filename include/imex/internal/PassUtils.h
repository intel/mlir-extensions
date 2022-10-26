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

/// @return get ::mlir::IntegerAttr with given Value and bitwidth W
template <int W = 64, typename T = int64_t>
::mlir::IntegerAttr getIntAttr(::mlir::OpBuilder &builder, T val) {
  return builder.getIntegerAttr(builder.getIntegerType(W), val);
}

/// @return new integer ::mlir::Value with given Value and bitwidth W
template <int W = 64, typename T = int64_t>
::mlir::Value createInt(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
                        T val) {
  auto attr = getIntAttr<W>(builder, val);
  return builder.create<::mlir::arith::ConstantOp>(loc, attr);
}

/// @return new index ::mlir::Value with given Value
inline ::mlir::Value createIndex(const ::mlir::Location &loc,
                                 ::mlir::OpBuilder &builder, uint64_t val) {
  auto attr = builder.getIndexAttr(val);
  return builder.create<::mlir::arith::ConstantOp>(loc, attr);
}

} // namespace imex
#endif // _IMEX_PASSUTILS_H_
