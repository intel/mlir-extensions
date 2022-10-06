//===-- PasUtils.h - PTensor utils ------------------------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header file defines utils for PTensor passes.
///
//===----------------------------------------------------------------------===//

#ifndef _PTENSOR_UTILS_H_INCLUDED_
#define _PTENSOR_UTILS_H_INCLUDED_

#include <imex/internal/PassUtils.h>

namespace imex {

/// create operations computing the count of elements a
/// arange(start, stop, step) would have.
/// @return number of elements an arange(start, stop, step) would have
/// (::mlir::Value)
inline ::mlir::Value createCountARange(::mlir::OpBuilder &builder,
                                       ::mlir::Location loc,
                                       ::mlir::Value start, ::mlir::Value stop,
                                       ::mlir::Value step) {
  // Create constants 0, 1, -1 for later
  auto zero = createInt(loc, builder, 0);
  auto one = createInt(loc, builder, 1);
  auto mone = createInt(loc, builder, -1);

  // Compute number of elements as
  //   (stop - start + step + (step < 0 ? 1 : -1)) / step
  auto cond = builder.create<mlir::arith::CmpIOp>(
      loc, ::mlir::arith::CmpIPredicate::ult, step, zero);
  auto increment = builder.create<mlir::arith::SelectOp>(loc, cond, one, mone);
  auto tmp1 = builder.create<mlir::arith::AddIOp>(loc, stop, step);
  auto tmp2 = builder.create<mlir::arith::AddIOp>(loc, tmp1, increment);
  auto tmp3 = builder.create<mlir::arith::SubIOp>(loc, tmp2, start);
  auto count =
      builder.create<mlir::arith::DivUIOp>(loc, tmp3, step).getResult();
  return builder
      .create<::mlir::arith::IndexCastOp>(loc, builder.getIndexType(), count)
      .getResult();
}

} // namespace imex

#endif // _PTENSOR_UTILS_H_INCLUDED_
