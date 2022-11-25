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

#include <imex/Utils/ArithUtils.h>

namespace imex {

/// create operations computing the count of elements a
/// arange(start, stop, step) would have.
/// @return number of elements an arange(start, stop, step) would have
/// (::mlir::Value)
inline EasyInt createCountARange(::mlir::OpBuilder &builder,
                                 ::mlir::Location loc, const EasyInt &start,
                                 const EasyInt &stop, const EasyInt &step) {
  // Create constants 0, 1, -1 for later
  auto zero = ::imex::EasyInt::create(loc, builder, 0, true);
  auto one = ::imex::EasyInt::create(loc, builder, 1, true);
  auto mone = ::imex::EasyInt::create(loc, builder, -1, true);

  // Compute number of elements as
  //   (stop - start + step + (step < 0 ? 1 : -1)) / step
  auto increment = step.ult(zero).select(one, mone);
  return (stop - start + step + increment) / step;
}

} // namespace imex

#endif // _PTENSOR_UTILS_H_INCLUDED_
