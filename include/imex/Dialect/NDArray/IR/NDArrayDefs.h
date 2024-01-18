//===- NDArrayDefs.h - NDArray dialect  -------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the NDArray dialect's enums and other defs.
///
//===----------------------------------------------------------------------===//

#ifndef _NDArray_DEFS_H_INCLUDED_
#define _NDArray_DEFS_H_INCLUDED_

namespace imex {
namespace ndarray {

enum DType : int8_t {
  F64,
  F32,
  I64,
  U64,
  I32,
  U32,
  I16,
  U16,
  I8,
  U8,
  I1,
  DTYPE_LAST
};

/// The set of supported elementwise binary operations
enum EWBinOpId : int {
  ADD,
  ATAN2,
  BITWISE_AND,
  BITWISE_LEFT_SHIFT,
  BITWISE_OR,
  BITWISE_RIGHT_SHIFT,
  BITWISE_XOR,
  EQUAL,
  FLOOR_DIVIDE,
  GREATER,
  GREATER_EQUAL,
  LESS,
  LESS_EQUAL,
  LOGADDEXP,
  LOGICAL_AND,
  LOGICAL_OR,
  LOGICAL_XOR,
  MATMUL,
  MAXIMUM,
  MINIMUM,
  MODULO,
  MULTIPLY,
  NOT_EQUAL,
  POWER,
  SUBTRACT,
  TRUE_DIVIDE,
  EWBINOPID_LAST
};

/// The set of supported elementwise unary operations
enum EWUnyOpId : int {
  ABS,
  ACOS,
  ACOSH,
  ASIN,
  ASINH,
  ATAN,
  ATANH,
  CEIL,
  COS,
  COSH,
  ERF,
  EXP,
  EXPM1,
  FLOOR,
  ISFINITE,
  ISINF,
  ISNAN,
  LOG,
  LOG1P,
  LOG2,
  LOG10,
  ROUND,
  SIGN,
  SIN,
  SINH,
  SQUARE,
  SQRT,
  TAN,
  TANH,
  TRUNC,
  NEGATIVE,
  POSITIVE,
  LOGICAL_NOT,
  EWUNYOPID_LAST
};

/// The set of supported reduction operations
enum ReduceOpId : int { MAX, MEAN, MIN, PROD, SUM, STD, VAR, REDUCEOPID_LAST };

} // namespace ndarray
} // namespace imex

#endif // _NDArray_DEFS_H_INCLUDED_
