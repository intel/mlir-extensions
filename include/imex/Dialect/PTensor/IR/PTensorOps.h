//===- PTensorOps.h - PTensor dialect  -------------------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the PTensor dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#ifndef _PTensor_OPS_H_INCLUDED_
#define _PTensor_OPS_H_INCLUDED_

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>

namespace imex {
namespace ptensor {

/// The set of supported elementwise binary operations
enum EWBinOpId : int {
  ADD,
  AND,
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
  LSHIFT,
  MATMUL,
  MAXIMUM,
  MINIMUM,
  MODULO,
  MULTIPLY,
  NOT_EQUAL,
  OR,
  POWER,
  SUBTRACT,
  TRUE_DIVIDE,
  XOR,
  EWBINOPID_LAST
};

/// The set of supported reduction operations
enum ReduceOpId : int { MAX, MEAN, MIN, PROD, SUM, STD, VAR, REDUCEOPID_LAST };

} // namespace ptensor
} // namespace imex

#include <imex/Dialect/PTensor/IR/PTensorOpsDialect.h.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/PTensor/IR/PTensorOpsTypes.h.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/PTensor/IR/PTensorOps.h.inc>

#endif // _PTensor_OPS_H_INCLUDED_
