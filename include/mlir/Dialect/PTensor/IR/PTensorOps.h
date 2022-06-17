// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef _PTensor_OPS_H_INCLUDED_
#define _PTensor_OPS_H_INCLUDED_

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>

namespace ptensor {

// The set of supported operations
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

enum ReduceOpId : int { MAX, MEAN, MIN, PROD, SUM, STD, VAR, REDUCEOPID_LAST };

} // namespace ptensor

#include <mlir/Dialect/PTensor/IR/PTensorOpsDialect.h.inc>
#define GET_TYPEDEF_CLASSES
#include <mlir/Dialect/PTensor/IR/PTensorOpsTypes.h.inc>
#define GET_OP_CLASSES
#include <mlir/Dialect/PTensor/IR/PTensorOps.h.inc>

#endif // _PTensor_OPS_H_INCLUDED_
