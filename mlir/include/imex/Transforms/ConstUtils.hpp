// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Value.h>

namespace mlir {
class Operation;
class IntegerAttr;
} // namespace mlir

namespace imex {
mlir::Attribute getConstVal(mlir::Operation *op);
mlir::Attribute getConstVal(mlir::Value op);

template <typename T> T getConstVal(mlir::Operation *op) {
  return getConstVal(op).dyn_cast_or_null<T>();
}

template <typename T> T getConstVal(mlir::Value op) {
  return getConstVal(op).dyn_cast_or_null<T>();
}

mlir::Attribute getConstAttr(mlir::Type type, double val);

int64_t getIntAttrValue(mlir::IntegerAttr attr);
} // namespace imex
