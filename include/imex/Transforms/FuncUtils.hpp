//===- FuncUtils.hpp -  --------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
///

#pragma once

#include <mlir/IR/Builders.h>

namespace mlir {
class ModuleOp;
class OpBuilder;
class FunctionType;
class Operation;

namespace func {
class FuncOp;
}

} // namespace mlir

namespace llvm {
class StringRef;
}

namespace imex {
mlir::func::FuncOp addFunction(mlir::OpBuilder &builder, mlir::ModuleOp module,
                               llvm::StringRef name, mlir::FunctionType type);

struct AllocaInsertionPoint {
  AllocaInsertionPoint(mlir::Operation *inst);

  template <typename F> auto insert(mlir::OpBuilder &builder, F &&func) {
    assert(nullptr != insertionPoint);
    if (builder.getBlock() == insertionPoint->getBlock()) {
      return func();
    }
    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(insertionPoint);
    return func();
  }

private:
  mlir::Operation *insertionPoint = nullptr;
};
} // namespace imex
