// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
