// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <mlir/IR/Builders.h>

namespace mlir {
class ModuleOp;
class OpBuilder;
class FunctionType;
class Operation;
} // namespace mlir

namespace llvm {
class StringRef;
}

namespace mlir {
namespace func {
class FuncOp;
}
} // namespace mlir

namespace plier {
mlir::func::FuncOp add_function(mlir::OpBuilder &builder, mlir::ModuleOp module,
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
} // namespace plier
