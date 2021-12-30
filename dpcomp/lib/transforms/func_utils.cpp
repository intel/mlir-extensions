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

#include "plier/transforms/func_utils.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/StringRef.h>

mlir::FuncOp plier::add_function(mlir::OpBuilder &builder,
                                 mlir::ModuleOp module, llvm::StringRef name,
                                 mlir::FunctionType type) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  // Insert before module terminator.
  builder.setInsertionPoint(module.getBody(),
                            std::prev(module.getBody()->end()));
  auto func = builder.create<mlir::FuncOp>(builder.getUnknownLoc(), name, type);
  func.setPrivate();
  return func;
}

plier::AllocaInsertionPoint::AllocaInsertionPoint(mlir::Operation *inst) {
  assert(nullptr != inst);
  auto parent = inst->getParentWithTrait<mlir::OpTrait::IsIsolatedFromAbove>();
  assert(parent->getNumRegions() == 1);
  assert(!parent->getRegions().front().empty());
  auto &block = parent->getRegions().front().front();
  assert(!block.empty());
  insertionPoint = &block.front();
}
