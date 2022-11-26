// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/FuncUtils.hpp"

#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

mlir::func::FuncOp imex::addFunction(mlir::OpBuilder &builder,
                                     mlir::ModuleOp module,
                                     llvm::StringRef name,
                                     mlir::FunctionType type) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  // Insert before module terminator.
  builder.setInsertionPoint(module.getBody(),
                            std::prev(module.getBody()->end()));
  auto func =
      builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), name, type);
  func.setPrivate();
  return func;
}

imex::AllocaInsertionPoint::AllocaInsertionPoint(mlir::Operation *inst) {
  assert(nullptr != inst);
  auto parent = inst->getParentWithTrait<mlir::OpTrait::IsIsolatedFromAbove>();
  assert(parent->getNumRegions() == 1);
  assert(!parent->getRegions().front().empty());
  auto &block = parent->getRegions().front().front();
  assert(!block.empty());
  insertionPoint = &block.front();
}

std::string imex::getUniqueLLVMGlobalName(mlir::ModuleOp mod,
                                          mlir::StringRef srcName) {
  auto globals = mod.getOps<mlir::LLVM::GlobalOp>();
  for (int i = 0;; ++i) {
    auto name =
        (i == 0 ? std::string(srcName) : (srcName + llvm::Twine(i)).str());
    auto isSameName = [&](mlir::LLVM::GlobalOp global) {
      return global.getName() == name;
    };
    if (llvm::find_if(globals, isSameName) == globals.end())
      return name;
  }
}
