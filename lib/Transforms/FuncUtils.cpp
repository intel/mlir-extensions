//===- FuncUtils.cpp -  --------------*- C++ -*-===//
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

#include "imex/Transforms/FuncUtils.hpp"

#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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
