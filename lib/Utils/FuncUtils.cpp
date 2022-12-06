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
/// This file defines a utility function to find insertion point for given
/// Op (for instance LaunchOp) having specific trait
///

#include "imex/Utils/FuncUtils.hpp"

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

/// Gets isolated region/block with IsolatedFromAbove trait for a given
/// operation and sets that location as insertion point into the module.
/// Isolated here refers to the regions of operations are known to be isolated
/// from above and will not capture, or reference, SSA values defined above the
/// region scope.
imex::AllocaInsertionPoint::AllocaInsertionPoint(mlir::Operation *inst) {
  assert(nullptr != inst);
  auto parent = inst->getParentWithTrait<mlir::OpTrait::IsIsolatedFromAbove>();
  assert(parent->getNumRegions() == 1);
  assert(!parent->getRegions().front().empty());
  auto &block = parent->getRegions().front().front();
  assert(!block.empty());
  insertionPoint = &block.front();
}
