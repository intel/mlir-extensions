// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>

namespace ptensor {

void PTensorDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/PTensor/IR/PTensorOpsTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <imex/Dialect/PTensor/IR/PTensorOps.cpp.inc>
      >();
}

} // namespace ptensor

#include <imex/Dialect/PTensor/IR/PTensorOpsDialect.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/PTensor/IR/PTensorOpsTypes.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/PTensor/IR/PTensorOps.cpp.inc>
