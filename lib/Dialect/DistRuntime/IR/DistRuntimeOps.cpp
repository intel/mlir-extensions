//===- DistRuntimeOps.cpp - DistRuntime dialect -------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the DistRuntime dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/DistRuntime/IR/DistRuntimeOps.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

namespace imex {
namespace distruntime {

void DistRuntimeDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/DistRuntime/IR/DistRuntimeOpsTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <imex/Dialect/DistRuntime/IR/DistRuntimeOps.cpp.inc>
      >();
}

} // namespace distruntime
} // namespace imex

namespace mlir {
#include <imex/Dialect/DistRuntime/IR/DistRuntimeOpsIFaces.cpp.inc>
}
#include <imex/Dialect/DistRuntime/IR/DistRuntimeOpsDialect.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/DistRuntime/IR/DistRuntimeOpsTypes.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/DistRuntime/IR/DistRuntimeOps.cpp.inc>
