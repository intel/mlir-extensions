//===- GPUXOps.cpp - GPUX dialect -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the GPUX dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/GPUX/IR/GPUXOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/InliningUtils.h>

namespace imex {
namespace gpux {

void GPUXDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/GPUX/IR/GPUXOpsTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <imex/Dialect/GPUX/IR/GPUXOps.cpp.inc>
      >();
}

} // namespace gpux
} // namespace imex

#include <imex/Dialect/GPUX/IR/GPUXOpsDialect.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/GPUX/IR/GPUXOpsTypes.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/GPUX/IR/GPUXOps.cpp.inc>
