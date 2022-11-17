//===- PTensor.cpp - PTensor dialect  --------------------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the PTensor dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/internal/PassUtils.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/DialectImplementation.h>

namespace imex {
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
} // namespace imex

#include <imex/Dialect/PTensor/IR/PTensorOpsDialect.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/PTensor/IR/PTensorOpsTypes.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/PTensor/IR/PTensorOps.cpp.inc>

namespace imex {
namespace ptensor {

::mlir::MemRefType PTensorType::getMemRefType() {
  return ::imex::getMemRefType(getContext(), getRank(), getElementType());
}

::mlir::RankedTensorType PTensorType::getTensorType() {
  return ::imex::getTensorType(getContext(), getRank(), getElementType());
}

} // namespace ptensor
} // namespace imex
