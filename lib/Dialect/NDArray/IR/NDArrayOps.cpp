//===- NDArrayOps.cpp - NDArray dialect  -----------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the NDArray dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Utils/PassUtils.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/DialectImplementation.h>

namespace imex {
namespace ndarray {

void NDArrayDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/NDArray/IR/NDArrayOpsTypes.cpp.inc>
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include <imex/Dialect/NDArray/IR/NDArrayOpsAttrs.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <imex/Dialect/NDArray/IR/NDArrayOps.cpp.inc>
      >();
}

} // namespace ndarray
} // namespace imex

bool imex::ndarray::isUnitShape(const llvm::ArrayRef<int64_t> shp) {
  for (auto d : shp) {
    if (d != 1)
      return false;
  }
  return true;
}

bool imex::ndarray::hasZeroSize(const llvm::ArrayRef<int64_t> shp) {
  for (auto d : shp) {
    if (d == 0)
      return true;
  }
  return false;
}

#include <imex/Dialect/NDArray/IR/NDArrayOpsDialect.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/NDArray/IR/NDArrayOpsTypes.cpp.inc>
#define GET_ATTRDEF_CLASSES
#include <imex/Dialect/NDArray/IR/NDArrayOpsAttrs.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/NDArray/IR/NDArrayOps.cpp.inc>
