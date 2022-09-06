//===- IntelGpuOps.cpp - IntelGpu dialect -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the IntelGpu dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/IntelGpu/IR/IntelGpuOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/InliningUtils.h>

namespace imex {
namespace intel_gpu {

void IntelGpuDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/IntelGpu/IR/IntelGpuOpsTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <imex/Dialect/IntelGpu/IR/IntelGpuOps.cpp.inc>
      >();
  addTypes<OpaqueType>();
}

OpaqueType OpaqueType::get(mlir::MLIRContext *context) {
  assert(context);
  return Base::get(context);
}

} // namespace intel_gpu
} // namespace imex

#include <imex/Dialect/IntelGpu/IR/IntelGpuOpsDialect.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/IntelGpu/IR/IntelGpuOpsTypes.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/IntelGpu/IR/IntelGpuOps.cpp.inc>
