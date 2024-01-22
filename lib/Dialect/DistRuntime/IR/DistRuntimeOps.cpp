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

extern "C" {
int _idtr_nprocs(void *) __attribute__((weak));
int _idtr_prank(void *) __attribute__((weak));
}

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

static auto DNDA_NPROCS = getenv("DNDA_NPROCS");
static auto DNDA_PRANK = getenv("DNDA_PRANK");

::mlir::OpFoldResult TeamSizeOp::fold(FoldAdaptor adaptor) {
  // call runtime at compile time if available and team is constant
  if (DNDA_NPROCS) {
    auto np = std::stoi(DNDA_NPROCS);
    ::mlir::Builder builder(getContext());
    return builder.getIndexAttr(np);
  }
  if (_idtr_nprocs != NULL) {
    ::mlir::Builder builder(getContext());
    auto team = adaptor.getTeam().cast<::mlir::IntegerAttr>().getInt();
    auto np = _idtr_nprocs(reinterpret_cast<void *>(team));
    return builder.getIndexAttr(np);
  }
  return nullptr;
}

::mlir::OpFoldResult TeamMemberOp::fold(FoldAdaptor adaptor) {
  // call runtime at compile time if available and team is constant
  if (DNDA_PRANK) {
    auto np = std::stoi(DNDA_PRANK);
    ::mlir::Builder builder(getContext());
    return builder.getIndexAttr(np);
  }
  if (_idtr_prank != NULL) {
    ::mlir::Builder builder(getContext());
    auto team = adaptor.getTeam().cast<::mlir::IntegerAttr>().getInt();
    auto np = _idtr_prank(reinterpret_cast<void *>(team));
    return builder.getIndexAttr(np);
  }
  return nullptr;
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
/// Ported from mlir::tensor dialect
mlir::Operation *imex::distruntime::DistRuntimeDialect::materializeConstant(
    mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type type,
    mlir::Location loc) {
  if (auto op = mlir::arith::ConstantOp::materialize(builder, value, type, loc))
    return op;
  return nullptr;
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
