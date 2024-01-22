//===- DistRuntimeOps.h - DistRuntime dialect  -------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the DistRuntime dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _DistRuntime_OPS_H_INCLUDED_
#define _DistRuntime_OPS_H_INCLUDED_

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace mlir {
#include <imex/Dialect/DistRuntime/IR/DistRuntimeOpsIFaces.h.inc>
}
namespace imex {
namespace distruntime {
using ::mlir::AsyncOpInterface;
} // namespace distruntime
} // namespace imex

#include <imex/Dialect/DistRuntime/IR/DistRuntimeOpsDialect.h.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/DistRuntime/IR/DistRuntimeOpsTypes.h.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/DistRuntime/IR/DistRuntimeOps.h.inc>

#endif // _DistRuntime_OPS_H_INCLUDED_
