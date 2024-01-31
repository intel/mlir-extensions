//===-- PassDetail.h - DistRuntime pass details --------*- tablegen -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header file defines prototypes for DistRuntime dialect passes.
///
//===----------------------------------------------------------------------===//

#ifndef _DistRuntime_PASSDETAIL_H_INCLUDED_
#define _DistRuntime_PASSDETAIL_H_INCLUDED_

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>

namespace imex {

namespace ndarray {
class NDArrayDialect;
} // namespace ndarray

namespace distruntime {
class DistRuntimeDialect;
} // namespace distruntime

} // namespace imex

namespace mlir {
namespace func {
class FuncDialect;
} // namespace func

namespace memref {
class MemRefDialect;
} // namespace memref

namespace bufferization {
class BufferizationDialect;
} // namespace bufferization

} // namespace mlir

namespace imex {

#define GEN_PASS_CLASSES
#include <imex/Dialect/DistRuntime/Transforms/Passes.h.inc>

} // namespace imex

#endif // _DistRuntime_PASSDETAIL_H_INCLUDED_
