//===-- PassDetail.h - PTensor pass details ----------------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header file defines prototypes for PTensor dialect passes.
///
//===----------------------------------------------------------------------===//

#ifndef _PTENSOR_PASSDETAIL_H_INCLUDED_
#define _PTENSOR_PASSDETAIL_H_INCLUDED_

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>

namespace mlir {

namespace affine {
class AffineDialect;
}

namespace arith {
class ArithDialect;
} // namespace arith

namespace memref {
class MemRefDialect;
} // namespace memref

namespace tensor {
class TensorDialect;
} // namespace tensor

namespace shape {
class ShapeDialect;
} // namespace shape

namespace linalg {
class LinalgDialect;
} // namespace linalg

} // namespace mlir

namespace imex {

namespace dist {
class DistDialect;
} // namespace dist

namespace ptensor {
class PTensorDialect;
} // namespace ptensor

#define GEN_PASS_CLASSES
#include <imex/Dialect/PTensor/Transforms/Passes.h.inc>

} // namespace imex

#endif // _PTENSOR_PASSDETAIL_H_INCLUDED_
