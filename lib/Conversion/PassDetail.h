//===- IMEXPassDetail.h - Conversion Pass class details ---------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares IMEX Passes.
//
//===----------------------------------------------------------------------===//

#ifndef _IMEX_CONVERSION_PASSDETAIL_H_
#define _IMEXCONVERSION_PASSDETAIL_H_

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>

namespace mlir {
class AffineDialect;

namespace arith {
class ArithmeticDialect;
} // namespace arith

namespace func {
class FuncDialect;
} // namespace func

namespace linalg {
class LinalgDialect;
} // namespace linalg

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

namespace math {
class MathDialect;
} // namespace math

namespace memref {
class MemRefDialect;
} // namespace memref

namespace scf {
class SCFDialect;
} // namespace scf

namespace shape {
class ShapeDialect;
} // namespace shape

namespace spirv {
class SPIRVDialect;
} // namespace spirv

namespace tensor {
class TensorDialect;
} // namespace tensor
} // namespace mlir

namespace imex {
namespace ptensor {
class PTensorDialect;
} // namespace ptensor

namespace dist {
class DistDialect;
} // namespace dist

#define GEN_PASS_CLASSES
#include <imex/Conversion/Passes.h.inc>

} // namespace imex

#endif // _IMEXCONVERSION_PASSDETAIL_H_
