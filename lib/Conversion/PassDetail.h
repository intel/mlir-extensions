//===- IMEXPassDetail.h - Conversion Pass class details ---------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
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
#define _IMEX_CONVERSION_PASSDETAIL_H_

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>

namespace mlir {
namespace affine {
class AffineDialect;
}

namespace arith {
class ArithDialect;
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

namespace bufferization {
class BufferizationDialect;
} // namespace bufferization

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

namespace tosa {
class TosaDialect;
} // namespace tosa

namespace gpu {
class GPUDialect;
} // namespace gpu

namespace vector {
class VectorDialect;
}
} // namespace mlir

namespace imex {
namespace ndarray {
class NDArrayDialect;
} // namespace ndarray

namespace dist {
class DistDialect;
} // namespace dist

namespace distruntime {
class DistRuntimeDialect;
} // namespace distruntime

namespace region {
class RegionDialect;
} // namespace region

namespace gpux {
class GPUXDialect;
} // namespace gpux

namespace xegpu {
class XeGPUDialect;
}

namespace xetile {
class XeTileDialect;
}

#define GEN_PASS_CLASSES
#include <imex/Conversion/Passes.h.inc>

} // namespace imex

#endif // _IMEX_CONVERSION_PASSDETAIL_H_
