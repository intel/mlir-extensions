//===- PassDetail.h - Transforms Pass class details -------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMS_PASSDETAIL_H_
#define TRANSFORMS_PASSDETAIL_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace gpu {
class GPUDialect;
}
namespace spirv {
class SPIRVDialect;
}

namespace scf {
class SCFDialect;
}

namespace linalg {
class LinalgDialect;
}

namespace vector {
class VectorDialect;
}
} // end namespace mlir

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"

namespace imex {
#define GEN_PASS_CLASSES
#include "imex/Transforms/Passes.h.inc"

} // namespace imex

#endif // TRANSFORMS_PASSDETAIL_H
