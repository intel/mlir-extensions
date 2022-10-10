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

} // end namespace mlir

namespace imex {
#define GEN_PASS_CLASSES
#include "imex/Transforms/Passes.h.inc"

} // namespace imex

#endif // TRANSFORMS_PASSDETAIL_H
