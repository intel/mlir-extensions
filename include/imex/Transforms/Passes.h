//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for IMEX transformation passes
//
//===----------------------------------------------------------------------===//

#ifndef IMEX_TRANSFORMS_PASSES_H_
#define IMEX_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace imex {
//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//
std::unique_ptr<mlir::Pass> createSerializeSPIRVPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

#endif // IMEX_TRANSFORMS_PASSES_H_
