//===-- Passes.h - Dist pass declaration file -------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header file defines prototypes that expose pass constructors for the
/// Dist dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _Dist_PASSES_H_INCLUDED_
#define _Dist_PASSES_H_INCLUDED_

#include <mlir/Pass/Pass.h>

namespace mlir {
class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T> class OperationPass;
class RewritePatternSet;
} // namespace mlir

namespace imex {

//===----------------------------------------------------------------------===//
/// Dist passes.
//===----------------------------------------------------------------------===//

/// Create a DistCoalesce pass
std::unique_ptr<::mlir::Pass> createDistCoalescePass();
/// Create DistInferEWBinopPass
std::unique_ptr<::mlir::Pass> createDistInferEWCoresPass();

#define GEN_PASS_DECL
#include <imex/Dialect/Dist/Transforms/Passes.h.inc>

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include <imex/Dialect/Dist/Transforms/Passes.h.inc>

} // namespace imex

#endif // _Dist_PASSES_H_INCLUDED_
