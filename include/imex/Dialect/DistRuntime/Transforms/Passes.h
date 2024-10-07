//===-- Passes.h - DistRuntime pass declaration file ------*- tablegen -*-===//
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
/// DistRuntime dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _DistRuntime_PASSES_H_INCLUDED_
#define _DistRuntime_PASSES_H_INCLUDED_

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
/// DistRuntime passes.
//===----------------------------------------------------------------------===//

std::unique_ptr<::mlir::Pass> createDistRuntimeToIDTRPass();
std::unique_ptr<::mlir::Pass> createOverlapCommAndComputePass();
std::unique_ptr<::mlir::Pass> createAddCommCacheKeysPass();

#define GEN_PASS_DECL_OVERLAPCOMMANDCOMPUTE
#define GEN_PASS_DECL_ADDCOMMCACHEKEYS
#define GEN_PASS_DECL_LOWERDISTRUNTIMETOIDTR
#include <imex/Dialect/DistRuntime/Transforms/Passes.h.inc>

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include <imex/Dialect/DistRuntime/Transforms/Passes.h.inc>

} // namespace imex

#endif // _DistRuntime_PASSES_H_INCLUDED_
