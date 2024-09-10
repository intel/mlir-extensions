//===-- Passes.h - NDArray pass declaration file ----------------*- C++ -*-===//
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
/// NDArray dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _NDARRAY_PASSES_H_INCLUDED_
#define _NDARRAY_PASSES_H_INCLUDED_

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
/// NDArray passes.
//===----------------------------------------------------------------------===//

/// Create a NDArrayDist pass
std::unique_ptr<::mlir::Pass> createNDArrayDistPass();

/// Populate the given list with patterns which add Dist-Ops to NDArray ops
void populateNDArrayDistPatterns(::mlir::LLVMTypeConverter &converter,
                                 ::mlir::RewritePatternSet &patterns);

/// Create a AddGPURegions pass
std::unique_ptr<::mlir::Pass> createAddGPURegionsPass();

#define GEN_PASS_DECL
#include <imex/Dialect/NDArray/Transforms/Passes.h.inc>

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include <imex/Dialect/NDArray/Transforms/Passes.h.inc>

} // namespace imex

#endif // _NDARRAY_PASSES_H_INCLUDED_
