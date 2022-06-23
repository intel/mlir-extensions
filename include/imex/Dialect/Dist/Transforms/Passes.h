//===-- Passes.h - Dist pass declaration file --------------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
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

/// Create a DistElim pass
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>> createDistElimPass();

/// Populate the given list with patterns that eliminate Dist ops
void populateDistElimConversionPatterns(::mlir::LLVMTypeConverter &converter,
                                        ::mlir::RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include <imex/Dialect/Dist/Transforms/Passes.h.inc>

} // namespace imex

#endif // _Dist_PASSES_H_INCLUDED_
