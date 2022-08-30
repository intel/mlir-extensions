//===-- Passes.h - IntelGpu pass declaration file --------*- tablegen -*-===//
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
/// IntelGpu dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _IntelGpu_PASSES_H_INCLUDED_
#define _IntelGpu_PASSES_H_INCLUDED_

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
/// IntelGpu passes.
//===----------------------------------------------------------------------===//

/// FIXME
std::unique_ptr<::mlir::Pass> createFIXMEPass();

/// Populate the given list with patterns that eliminate Dist ops
void populateFIXMEPatterns(::mlir::LLVMTypeConverter &converter,
                           ::mlir::RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include <imex/Dialect/IntelGpu/Transforms/Passes.h.inc>

} // namespace imex

#endif // _IntelGpu_PASSES_H_INCLUDED_
