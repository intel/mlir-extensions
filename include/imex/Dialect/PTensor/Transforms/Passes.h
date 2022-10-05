//===-- Passes.h - PTensor pass declaration file ----------------*- C++ -*-===//
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
/// PTensor dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _PTENSOR_PASSES_H_INCLUDED_
#define _PTENSOR_PASSES_H_INCLUDED_

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
/// PTensor passes.
//===----------------------------------------------------------------------===//

/// Create a PTensorDist pass
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createPTensorDistPass();

/// Populate the given list with patterns which add Dist-Ops to PTensor ops
void populatePTensorDistPatterns(::mlir::LLVMTypeConverter &converter,
                                 ::mlir::RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include <imex/Dialect/PTensor/Transforms/Passes.h.inc>

} // namespace imex

#endif // _PTENSOR_PASSES_H_INCLUDED_
