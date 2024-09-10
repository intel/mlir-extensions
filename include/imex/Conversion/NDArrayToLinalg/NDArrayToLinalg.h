//===- NDArrayToLinalg.h - NDArrayToLinalg conversion  ---------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the NDArrayToLinalg conversion, converting the NDArray
/// dialect to the Linalg and Dist dialects.
///
//===----------------------------------------------------------------------===//

#ifndef _NDArrayToLinalg_H_INCLUDED_
#define _NDArrayToLinalg_H_INCLUDED_

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T> class OperationPass;
class RewritePatternSet;
} // namespace mlir

namespace imex {
#define GEN_PASS_DECL_CONVERTNDARRAYTOLINALG
#include "imex/Conversion/Passes.h.inc"

/// Populate the given list with patterns which convert NDArray ops to Linalg
/// and Dist
void populateNDArrayToLinalgConversionPatterns(
    ::mlir::LLVMTypeConverter &converter, ::mlir::RewritePatternSet &patterns);

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::Pass> createConvertNDArrayToLinalgPass();

} // namespace imex

#endif // _NDArrayToLinalg_H_INCLUDED_
