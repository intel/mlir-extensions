//===- PTensorToLinalg.h - PTensorToLinalg conversion  ---------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the PTensorToLinalg conversion, converting the PTensor
/// dialect to the Linalg and Dist dialects.
///
//===----------------------------------------------------------------------===//

#ifndef _PTensorToLinalg_H_INCLUDED_
#define _PTensorToLinalg_H_INCLUDED_

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T> class OperationPass;
class RewritePatternSet;
} // namespace mlir

namespace imex {
/// Populate the given list with patterns which convert PTensor ops to Linalg
/// and Dist
void populatePTensorToLinalgConversionPatterns(
    ::mlir::LLVMTypeConverter &converter, ::mlir::RewritePatternSet &patterns);

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertPTensorToLinalgPass();

} // namespace imex

#endif // _PTensorToLinalg_H_INCLUDED_
