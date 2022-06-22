//===- DistElim.h - PTensorToLinalg conversion  ---------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines elimination of the Dist dialects, leading to local-only
/// operation
///
//===----------------------------------------------------------------------===//

#ifndef _DistElim_H_INCLUDED_
#define _DistElim_H_INCLUDED_

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
/// Populate the given list with patterns that eliminate Dist ops
void populateDistElimConversionPatterns(::mlir::LLVMTypeConverter &converter,
                                        ::mlir::RewritePatternSet &patterns);

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>> createDistElimPass();

} // namespace imex

#endif // _DistElim_H_INCLUDED_
