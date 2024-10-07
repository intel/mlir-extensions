//===- DropRegions.h - DropRegions conversion  -------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the DropRegions conversion, converting the Region
/// dialect to the GPU dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _DropRegions_H_INCLUDED_
#define _DropRegions_H_INCLUDED_

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
#define GEN_PASS_DECL_DROPREGIONS
#include "imex/Conversion/Passes.h.inc"

/// Populate the given list with patterns rewrite Region Ops
void populateDropRegionsConversionPatterns(::mlir::LLVMTypeConverter &converter,
                                           ::mlir::RewritePatternSet &patterns);

/// Create a pass to convert the Region dialect to the GPU dialect.
std::unique_ptr<::mlir::Pass> createDropRegionsPass();

} // namespace imex

#endif // _DropRegions_H_INCLUDED_
