//===- DistToStandard.h - DistToStandard conversion  ------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the DistToStandard conversion, converting the Dist
/// dialect to standard dialects.
///
//===----------------------------------------------------------------------===//

#ifndef _DistToStandard_H_INCLUDED_
#define _DistToStandard_H_INCLUDED_

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
/// Populate the given list with patterns rewrite Dist Ops
void populateDistToStandardConversionPatterns(
    ::mlir::LLVMTypeConverter &converter, ::mlir::RewritePatternSet &patterns);

/// Create a pass to convert the Dist dialect to the Standard dialect.
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertDistToStandardPass();

extern void createDistRTFuncDefs(::mlir::Location &loc,
                                 ::mlir::OpBuilder &builder,
                                 ::mlir::ModuleOp &module);

} // namespace imex

#endif // _DistToStandard_H_INCLUDED_
