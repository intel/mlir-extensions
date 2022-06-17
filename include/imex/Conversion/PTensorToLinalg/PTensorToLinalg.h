// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Converting PTensor to Linalg and Dist

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
/// Populate the given list with patterns that eliminate Dist ops
void populatePTensorToLinalgConversionPatterns(
    ::mlir::LLVMTypeConverter &converter, ::mlir::RewritePatternSet &patterns);

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertPTensorToLinalgPass();

} // namespace imex

#endif // _PTensorToLinalg_H_INCLUDED_
