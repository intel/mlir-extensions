// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class MLIRContext;
class Pass;
class RewritePatternSet;
} // namespace mlir

namespace imex {
void populateNtensorToLinalgPatterns(mlir::MLIRContext &context,
                                     mlir::RewritePatternSet &patterns);

/// Creates a pass for ntensor alias analysis, required by ntensor-to-linalg.
std::unique_ptr<mlir::Pass> createNtensorAliasAnalysisPass();

/// Creates a pass to convert ntensor array ops to linalg.
std::unique_ptr<mlir::Pass> createNtensorToLinalgPass();
} // namespace imex
