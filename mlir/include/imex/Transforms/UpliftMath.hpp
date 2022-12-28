// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class MLIRContext;
class RewritePatternSet;
class Pass;
} // namespace mlir

namespace imex {
void populateUpliftMathPatterns(mlir::RewritePatternSet &patterns);
void populateUpliftFMAPatterns(mlir::RewritePatternSet &patterns);

/// This pass tries to uplift libm-style func call to math dialect ops.
std::unique_ptr<mlir::Pass> createUpliftMathPass();

/// This pass tries to uplift sequence of arith ops to math.fma op.
std::unique_ptr<mlir::Pass> createUpliftFMAPass();
} // namespace imex
