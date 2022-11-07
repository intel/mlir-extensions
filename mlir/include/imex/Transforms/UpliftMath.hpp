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
void populateUpliftmathPatterns(mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createUpliftMathPass();
} // namespace imex
