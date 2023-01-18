// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;
} // namespace mlir

namespace imex {
void populatePromoteToParallelPatterns(mlir::RewritePatternSet &patterns);

/// This pass tries to promote `scf.for` ops to `scf.parallel`.
std::unique_ptr<mlir::Pass> createPromoteToParallelPass();
} // namespace imex
