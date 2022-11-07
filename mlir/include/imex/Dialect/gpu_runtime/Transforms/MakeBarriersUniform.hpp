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

namespace gpu_runtime {
void populateMakeBarriersUniformPatterns(mlir::RewritePatternSet &patterns);

/// gpu barriers ops require uniform control flow, this pass tries to rearrange
/// control flow in a way to satisfy this requirement.
std::unique_ptr<mlir::Pass> createMakeBarriersUniformPass();
} // namespace gpu_runtime
