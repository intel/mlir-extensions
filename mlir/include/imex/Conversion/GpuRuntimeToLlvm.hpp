// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class ConversionTarget;
class LLVMTypeConverter;
class Pass;
class RewritePatternSet;
} // namespace mlir

namespace gpu_runtime {

std::unique_ptr<mlir::Pass> createEnumerateEventsPass();

/// Populates the given list with patterns that convert from gpu_runtime to
/// LLVM.
void populateGpuToLLVMPatternsAndLegality(mlir::LLVMTypeConverter &converter,
                                          mlir::RewritePatternSet &patterns,
                                          mlir::ConversionTarget &target);

std::unique_ptr<mlir::Pass> createGPUToLLVMPass();

} // namespace gpu_runtime
