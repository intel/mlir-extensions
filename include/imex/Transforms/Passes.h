//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for IMEX transformation passes
//
//===----------------------------------------------------------------------===//

#ifndef IMEX_TRANSFORMS_PASSES_H_
#define IMEX_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include <functional>
#include <memory>

namespace imex {
//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//
std::unique_ptr<mlir::Pass> createSerializeSPIRVPass();
std::unique_ptr<mlir::Pass>
createInsertGPUAllocsPass(const char *clientAPI = "vulkan");
std::unique_ptr<mlir::Pass> createInsertGPUCopyPass();
std::unique_ptr<mlir::Pass> createSetSPIRVCapabilitiesPass();
std::unique_ptr<mlir::Pass>
createSetSPIRVAbiAttributePass(const char *clientAPI = "vulkan");
std::unique_ptr<mlir::Pass> createAddOuterParallelLoopPass();
std::unique_ptr<mlir::Pass> createLowerMemRefCopyPass();
std::unique_ptr<mlir::Pass> createBF16ToGPUPass();
std::unique_ptr<mlir::Pass> createCastIndexPass();
std::unique_ptr<mlir::Pass> createRemoveTemporariesPass();
std::unique_ptr<mlir::Pass> createRemoveSingleElemVectorPass();
std::unique_ptr<mlir::Pass>
createOptimizeTransposePass(const std::string &device = "pvc");
std::unique_ptr<mlir::Pass> createHoistTransposePass();
std::unique_ptr<mlir::Pass> createVnniTransformationPass();
std::unique_ptr<mlir::Pass> createEmulateNonNativeBF16Pass();
std::unique_ptr<mlir::Pass> createTileLoopsPass();
std::unique_ptr<mlir::Pass> createMaterializeMatrixOpPass();

#define GEN_PASS_DECL
#include "imex/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

#endif // IMEX_TRANSFORMS_PASSES_H_
