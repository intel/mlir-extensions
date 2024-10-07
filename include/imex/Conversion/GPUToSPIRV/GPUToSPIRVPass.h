//===- GPUToSPIRVPass.h - GPUToSPIRV conversion  ---------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the GPUToSPIRV conversion,
///
//===----------------------------------------------------------------------===//

#ifndef IMEX_GPUTOSPIRV_PASS_H_
#define IMEX_GPUTOSPIRV_PASS_H_
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Transforms/DialectConversion.h>

#include <memory>

namespace mlir {
class SPIRVTypeConverter;
class RewritePatternSet;
class Pass;
struct ScfToSPIRVContextImpl;
class ModuleOp;
template <typename T> class OperationPass;

} // namespace mlir

namespace imex {
#define GEN_PASS_DECL_CONVERTGPUXTOSPIRV
#include "imex/Conversion/Passes.h.inc"

void populateGPUPrintfToSPIRVPatterns(mlir::SPIRVTypeConverter &typeConverter,
                                      mlir::RewritePatternSet &patterns);

/// Create a pass
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertGPUXToSPIRVPass(bool mapMemorySpace = true);

} // namespace imex

#endif // IMEX_GPUTOSPIRV_PASS_H_
