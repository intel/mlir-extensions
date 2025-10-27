//===- ConvertToSPIRV.h - Converts everything to SPIR-V dialect - *-C++ -*-===//
//
// Copyright 2025 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IMEX_CONVERSION_CONVERTTOSPIRV_H
#define IMEX_CONVERSION_CONVERTTOSPIRV_H

#include "imex/Utils/XeCommon.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class ConversionTarget;
class SPIRVTypeConverter;
class Pass;
class Operation;
class RewritePatternSet;
template <typename T> class OperationPass;
} // namespace mlir

namespace imex {
#define GEN_PASS_DECL_CONVERTTOSPIRV
#include "imex/Conversion/Passes.h.inc"

std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertToSPIRVPass();

} // namespace imex
#endif
