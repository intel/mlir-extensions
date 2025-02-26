//===- ArithToVC.h - Conversion---------------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements conversion of the select arith dialect operations into
/// Func dialect calls to vc-intrinsics functions
///
//===----------------------------------------------------------------------===//
#ifndef IMEX_CONVERSION_ARITHTOVC_H
#define IMEX_CONVERSION_ARITHTOVC_H

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>

#include "imex/Utils/XeCommon.h"

namespace mlir {

class ConversionTarget;
class LLVMTypeConverter;
class Pass;
class Operation;
class RewritePatternSet;
template <typename T> class OperationPass;

namespace gpu {
class GPUModuleOp;
} // namespace gpu

} // namespace mlir

namespace imex {
#define GEN_PASS_DECL_CONVERTARITHTOVC
#include "imex/Conversion/Passes.h.inc"

void populateArithToVCPatterns(
    ::mlir::TypeConverter &typeConverter, ::mlir::RewritePatternSet &patterns,
    bool enableHighPrecisionInterimCalculation = false);
void configureArithToVCConversionLegality(::mlir::ConversionTarget &target);
std::unique_ptr<::mlir::OperationPass<::mlir::gpu::GPUModuleOp>>
createConvertArithToVCPass();

} // namespace imex
#endif
