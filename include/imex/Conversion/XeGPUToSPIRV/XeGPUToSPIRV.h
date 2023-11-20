//===- XeGPUToSPIRV.h - Conversion---------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements conversion of the XeGPU dialect
///
//===----------------------------------------------------------------------===//
#ifndef IMEX_CONVERSION_XEGPUTOSPIRV_H
#define IMEX_CONVERSION_XEGPUTOSPIRV_H

#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Transforms/DialectConversion.h>
namespace mlir {
class SPIRVTypeConverter;
class RewritePatternSet;
class Pass;
} // namespace mlir

namespace imex {
// XeGPU to VC Intrinsics pattern
void populateXeGPUToVCIntrinsicsPatterns(
    mlir::SPIRVTypeConverter &typeConverter, mlir::RewritePatternSet &patterns);
// XeGPU to genISA Intrinsics pattern
void populateXeGPUToGenISAPatterns(mlir::SPIRVTypeConverter &typeConverter,
                                   mlir::RewritePatternSet &patterns);
} // namespace imex

#endif // IMEX_CONVERSION_XEGPUTOSPIRV_H
