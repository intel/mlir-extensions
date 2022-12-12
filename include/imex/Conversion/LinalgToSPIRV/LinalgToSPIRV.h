//===- LinalgToSPIRV.h - LinalgToSPIRV conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the LinalgToSPIRV conversion, converting the Linalg
/// dialect to the SPIRV dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _LinalgToSPIRV_H_INCLUDED_
#define _LinalgToSPIRV_H_INCLUDED_

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace imex {

/// Populate the given list with patterns rewrite Linalg Ops
void populateLinalgToSPIRVConversionPatterns(
    ::mlir::LLVMTypeConverter &converter, ::mlir::RewritePatternSet &patterns);

/// Create a pass to convert the Linalg dialect to the SPIRV dialect.
std::unique_ptr<::mlir::Pass> createConvertLinalgToSPIRVPass();

} // namespace imex

#endif // _LinalgToSPIRV_H_INCLUDED_
