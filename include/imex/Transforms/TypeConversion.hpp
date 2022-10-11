//===- TypeConversion.hpp - ------------------------------------*-*- C++
//-*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
///
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
class ConversionTarget;
class MLIRContext;
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace imex {
void populateControlFlowTypeConversionRewritesAndTarget(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target);

// void populateTupleTypeConverter(mlir::MLIRContext &context,
//                                 mlir::TypeConverter &typeConverter);

// void populateTupleTypeConversionRewritesAndTarget(
//     mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
//     mlir::ConversionTarget &target);
} // namespace imex
