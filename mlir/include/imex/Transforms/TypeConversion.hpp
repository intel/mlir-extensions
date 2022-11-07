// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

void populateTupleTypeConverter(mlir::TypeConverter &typeConverter);

void populateTupleTypeConversionRewritesAndTarget(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target);
} // namespace imex
