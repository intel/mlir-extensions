// SPDX-FileCopyrightText: 2022 Intel Corporation
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
/// Convert util ops according to provided type converter.
void populateUtilConversionPatterns(mlir::TypeConverter &converter,
                                    mlir::RewritePatternSet &patterns,
                                    mlir::ConversionTarget &target);
} // namespace imex
