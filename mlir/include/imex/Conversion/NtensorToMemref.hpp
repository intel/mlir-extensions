// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class ConversionTarget;
class MLIRContext;
class Pass;
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace imex {
void populateNtensorToMemrefRewritesAndTarget(mlir::MLIRContext &context,
                                              mlir::TypeConverter &converter,
                                              mlir::RewritePatternSet &patterns,
                                              mlir::ConversionTarget &target);

std::unique_ptr<mlir::Pass> createNtensorToMemrefPass();
} // namespace imex
