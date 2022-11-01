// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class TypeConverter;
class RewritePatternSet;
class ConversionTarget;
class Pass;
} // namespace mlir

namespace imex {
void populatePromoteBoolMemrefConversionRewritesAndTarget(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target);

std::unique_ptr<mlir::Pass> createPromoteBoolMemrefPass();
} // namespace imex
