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
void populateMakeSignlessRewritesAndTarget(mlir::TypeConverter &converter,
                                           mlir::RewritePatternSet &patterns,
                                           mlir::ConversionTarget &target);

/// Convert types of various signedness to corresponding signless type.
std::unique_ptr<mlir::Pass> createMakeSignlessPass();
} // namespace imex
