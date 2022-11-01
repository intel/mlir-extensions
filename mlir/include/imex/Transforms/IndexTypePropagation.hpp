// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace mlir {
class RewritePatternSet;
class MLIRContext;
} // namespace mlir

namespace imex {
void populateIndexPropagatePatterns(mlir::MLIRContext &context,
                                    mlir::RewritePatternSet &patterns);
}
