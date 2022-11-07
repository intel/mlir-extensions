// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class MLIRContext;
class Pass;
class RewritePatternSet;
} // namespace mlir

namespace imex {
namespace ntensor {
void populateResolveArrayOpsPatterns(mlir::RewritePatternSet &patterns);

/// This pass translates high level array manipulation ops into primitive
/// ops like `resolve_index`, `subview`, `load`, `store` etc.
std::unique_ptr<mlir::Pass> createResolveArrayOpsPass();
} // namespace ntensor
} // namespace imex
