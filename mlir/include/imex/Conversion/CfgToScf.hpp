// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace imex {
/// Converts function body from CFG form to SCF dialect ops.
std::unique_ptr<mlir::Pass> createCFGToSCFPass();
} // namespace imex
