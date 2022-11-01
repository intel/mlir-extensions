// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace imex {
/// Expand tuple into individual elements.
std::unique_ptr<mlir::Pass> createExpandTuplePass();
} // namespace imex
