// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace imex {
namespace ntensor {
/// This pass tries to remove redundant `ntensor.copy` ops.
std::unique_ptr<mlir::Pass> createCopyRemovalPass();
} // namespace ntensor
} // namespace imex
