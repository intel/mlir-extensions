// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <functional>
#include <memory>
#include <string>

namespace mlir {
class OpPassManager;
class Pass;
} // namespace mlir

namespace imex {
/// Create composite pass, which runs selected set of passes until fixed point
/// or maximum number of iterations reached.
std::unique_ptr<mlir::Pass>
createCompositePass(std::string name,
                    std::function<void(mlir::OpPassManager &)> populateFunc);
} // namespace imex
