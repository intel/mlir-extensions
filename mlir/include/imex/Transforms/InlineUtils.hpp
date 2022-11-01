// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace imex {
std::unique_ptr<mlir::Pass> createForceInlinePass();
}
