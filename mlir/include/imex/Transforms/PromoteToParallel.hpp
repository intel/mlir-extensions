// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace mlir {
class RewritePatternSet;
}

namespace imex {
void populatePromoteToParallelPatterns(mlir::RewritePatternSet &patterns);
}
