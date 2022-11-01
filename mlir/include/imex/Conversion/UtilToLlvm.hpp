// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <functional>
#include <memory>

namespace mlir {
class LowerToLLVMOptions;
class MLIRContext;
class Pass;
} // namespace mlir

namespace imex {

/// Convert operations from the imex_util dialect to the LLVM dialect.
///
/// TODO: We cannot pass LowerToLLVMOptions directly to the pass as it requires
/// mlir context which is not yet available at this point, pass creation
/// function instead.
std::unique_ptr<mlir::Pass> createUtilToLLVMPass(
    std::function<mlir::LowerToLLVMOptions(mlir::MLIRContext &)> optsGetter);
} // namespace imex
