// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>

namespace mlir {
struct LogicalResult;
class Value;
class ValueRange;
class PatternRewriter;
class Type;
namespace scf {
class ForOp;
}
} // namespace mlir

namespace plier {
class PyCallOp;
}

namespace imex {
mlir::LogicalResult
lowerRange(plier::PyCallOp op, mlir::ValueRange operands,
           llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs,
           mlir::PatternRewriter &rewriter,
           llvm::function_ref<void(mlir::scf::ForOp)> results = nullptr);
}
