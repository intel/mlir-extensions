// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>

namespace mlir {
struct LogicalResult;
class Value;
class PatternRewriter;
class Type;
namespace scf {
class ForOp;
}
} // namespace mlir

namespace plier {
class PyCallOp;
}

mlir::LogicalResult
lowerRange(plier::PyCallOp op, llvm::ArrayRef<mlir::Value> operands,
           llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs,
           mlir::PatternRewriter &rewriter,
           llvm::function_ref<void(mlir::scf::ForOp)> results = nullptr);
