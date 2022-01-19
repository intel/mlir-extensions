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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

namespace mlir {
struct LogicalResult;
class PatternRewriter;
class Value;
class Location;
class OpBuilder;
class Type;
class Region;
namespace scf {
class ForOp;
class WhileOp;
} // namespace scf
} // namespace mlir

namespace plier {
class GetiterOp;
}

namespace plier {
bool canLowerWhileToFor(mlir::scf::WhileOp whileOp);
llvm::SmallVector<mlir::scf::ForOp, 2> lowerWhileToFor(
    mlir::scf::WhileOp whileOp, mlir::PatternRewriter &builder,
    llvm::function_ref<std::tuple<mlir::Value, mlir::Value, mlir::Value>(
        mlir::OpBuilder &, mlir::Location)>
        getBounds,
    llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Location,
                                   mlir::Type, mlir::Value)>
        getIterVal);
mlir::LogicalResult lowerWhileToFor(
    plier::GetiterOp getiter, mlir::PatternRewriter &builder,
    llvm::function_ref<std::tuple<mlir::Value, mlir::Value, mlir::Value>(
        mlir::OpBuilder &, mlir::Location)>
        getBounds,
    llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Location,
                                   mlir::Type, mlir::Value)>
        getIterVal,
    llvm::function_ref<void(mlir::scf::ForOp)> results = nullptr);

mlir::LogicalResult naivelyFuseParallelOps(mlir::Region &region);
mlir::LogicalResult prepareForFusion(mlir::Region &region);
} // namespace plier
