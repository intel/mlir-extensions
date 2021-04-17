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

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir
{
class TypeConverter;
}

namespace plier
{
struct FuncOpSignatureConversion : public mlir::OpRewritePattern<mlir::FuncOp>
{
    FuncOpSignatureConversion(mlir::TypeConverter& conv,
                              mlir::MLIRContext* ctx);

    /// Hook for derived classes to implement combined matching and rewriting.
    mlir::LogicalResult
    matchAndRewrite(mlir::FuncOp funcOp, mlir::PatternRewriter &rewriter) const override;

private:
    mlir::TypeConverter& converter;
};
}
