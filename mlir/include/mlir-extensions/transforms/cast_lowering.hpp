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

#include <functional>

#include "mlir-extensions/dialect/plier/dialect.hpp"

#include <mlir/IR/PatternMatch.h>

namespace mlir {
class TypeConverter;
}

namespace plier {
struct CastOpLowering : public mlir::OpRewritePattern<plier::CastOp> {
  using cast_t = std::function<mlir::Value(
      mlir::PatternRewriter &, mlir::Location, mlir::Value, mlir::Type)>;

  CastOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context,
                 cast_t cast_func = nullptr);

  mlir::LogicalResult
  matchAndRewrite(plier::CastOp op,
                  mlir::PatternRewriter &rewriter) const override;

private:
  mlir::TypeConverter &converter;
  cast_t castFunc;
};
} // namespace plier
