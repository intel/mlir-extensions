// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <functional>

#include "imex/Dialect/plier/Dialect.hpp"

#include <mlir/IR/PatternMatch.h>

namespace mlir {
class TypeConverter;
}

namespace imex {
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
} // namespace imex
