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

#include "plier/transforms/uplift_math_calls.hpp"

#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

template <typename Op>
static mlir::Operation *replaceOp1(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::ValueRange args) {
  if (args.size() != 1)
    return nullptr;

  return builder.create<Op>(loc, args.front());
}

mlir::LogicalResult
plier::UpliftMathCalls::matchAndRewrite(mlir::CallOp op,
                                        mlir::PatternRewriter &rewriter) const {
  auto funcName = op.getCallee();
  if (funcName.empty())
    return mlir::failure();

  auto isNotValidType = [](mlir::Type t) { return !t.isIntOrFloat(); };

  if (llvm::any_of(op.getOperandTypes(), isNotValidType) ||
      op.getNumResults() != 1 ||
      llvm::any_of(op.getResultTypes(), isNotValidType))
    return mlir::failure();

  llvm::StringRef funcNameF =
      (funcName.front() == 'f' ? funcName.drop_front() : llvm::StringRef{});

  using func_t =
      mlir::Operation *(*)(mlir::OpBuilder &, mlir::Location, mlir::ValueRange);
  const std::pair<llvm::StringRef, func_t> handlers[] = {
      {"log", &replaceOp1<mlir::math::LogOp>},
      {"sqrt", &replaceOp1<mlir::math::SqrtOp>},
      {"exp", &replaceOp1<mlir::math::ExpOp>},
      {"sin", &replaceOp1<mlir::math::SinOp>},
      {"cos", &replaceOp1<mlir::math::CosOp>},
      {"erf", &replaceOp1<mlir::math::ErfOp>},
  };

  for (auto &handler : handlers) {
    auto name = handler.first;
    if (name == funcName || name == funcNameF) {
      auto res = handler.second(rewriter, op.getLoc(), op.operands());
      if (!res)
        return mlir::failure();

      assert(res->getNumResults() == op->getNumResults());
      rewriter.replaceOp(op, res->getResults());
      return mlir::success();
    }
  }
  return mlir::failure();
}
