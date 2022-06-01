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

#include "loop_utils.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/IR/PatternMatch.h>

#include "mlir-extensions/Dialect/plier/dialect.hpp"
#include "mlir-extensions/Transforms/loop_utils.hpp"

mlir::LogicalResult
lowerRange(plier::PyCallOp op, mlir::ValueRange operands,
           llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs,
           mlir::PatternRewriter &rewriter,
           llvm::function_ref<void(mlir::scf::ForOp)> results) {
  if (!kwargs.empty())
    return mlir::failure();

  if ((operands.size() < 1 || operands.size() > 3) ||
      !llvm::all_of(operands, [](mlir::Value val) {
        return val.getType().isa<mlir::IntegerType>();
      }))
    return mlir::failure();

  mlir::Value val = op.getResult();
  if (!val.getUsers().empty()) {
    auto user = mlir::dyn_cast<plier::GetiterOp>(*val.getUsers().begin());
    auto getBounds = [&](mlir::OpBuilder &builder, mlir::Location loc) {
      auto lowerBound =
          (operands.size() >= 2
               ? operands[0]
               : builder.create<mlir::arith::ConstantIndexOp>(loc, 0));
      auto upperBound = (operands.size() >= 2 ? operands[1] : operands[0]);
      auto step = (operands.size() == 3
                       ? operands[2]
                       : builder.create<mlir::arith::ConstantIndexOp>(loc, 1));
      return std::make_tuple(lowerBound, upperBound, step);
    };
    auto getIndex = [](mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Type dst_type, mlir::Value index) {
      return builder.create<plier::CastOp>(loc, dst_type, index);
    };
    if (!user || mlir::failed(lowerWhileToFor(user, rewriter, getBounds,
                                              getIndex, results)))
      return mlir::failure();
  }

  if (val.getUsers().empty())
    rewriter.eraseOp(op);

  return mlir::success();
}
