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

#include "plier/rewrites/call_lowering.hpp"

static mlir::Value skipCasts(mlir::Value val) {
  auto getArg = [](mlir::Value arg) -> mlir::Value {
    auto cast = arg.getDefiningOp<mlir::UnrealizedConversionCastOp>();
    if (!cast)
      return {};

    auto inputs = cast.inputs();
    if (inputs.size() != 1)
      return {};

    return inputs.front();
  };
  while (auto arg = getArg(val))
    val = arg;

  return val;
};

mlir::LogicalResult
plier::CallOpLowering::matchAndRewrite(plier::PyCallOp op,
                                       mlir::PatternRewriter &rewriter) const {
  if (op.varargs())
    return mlir::failure();

  auto funcName = op.func_name();

  llvm::SmallVector<mlir::Value> args;
  args.reserve(op.args().size() + 1);
  auto func = op.func();
  auto getattr = mlir::dyn_cast_or_null<plier::GetattrOp>(func.getDefiningOp());
  if (getattr)
    args.emplace_back(skipCasts(getattr.getOperand()));

  for (auto arg : op.args())
    args.emplace_back(skipCasts(arg));

  llvm::SmallVector<std::pair<llvm::StringRef, mlir::Value>> kwargs;
  for (auto it : llvm::zip(op.kwargs(), op.kw_names())) {
    auto arg = skipCasts(std::get<0>(it));
    auto name = std::get<1>(it).cast<mlir::StringAttr>();
    kwargs.emplace_back(name.getValue(), arg);
  }

  auto loc = op.getLoc();
  return resolveCall(op, funcName, loc, rewriter, args, kwargs);
}
