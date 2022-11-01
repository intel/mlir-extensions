// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/CallLowering.hpp"
#include "imex/Dialect/imex_util/Dialect.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>

static mlir::Value skipCasts(mlir::Value val) {
  auto getArg = [](mlir::Value arg) -> mlir::Value {
    auto cast = arg.getDefiningOp<mlir::UnrealizedConversionCastOp>();
    if (!cast)
      return {};

    auto inputs = cast.getInputs();
    if (inputs.size() != 1)
      return {};

    return inputs.front();
  };

  while (auto arg = getArg(val))
    val = arg;

  return val;
};

mlir::LogicalResult imex::ExpandCallVarargs::matchAndRewrite(
    plier::PyCallOp op, mlir::PatternRewriter &rewriter) const {
  auto vararg = op.getVarargs();
  if (!vararg)
    return mlir::failure();

  vararg = skipCasts(vararg);

  auto varargType = vararg.getType().dyn_cast<mlir::TupleType>();
  if (!varargType)
    return mlir::failure();

  auto argsCount = op.getArgs().size();
  auto varargsCount = varargType.size();
  llvm::SmallVector<mlir::Value> args(argsCount + varargsCount);
  llvm::copy(op.getArgs(), args.begin());

  auto loc = op.getLoc();
  for (auto i : llvm::seq<size_t>(0, varargsCount)) {
    auto type = varargType.getType(i);
    auto index = rewriter.create<mlir::arith::ConstantIndexOp>(
        loc, static_cast<int64_t>(i));
    args[argsCount + i] =
        rewriter.create<imex::util::TupleExtractOp>(loc, type, vararg, index);
  }

  auto resType = op.getType();
  rewriter.replaceOpWithNewOp<plier::PyCallOp>(
      op, resType, op.getFunc(), args, mlir::Value(), op.getKwargs(),
      op.getFuncName(), op.getKwNames());
  return mlir::success();
}

mlir::LogicalResult
imex::CallOpLowering::matchAndRewrite(plier::PyCallOp op,
                                      mlir::PatternRewriter &rewriter) const {
  if (op.getVarargs())
    return mlir::failure();

  auto funcName = op.getFuncName();

  llvm::SmallVector<mlir::Value> args;
  args.reserve(op.getArgs().size() + 1);
  auto func = op.getFunc();
  if (func) {
    auto getattr = func.getDefiningOp<plier::GetattrOp>();
    if (getattr)
      args.emplace_back(skipCasts(getattr.getOperand()));
  }

  for (auto arg : op.getArgs())
    args.emplace_back(skipCasts(arg));

  llvm::SmallVector<std::pair<llvm::StringRef, mlir::Value>> kwargs;
  for (auto it : llvm::zip(op.getKwargs(), op.getKwNames())) {
    auto arg = skipCasts(std::get<0>(it));
    auto name = std::get<1>(it).cast<mlir::StringAttr>();
    kwargs.emplace_back(name.getValue(), arg);
  }

  auto loc = op.getLoc();
  return resolveCall(op, funcName, loc, rewriter, args, kwargs);
}
