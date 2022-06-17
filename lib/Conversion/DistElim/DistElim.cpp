// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Eliminating dist calls (falling back to local compute)

#include <imex/Conversion/DistElim/DistElim.h>

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>

// dummy: constant op
template <typename Op>
void _toConst(Op op, mlir::PatternRewriter &rewriter, int64_t v = 0) {
  auto attr = rewriter.getIndexAttr(v);
  rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, attr);
}

// do nothing
::mlir::LogicalResult dist::ElimRegisterPTensorOp::matchAndRewrite(
    ::dist::RegisterPTensorOp op, mlir::PatternRewriter &rewriter) const {
  _toConst(op, rewriter);
  return ::mlir::success();
}

// do nothing
::mlir::LogicalResult dist::ElimLocalOffsetsOp::matchAndRewrite(
    ::dist::LocalOffsetsOp op, mlir::PatternRewriter &rewriter) const {
  _toConst(op, rewriter);
  return ::mlir::success();
}

// return orignal (global) shape
::mlir::LogicalResult
dist::ElimLocalShapeOp::matchAndRewrite(::dist::LocalShapeOp op,
                                        mlir::PatternRewriter &rewriter) const {
  auto x = op.ptensor().getDefiningOp<::dist::RegisterPTensorOp>();
  assert(x);
  x.shape().dump();
  rewriter.replaceOp(op, x.shape());
  return ::mlir::success();
}

// replace with identity cast
::mlir::LogicalResult
dist::ElimAllReduceOp::matchAndRewrite(::dist::AllReduceOp op,
                                       mlir::PatternRewriter &rewriter) const {
#if 0
    auto loc = op.getLoc();

    ::mlir::ModuleOp module = op->getParentOfType<::mlir::ModuleOp>();
    auto *context = module.getContext();
    constexpr auto _f = "printf";
    if(!module.lookupSymbol<::mlir::func::FuncOp>(_f)) {
        // auto st = rewriter.getStringAttr("dummy");
        // auto fn = ::mlir::FlatSymbolRefAttr::get(st);
        auto fn = ::llvm::StringRef(_f);
        auto ft = rewriter.getFunctionType({}, {});
        // Insert the printf function into the body of the parent module.
        ::mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<::mlir::func::FuncOp>(loc, _f, ft);
    }
    auto fa = ::mlir::SymbolRefAttr::get(context, _f);
    auto fc = rewriter.create<::mlir::func::CallOp>(loc, fa, ::mlir::TypeRange{});
#endif
  rewriter.replaceOpWithNewOp<::mlir::tensor::CastOp>(op, op.tensor().getType(),
                                                      op.tensor());
  return ::mlir::success();
}
