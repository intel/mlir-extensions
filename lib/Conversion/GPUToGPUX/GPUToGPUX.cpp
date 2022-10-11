//===- GPUToGPUX.cpp - GPUToGPUX conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the GPUToGPUX conversion, converting the GPU
/// dialect to the GPUX dialect.
///
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include <imex/Conversion/GPUToGPUX/GPUToGPUX.h>
#include <imex/Dialect/GPUX/IR/GPUXOps.h>
#include <imex/internal/PassWrapper.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>

#include <mlir/IR/BuiltinOps.h>

namespace imex {

static llvm::Optional<mlir::Value> getGpuStream(mlir::OpBuilder &builder,
                                                mlir::Operation *op) {
  assert(op);
  auto func = op->getParentOfType<mlir::func::FuncOp>();
  if (!func)
    return {};

  if (!llvm::hasSingleElement(func.getBody()))
    return {};

  auto &block = func.getBody().front();
  auto ops = block.getOps<imex::gpux::CreateStreamOp>();
  if (!ops.empty())
    return (*ops.begin()).getResult();

  mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(&block);
  auto loc = builder.getUnknownLoc();
  auto stream = builder.create<imex::gpux::CreateStreamOp>(loc).getResult();
  builder.setInsertionPoint(block.getTerminator());
  builder.create<imex::gpux::DestroyStreamOp>(loc, stream);
  return stream;
}

struct ExpandAllocOp : public mlir::OpRewritePattern<mlir::gpu::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::AllocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto stream = getGpuStream(rewriter, op);
    if (!stream)
      return mlir::failure();

    mlir::Type token = op.asyncToken() ? op.asyncToken().getType() : nullptr;
    rewriter.replaceOpWithNewOp<imex::gpux::AllocOp>(
        op, op.getType(), token, op.asyncDependencies(), *stream,
        op.dynamicSizes(), op.symbolOperands());

    return mlir::success();
  }
};

struct ExpandDeallocOp : public mlir::OpRewritePattern<mlir::gpu::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::DeallocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto stream = getGpuStream(rewriter, op);
    if (!stream)
      return mlir::failure();

    mlir::Type token = op.asyncToken() ? op.asyncToken().getType() : nullptr;
    rewriter.replaceOpWithNewOp<imex::gpux::DeallocOp>(
        op, token, op.asyncDependencies(), *stream, op.memref());

    return mlir::success();
  }
};

struct ExpandLaunchOp : public mlir::OpRewritePattern<mlir::gpu::LaunchFuncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::LaunchFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto gpuMod = mod.template lookupSymbol<mlir::gpu::GPUModuleOp>(
        op.getKernelModuleName());
    if (!gpuMod)
      return mlir::failure();

    auto gpuKernel =
        gpuMod.template lookupSymbol<mlir::gpu::GPUFuncOp>(op.getKernelName());
    if (!gpuKernel)
      return mlir::failure();

    auto stream = getGpuStream(rewriter, op);
    if (!stream)
      return mlir::failure();

    auto loc = op.getLoc();
    mlir::Value asyncToken = op.getAsyncToken();
    auto gpux_launch_func = rewriter.create<imex::gpux::LaunchFuncOp>(
        loc, *stream, gpuKernel, op.getGridSizeOperandValues(),
        op.getBlockSizeOperandValues(), op.dynamicSharedMemorySize(),
        op.operands(), asyncToken ? asyncToken.getType() : nullptr,
        op.getAsyncDependencies());
    rewriter.replaceOp(op, gpux_launch_func.getResults());
    return mlir::success();
  }
};

struct ExpandWaitOp : public mlir::OpRewritePattern<mlir::gpu::WaitOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::WaitOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto stream = getGpuStream(rewriter, op);
    if (!stream)
      return mlir::failure();

    mlir::Type token = op.asyncToken() ? op.asyncToken().getType() : nullptr;
    rewriter.replaceOpWithNewOp<imex::gpux::WaitOp>(
        op, token, op.asyncDependencies(), *stream);

    return mlir::success();
  }
};

struct GPUToGPUXPass : public ::imex::ConvertGPUToGPUXBase<GPUToGPUXPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns
        .insert<ExpandAllocOp, ExpandDeallocOp, ExpandLaunchOp, ExpandWaitOp>(
            ctx);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
}; // namespace imex

// Create a pass that convert GPU to GPUX
std::unique_ptr<::mlir::OperationPass<void>> createConvertGPUToGPUXPass() {
  return std::make_unique<GPUToGPUXPass>();
}

} // namespace imex
