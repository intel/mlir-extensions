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

#include <imex/Conversion/GPUToGPUX/GPUToGPUX.h>
#include <imex/Dialect/GPUX/IR/GPUXOps.h>
#include <imex/Utils/PassWrapper.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace imex {
#define GEN_PASS_DEF_CONVERTGPUTOGPUX
#include "imex/Conversion/Passes.h.inc"
} // namespace imex

namespace imex {

// This function creates a temporary stream if a stream is already not created
// in the function. If a stream is already present, it will just return that
// temporary stream to queue gpu operations on.

static mlir::Value getGpuStream(mlir::OpBuilder &builder, mlir::Operation *op) {
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
  auto stream =
      builder
          .create<imex::gpux::CreateStreamOp>(loc, mlir::Value{}, mlir::Value{})
          .getResult();
  builder.setInsertionPoint(block.getTerminator());
  builder.create<imex::gpux::DestroyStreamOp>(loc, stream);
  return stream;
}

// This pattern converts the gpu.alloc operation to gpux.alloc
// and adds a stream argument to it.
struct ConvertAllocOp : public mlir::OpRewritePattern<mlir::gpu::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::AllocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto stream = getGpuStream(rewriter, op);
    if (!stream)
      return mlir::failure();

    auto hostShared = op.getHostShared();
    mlir::Type token =
        op.getAsyncToken() ? op.getAsyncToken().getType() : nullptr;
    rewriter.replaceOpWithNewOp<imex::gpux::AllocOp>(
        op, op.getType(), token, op.getAsyncDependencies(), stream,
        op.getDynamicSizes(), op.getSymbolOperands(), hostShared);

    return mlir::success();
  }
};

// This pattern converts the gpu.dealloc operation to gpux.dealloc
// and adds a stream argument to it.
struct ConvertDeallocOp : public mlir::OpRewritePattern<mlir::gpu::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::DeallocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto stream = getGpuStream(rewriter, op);
    if (!stream)
      return mlir::failure();

    mlir::Type token =
        op.getAsyncToken() ? op.getAsyncToken().getType() : nullptr;
    rewriter.replaceOpWithNewOp<imex::gpux::DeallocOp>(
        op, token, op.getAsyncDependencies(), stream, op.getMemref());

    return mlir::success();
  }
};

// This pattern converts the gpu.launch_func operation to gpux.launch_func
// and adds a stream argument to it.
struct ConvertLaunchOp
    : public mlir::OpRewritePattern<mlir::gpu::LaunchFuncOp> {
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
        loc, stream, gpuKernel, op.getGridSizeOperandValues(),
        op.getBlockSizeOperandValues(), op.getDynamicSharedMemorySize(),
        op.getKernelOperands(), asyncToken ? asyncToken.getType() : nullptr,
        op.getAsyncDependencies());
    rewriter.replaceOp(op, gpux_launch_func.getResults());
    return mlir::success();
  }
};

// This pattern converts the gpu.wait operation to gpux.wait
// and adds a stream argument to it.
struct ConvertWaitOp : public mlir::OpRewritePattern<mlir::gpu::WaitOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::WaitOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto stream = getGpuStream(rewriter, op);
    if (!stream)
      return mlir::failure();

    mlir::Type token =
        op.getAsyncToken() ? op.getAsyncToken().getType() : nullptr;
    rewriter.replaceOpWithNewOp<imex::gpux::WaitOp>(
        op, token, op.getAsyncDependencies(), stream);

    return mlir::success();
  }
};

// This pattern converts the gpu.memcpy operation to gpux.memcpy
// and adds a stream argument to it.
struct ConvertMemcpyOp : public mlir::OpRewritePattern<mlir::gpu::MemcpyOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::MemcpyOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto stream = getGpuStream(rewriter, op);
    if (!stream)
      return mlir::failure();

    mlir::Type token =
        op.getAsyncToken() ? op.getAsyncToken().getType() : nullptr;
    rewriter.replaceOpWithNewOp<imex::gpux::MemcpyOp>(
        op, token, op.getAsyncDependencies(), stream, op.getDst(), op.getSrc());

    return mlir::success();
  }
};

// This pattern converts the gpu.memset operation to gpux.memset
// and adds a stream argument to it.
struct ConvertMemsetOp : public mlir::OpRewritePattern<mlir::gpu::MemsetOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::MemsetOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto stream = getGpuStream(rewriter, op);
    if (!stream)
      return mlir::failure();

    mlir::Type token =
        op.getAsyncToken() ? op.getAsyncToken().getType() : nullptr;
    rewriter.replaceOpWithNewOp<imex::gpux::MemsetOp>(
        op, token, op.getAsyncDependencies(), stream, op.getDst(),
        op.getValue());

    return mlir::success();
  }
};

// This pass converts the GPU dialect ops to our custom GPUX dialect ops
// which add a stream to the gpu dialect ops. These ops are then lowered
// LLVM dialect and eventually to stcl/l0 runtime calls.
struct GPUToGPUXPass : public imex::impl::ConvertGPUToGPUXBase<GPUToGPUXPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<ConvertAllocOp, ConvertDeallocOp, ConvertLaunchOp,
                    ConvertWaitOp, ConvertMemcpyOp, ConvertMemsetOp>(ctx);

    (void)mlir::applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

// Create a pass that convert GPU to GPUX
std::unique_ptr<::mlir::OperationPass<void>> createConvertGPUToGPUXPass() {
  return std::make_unique<GPUToGPUXPass>();
}

} // namespace imex
