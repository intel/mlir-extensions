//===- DistRuntimeToIDTR.cpp - DistRuntimeToIDTR Transform ------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements lowering distruntime ops to calls to IDTR.
//===----------------------------------------------------------------------===//

#include <imex/Dialect/DistRuntime/IR/DistRuntimeOps.h>
#include <imex/Dialect/DistRuntime/Transforms/Passes.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Dialect/NDArray/Utils/Utils.h>
#include <imex/Utils/PassUtils.h>
#include <imex/Utils/PassWrapper.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>

namespace imex {
#define GEN_PASS_DEF_LOWERDISTRUNTIMETOIDTR
#include <imex/Dialect/DistRuntime/Transforms/Passes.h.inc>
} // namespace imex

namespace imex {
namespace distruntime {

namespace {

// RuntimePrototypesOp -> func.func ops
// adding required function prototypes to the module level
struct RuntimePrototypes {

  // create function prototype fo given function name, arg-types and
  // return-types
  // If NoneTypes are present, it will generate mutiple functions, one for
  // each integer/float type, where all the NoneTypes get replaced by the
  // respective UnrankedMemref<elType>
  static void requireFunc(::mlir::Location &loc, ::mlir::OpBuilder &builder,
                          ::mlir::ModuleOp module, const char *fname,
                          ::mlir::TypeRange args, ::mlir::TypeRange results) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    // Insert before module terminator.
    builder.setInsertionPoint(module.getBody(),
                              std::prev(module.getBody()->end()));
    auto dataMRType = ::mlir::NoneType::get(builder.getContext());
    ::mlir::SmallVector<int> dmrs;
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i] == dataMRType)
        dmrs.emplace_back(i);
    }

    auto decl = [&](auto _fname, auto _args) {
      auto funcType = builder.getFunctionType(_args, results);
      auto func = ::mlir::func::FuncOp::create(builder, loc, _fname, funcType);
      func.setPrivate();
    };

    if (dmrs.empty()) {
      decl(fname, args);
    } else {
      ::imex::TypVec pargs(args);
      for (auto t :
           {::imex::ndarray::F64, ::imex::ndarray::F32, ::imex::ndarray::I64,
            ::imex::ndarray::I32, ::imex::ndarray::I16, ::imex::ndarray::I8,
            ::imex::ndarray::I1}) {
        auto elType = ::imex::ndarray::toMLIR(builder, t);
        auto mrtyp = ::mlir::UnrankedMemRefType::get(elType, {});
        for (auto i : dmrs) {
          pargs[i] = mrtyp;
        }
        auto tfname = mkTypedFunc(fname, elType);
        decl(tfname, pargs);
      }
    }
  }

  static void add_prototypes(::mlir::OpBuilder builder, ::mlir::Operation *op) {
    auto loc = op->getLoc();
    ::mlir::ModuleOp module = ::mlir::cast<mlir::ModuleOp>(op);
    auto indexType = builder.getIndexType();
    auto i64Type = builder.getI64Type();
    auto opType =
        builder.getIntegerType(sizeof(::imex::ndarray::ReduceOpId) * 8);
    auto i64MRType = ::mlir::UnrankedMemRefType::get(i64Type, {});
    // requireFunc will generate functions for multiple typed memref-types
    auto dataMRType = ::mlir::NoneType::get(builder.getContext());
    requireFunc(loc, builder, module, "printMemrefI64", {i64MRType}, {});
    auto idxMRType =
        ::mlir::UnrankedMemRefType::get(builder.getIndexType(), {});
    requireFunc(loc, builder, module, "printMemrefInd", {idxMRType}, {});
    requireFunc(loc, builder, module, "_idtr_nprocs", {i64Type}, {indexType});
    requireFunc(loc, builder, module, "_idtr_prank", {i64Type}, {indexType});
    requireFunc(loc, builder, module, "_idtr_reduce_all", {dataMRType, opType},
                {});
    requireFunc(loc, builder, module, "_idtr_copy_reshape",
                // team, gshape, loffs, lPart, ngshape, nloffs, nPart
                {i64Type, idxMRType, idxMRType, dataMRType, idxMRType,
                 idxMRType, dataMRType},
                {i64Type});
    requireFunc(loc, builder, module, "_idtr_update_halo",
                // team,    gshape,    loffs,     lPart,      bbOffset, bbShape,
                // lHalo,   rHalo, key
                {i64Type, idxMRType, idxMRType, dataMRType, idxMRType,
                 idxMRType, dataMRType, dataMRType, i64Type},
                {i64Type});
    requireFunc(loc, builder, module, "_idtr_wait",
                // handle
                {i64Type}, {});
    requireFunc(loc, builder, module, "_idtr_copy_permute",
                // team, gshape, loffs, lPart, nloffs, nPart, axes
                {i64Type, idxMRType, idxMRType, dataMRType, idxMRType,
                 dataMRType, idxMRType},
                {i64Type});
  }
};

/// Convert ::imex::distruntime::WaitOp into call to _idtr_wait
struct WaitOpPattern
    : public ::mlir::OpRewritePattern<::imex::distruntime::WaitOp> {
  using ::mlir::OpRewritePattern<::imex::distruntime::WaitOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::distruntime::WaitOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto fsa = rewriter.getStringAttr("_idtr_wait");
    rewriter.replaceOpWithNewOp<::mlir::func::CallOp>(
        op, fsa, ::mlir::TypeRange(), ::mlir::ValueRange{op.getHandle()});

    return ::mlir::success();
  }
};

struct CopyReshapeOpPattern
    : public ::mlir::OpRewritePattern<::imex::distruntime::CopyReshapeOp> {
  using ::mlir::OpRewritePattern<
      ::imex::distruntime::CopyReshapeOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::distruntime::CopyReshapeOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto lArray = op.getLArray();
    auto resType = op.getNlArray().getType();
    auto loc = op.getLoc();
    auto elType = resType.getElementType();
    auto team = op.getTeam();
    auto gShape = op.getGShape();
    auto lOffs = op.getLOffsets();
    auto ngShape = op.getNgShape();
    auto nlOffs = op.getNlOffsets();
    auto nlShape = op.getNlShape();

    // create output array with target size
    auto nlArray = ::mlir::tensor::EmptyOp::create(
        rewriter, loc, op.getNlArray().getType(), nlShape);

    auto idxType = rewriter.getIndexType();
    auto teamC = ::mlir::arith::ConstantOp::create(
        rewriter, loc, mlir::cast<::mlir::IntegerAttr>(team));
    auto gShapeMR = createURMemRefFromElements(rewriter, loc, idxType, gShape);
    auto lOffsMR = createURMemRefFromElements(rewriter, loc, idxType, lOffs);
    auto lArrayMR = ::imex::ndarray::mkURMemRef(loc, rewriter, lArray);
    auto ngShapeMR =
        createURMemRefFromElements(rewriter, loc, idxType, ngShape);
    auto nlOffsMR = createURMemRefFromElements(rewriter, loc, idxType, nlOffs);
    auto nlArrayMR = ::imex::ndarray::mkURMemRef(loc, rewriter, nlArray);

    auto fun =
        rewriter.getStringAttr(mkTypedFunc("_idtr_copy_reshape", elType));
    auto handle = ::mlir::func::CallOp::create(
        rewriter, loc, fun, rewriter.getI64Type(),
        ::mlir::ValueRange{teamC, gShapeMR, lOffsMR, lArrayMR, ngShapeMR,
                           nlOffsMR, nlArrayMR});
    rewriter.replaceOp(op, {handle.getResult(0), nlArray});
    return ::mlir::success();
  }
};

struct CopyPermuteOpPattern
    : public ::mlir::OpRewritePattern<::imex::distruntime::CopyPermuteOp> {
  using ::mlir::OpRewritePattern<
      ::imex::distruntime::CopyPermuteOp>::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::distruntime::CopyPermuteOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto lArray = op.getLArray();
    auto resType = op.getNlArray().getType();
    auto loc = op.getLoc();
    auto elType = resType.getElementType();
    auto team = op.getTeam();
    auto gShape = op.getGShape();
    auto lOffs = op.getLOffsets();
    auto nlOffs = op.getNlOffsets();
    auto nlShape = op.getNlShape();
    auto axes = op.getAxes();

    ::mlir::SmallVector<::mlir::Value> axesValues;
    for (auto axis : axes) {
      axesValues.emplace_back(
          ::mlir::arith::ConstantIndexOp::create(rewriter, loc, axis));
    }

    // create output array with target size
    auto nlArray = ::mlir::tensor::EmptyOp::create(
        rewriter, loc, op.getNlArray().getType(), nlShape);

    auto idxType = rewriter.getIndexType();
    auto teamC = ::mlir::arith::ConstantOp::create(
        rewriter, loc, mlir::cast<::mlir::IntegerAttr>(team));
    auto gShapeMR = createURMemRefFromElements(rewriter, loc, idxType, gShape);
    auto lOffsMR = createURMemRefFromElements(rewriter, loc, idxType, lOffs);
    auto lArrayMR = ::imex::ndarray::mkURMemRef(loc, rewriter, lArray);
    auto nlOffsMR = createURMemRefFromElements(rewriter, loc, idxType, nlOffs);
    auto nlArrayMR = ::imex::ndarray::mkURMemRef(loc, rewriter, nlArray);
    auto axesMR =
        createURMemRefFromElements(rewriter, loc, idxType, axesValues);

    auto fun =
        rewriter.getStringAttr(mkTypedFunc("_idtr_copy_permute", elType));
    auto handle = ::mlir::func::CallOp::create(
        rewriter, loc, fun, rewriter.getI64Type(),
        ::mlir::ValueRange{teamC, gShapeMR, lOffsMR, lArrayMR, nlOffsMR,
                           nlArrayMR, axesMR});
    rewriter.replaceOp(op, {handle.getResult(0), nlArray});
    return ::mlir::success();
  }
};

struct DistRuntimeToIDTRPass
    : public impl::LowerDistRuntimeToIDTRBase<DistRuntimeToIDTRPass> {

  DistRuntimeToIDTRPass() = default;

  void runOnOperation() override {

    ::mlir::OpBuilder builder(&getContext());
    RuntimePrototypes::add_prototypes(builder, this->getOperation());

    ::mlir::FrozenRewritePatternSet patterns;
    insertPatterns<CopyReshapeOpPattern, CopyPermuteOpPattern, WaitOpPattern>(
        getContext(), patterns);
    (void)::mlir::applyPatternsGreedily(this->getOperation(), patterns);
  }; // runOnOperation()

}; // DistRuntimeToIDTRPass

} // namespace
} // namespace distruntime

/// Create DistRuntimeToIDTRPass
std::unique_ptr<::mlir::Pass> createDistRuntimeToIDTRPass() {
  return std::make_unique<::imex::distruntime::DistRuntimeToIDTRPass>();
}
} // namespace imex
