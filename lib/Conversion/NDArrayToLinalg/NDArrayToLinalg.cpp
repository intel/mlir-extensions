//===- NDArrayToLinalg.cpp - NDArrayToLinalg conversion  -------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the NDArrayToLinalg conversion, converting the
/// NDArray dialect to the Linalg and helper dialects.
///
/// NDArrays will mostly lower to ::mlir::tensors and operations in
/// linalg/arith. In-place semantics are mapped to memrefs through the
/// bufferization dialect.
///
/// Currently we do not support propagating device data across function
/// boundaries.
///
/// The pass is based on a ConversionTarget, TypeConverters, legality checks and
/// conversion patterns.
///
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/NDArrayToLinalg/NDArrayToLinalg.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Dialect/NDArray/Transforms/Utils.h>
#include <imex/Utils/ArithUtils.h>
#include <imex/Utils/PassUtils.h>

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Pass/Pass.h>

#include <optional>

namespace imex {
#define GEN_PASS_DEF_CONVERTNDARRAYTOLINALG
#include "imex/Conversion/Passes.h.inc"
} // namespace imex

namespace imex {

/// Create a linalg generic op from given output, input and body
template <typename V, typename B>
auto createParFor(mlir::Location &loc, mlir::OpBuilder &builder, uint64_t rank,
                  ::mlir::Value out, const V &inputs, B bBuilder) {
  // map for output and input
  const ::mlir::AffineMap map =
      ::mlir::AffineMap::getMultiDimIdentityMap(rank, builder.getContext());
  ::mlir::SmallVector<::mlir::AffineMap> maps(1 + inputs.size(), map);
  ::mlir::SmallVector<::mlir::utils::IteratorType> iterators(
      rank, mlir::utils::IteratorType::parallel);

  return builder.create<::mlir::linalg::GenericOp>(
      loc, out.getType(), inputs, out, maps, iterators, bBuilder);
}

// *******************************
// ***** Individual patterns *****
// *******************************

namespace {

/// Convert ndarray.copy and its return type to memref.alloc + memref.copy.
struct CopyLowering : public ::mlir::OpRewritePattern<::imex::ndarray::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::CopyOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // check output type and get operands
    auto srcArTyp = op.getSource().getType();
    auto retArTyp = op.getType();
    auto loc = op.getLoc();
    auto src = op.getSource();
    auto rank = srcArTyp.getRank();
    ::imex::ValVec dynDims;

    // get dynamic shape
    auto tTyp = mlir::cast<::mlir::TensorType>(src.getType());
    for (int64_t i = 0; i < rank; ++i) {
      if (tTyp.isDynamicDim(i)) {
        dynDims.emplace_back(
            rewriter.createOrFold<::mlir::tensor::DimOp>(loc, src, i));
      }
    }
    // alloc memref
    auto mrTyp =
        ::mlir::MemRefType::get(tTyp.getShape(), tTyp.getElementType());
    auto mr = rewriter.create<::mlir::memref::AllocOp>(
        loc, mrTyp, dynDims, rewriter.getI64IntegerAttr(8));
    // and copy if non-0
    if (!imex::ndarray::hasZeroSize(retArTyp.getShape())) {
      auto srcMR = createToMemRef(loc, rewriter, src, getMemRefType(srcArTyp));
      // create a region with given env, add copy op within it
      auto env = rewriter.getStringAttr("protect_copy_op");
      rewriter.create<::imex::region::EnvironmentRegionOp>(
          loc, env, llvm::ArrayRef<mlir::Value>(), llvm::ArrayRef<mlir::Type>(),
          [&srcMR, &mr](::mlir::OpBuilder &builder, ::mlir::Location loc) {
            (void)builder.create<::mlir::memref::CopyOp>(loc, srcMR, mr);
            (void)builder.create<::imex::region::EnvironmentRegionYieldOp>(loc);
          });
    }
    // convert memref to tensor
    auto res = rewriter.create<::mlir::bufferization::ToTensorOp>(
        loc, retArTyp, mr, /*restrict=*/true,
        /*writable=*/true);
    rewriter.replaceOp(op, res);

    return ::mlir::success();
  }
};

/// Convert NDArray's ReshapeOp and its return type to Linalg/tensor.
/// Optionally creates a copy first.
struct ReshapeLowering
    : public ::mlir::OpRewritePattern<::imex::ndarray::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::ReshapeOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // check output type and get operands
    auto loc = op.getLoc();
    auto src = op.getSource();
    auto shape = op.getShape();

    if (op.getCopy().value_or(false)) {
      src = rewriter.create<::imex::ndarray::CopyOp>(
          loc, op.getSource().getType(), op.getSource());
    }

    auto shapeT = rewriter.create<::mlir::tensor::FromElementsOp>(loc, shape);
    rewriter.replaceOpWithNewOp<::mlir::tensor::ReshapeOp>(op, op.getType(),
                                                           src, shapeT);

    return ::mlir::success();
  }
};

/// Convert NDArray's subview to memref::subview.
/// Adjusted from NTensor
struct SubviewLowering
    : public ::mlir::OpRewritePattern<::imex::ndarray::SubviewOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::SubviewOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    auto srcTnsr = op.getSource();
    auto loc = op->getLoc();

    // convert src array to memref
    auto srcArType = srcTnsr.getType();
    auto srcMRType = imex::getMemRefType(op.getContext(), srcArType.getShape(),
                                         srcArType.getElementType());
    auto srcMR = createToMemRef(loc, rewriter, srcTnsr, srcMRType);

    auto offsets = ::mlir::getMixedValues(op.getStaticOffsets(),
                                          op.getOffsets(), rewriter);
    auto sizes =
        ::mlir::getMixedValues(op.getStaticSizes(), op.getSizes(), rewriter);
    auto strides = ::mlir::getMixedValues(op.getStaticStrides(),
                                          op.getStrides(), rewriter);

    auto resMRType = mlir::cast<::mlir::MemRefType>(
        ::mlir::memref::SubViewOp::inferRankReducedResultType(
            op.getType().getShape(), srcMRType, offsets, sizes, strides));

    auto sw = rewriter.create<::mlir::memref::SubViewOp>(
        loc, resMRType, srcMR, offsets, sizes, strides);

    // convert result to tensor
    auto res = rewriter.create<::mlir::bufferization::ToTensorOp>(
        loc, srcTnsr.getType(), sw,
        /*restrict=*/true, /*writable=*/true);
    rewriter.replaceOp(op, res.getResult());

    return ::mlir::success();
  }
};

/// Convert NDArray's insert_slice to memref
struct InsertSliceLowering
    : public ::mlir::OpRewritePattern<::imex::ndarray::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::InsertSliceOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // get operators
    auto src = op.getSource();
    auto dst = op.getDestination();
    auto srcTyp = src.getType();
    auto dstTyp = dst.getType();

    auto srcMRTyp = getMemRefType(op.getContext(), srcTyp.getShape(),
                                  srcTyp.getElementType());
    mlir::Value srcMR = createToMemRef(loc, rewriter, src, srcMRTyp);

    auto dstMRTyp = getMemRefType(op.getContext(), dstTyp.getShape(),
                                  dstTyp.getElementType());
    auto dstDefOp = dst.getDefiningOp();
    bool dstIsConst =
        dstDefOp && (mlir::isa<mlir::memref::GetGlobalOp>(dstDefOp) ||
                     dstDefOp->hasTrait<mlir::OpTrait::ConstantLike>());
    assert(!dstIsConst &&
           "InsertSliceOp does not support constant destination");
    auto dstMR = createToMemRef(loc, rewriter, dst, dstMRTyp, dstIsConst);

    auto slcOffs = ::mlir::getMixedValues(op.getStaticOffsets(),
                                          op.getOffsets(), rewriter);
    auto slcSizes =
        ::mlir::getMixedValues(op.getStaticSizes(), op.getSizes(), rewriter);
    auto slcStrides = ::mlir::getMixedValues(op.getStaticStrides(),
                                             op.getStrides(), rewriter);

    auto view = rewriter.create<::mlir::memref::SubViewOp>(
        loc, dstMR, slcOffs, slcSizes, slcStrides);

    auto srcRank = srcMRTyp.getRank();
    auto dstRank = dstMRTyp.getRank();
    // FIXME properly handle broadcasting
    if (srcRank == 0) {
      // emit a loop that broadcasts a scalar to dst shape
      // construct broadcasting affine map; srcRank==0 case is simple
      auto srcMap =
          ::mlir::AffineMap::get(dstRank, srcRank, {}, rewriter.getContext());
      auto dstMap = rewriter.getMultiDimIdentityMap(dstRank);
      ::mlir::SmallVector<mlir::utils::IteratorType> iterators(
          dstRank, ::mlir::utils::IteratorType::parallel);
      auto copyOp = rewriter.create<::mlir::linalg::GenericOp>(
          loc, srcMR, view.getResult(), ::mlir::ArrayRef({srcMap, dstMap}),
          iterators,
          [](::mlir::OpBuilder &b, ::mlir::Location loc,
             ::mlir::ValueRange args) {
            b.create<::mlir::linalg::YieldOp>(loc, args.front());
          });
      rewriter.replaceOp(op, copyOp);
      return ::mlir::success();
    }

    rewriter.replaceOpWithNewOp<::mlir::memref::CopyOp>(op, srcMR, view);
    return ::mlir::success();
  }
};

/// Convert NDArray's linspace and its return type to Linalg/tensor.
/// Also needs some arith and affine (for linalg::genericop).
struct LinSpaceLowering
    : public ::mlir::OpRewritePattern<::imex::ndarray::LinSpaceOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::LinSpaceOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto start = op.getStart();
    auto stop = op.getStop();
    auto count = op.getNum();
    bool endpoint = op.getEndpoint();
    auto retArTyp = op.getType();
    auto rank = retArTyp.getRank();
    auto elTyp = retArTyp.getElementType();

    if (!(start.getType().isIntOrIndexOrFloat() &&
          stop.getType().isIntOrIndexOrFloat() &&
          count.getType().isIntOrIndex() && retArTyp)) {
      return ::mlir::failure();
    } // FIXME type promotion

    // cast types and get step
    auto bw = elTyp.isIndex() ? 64 : elTyp.getIntOrFloatBitWidth();
    ::mlir::Type cType =
        bw > 32 ? rewriter.getF64Type()
                : (bw > 16 ? rewriter.getF32Type() : rewriter.getF16Type());
    count = createIndexCast(loc, rewriter, count);
    start = createCast(loc, rewriter, start, cType);
    stop = createCast(loc, rewriter, stop, cType);
    auto step =
        createStepLinSpace(rewriter, loc, start, stop, count, endpoint, cType);

    // init tensor
    auto tensor = retArTyp.hasStaticShape()
                      ? rewriter
                            .create<::mlir::tensor::EmptyOp>(
                                loc, retArTyp.getShape(), elTyp)
                            .getResult()
                      : createEmptyTensor(rewriter, loc, elTyp, {count});

    // The loop body fills with values
    // accepting no input, the lambda simply captures start and step
    auto body = [&](::mlir::OpBuilder &builder, ::mlir::Location loc,
                    ::mlir::ValueRange args) {
      auto dim = getIntAttr(builder, 0);
      auto idx =
          createCast(loc, builder,
                     builder.create<::mlir::linalg::IndexOp>(loc, dim), cType);
      ::mlir::Value val = builder.createOrFold<::mlir::arith::AddFOp>(
          loc, builder.createOrFold<::mlir::arith::MulFOp>(loc, step, idx),
          start);
      (void)builder.create<::mlir::linalg::YieldOp>(
          loc, createCast(loc, rewriter, val, elTyp));
    };

    auto res =
        createParFor(loc, rewriter, rank, tensor, ::mlir::ValueRange(), body);
    rewriter.replaceOp(op, res.getResult(0));

    return ::mlir::success();
  }
};

/// Convert ndarray.delete and its return type to memref.dealloc.
struct DeleteLowering
    : public ::mlir::OpRewritePattern<::imex::ndarray::DeleteOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::DeleteOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto inp = op.getInput();
    auto inpMR = createToMemRef(op.getLoc(), rewriter, inp,
                                getMemRefType(op.getInput().getType()));
    auto newOp =
        rewriter.replaceOpWithNewOp<::mlir::memref::DeallocOp>(op, inpMR);
    newOp->setAttrs(op->getAttrs());

    return ::mlir::success();
  }
};

/// Convert ndarray.cast_elemtype to linalg.generic with cast
struct CastElemTypeLowering
    : public ::mlir::OpRewritePattern<::imex::ndarray::CastElemTypeOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::CastElemTypeOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = op.getInput();
    auto srcArType = op.getInput().getType();
    auto dstArType = op.getType();

    // verify identical shape
    assert(dstArType.getRank() == srcArType.getRank());
    assert(dstArType.getShape() == srcArType.getShape());

    auto dstElType = dstArType.getElementType();
    auto rank = srcArType.getRank();
    auto map = rewriter.getMultiDimIdentityMap(rank);
    ::mlir::SmallVector<mlir::utils::IteratorType> iterators(
        rank, ::mlir::utils::IteratorType::parallel);

    // identical types
    if (srcArType == dstArType) {
      if (op.getCopy().value_or(false)) {
        // emit a copy op
        rewriter.replaceOpWithNewOp<::imex::ndarray::CopyOp>(op, dstArType,
                                                             src);
        return ::mlir::success();
      } else {
        // eliminate cast op
        rewriter.replaceAllUsesWith(op.getResult(), op.getInput());
        rewriter.eraseOp(op);
        return ::mlir::success();
      }
    }

    auto dst = createEmptyTensor(rewriter, loc, dstArType, src);
    (void)rewriter.replaceOpWithNewOp<::mlir::linalg::GenericOp>(
        op, dstArType, src, dst, ::mlir::ArrayRef({map, map}), iterators,
        [dstElType](::mlir::OpBuilder &b, ::mlir::Location loc,
                    ::mlir::ValueRange args) {
          auto val = createCast(loc, b, args[0], dstElType);
          b.create<::mlir::linalg::YieldOp>(loc, val);
        });

    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

/// Convert NDArray to Linalg.
/// After success, no more NDArray should be left, replaced by Linalg & Affine
/// & Arith. Use a type converter to get rid of NDArrayType.
struct ConvertNDArrayToLinalgPass
    : public ::imex::impl::ConvertNDArrayToLinalgBase<
          ConvertNDArrayToLinalgPass> {

  ConvertNDArrayToLinalgPass() = default;

  void runOnOperation() override {
    auto root = this->getOperation();
    auto &ctxt = getContext();
    ::mlir::IRRewriter rewriter(&getContext());

    root->walk([&](::mlir::Operation *op) {
      mlir::Value base;
      if (auto iOp = mlir::dyn_cast<imex::ndarray::InsertSliceOp>(op)) {
        base = iOp.getDestination();
      } else if (auto svOp = mlir::dyn_cast<imex::ndarray::SubviewOp>(op)) {
        base = svOp.getSource();
      }
      if (base) {
        auto defOp = base.getDefiningOp();
        if (defOp && (defOp->hasTrait<mlir::OpTrait::ConstantLike>() ||
                      mlir::isa<mlir::memref::GetGlobalOp>(defOp))) {
          mlir::OpBuilder::InsertionGuard g(rewriter);
          rewriter.setInsertionPointAfter(defOp);
          auto copyOp = rewriter.create<imex::ndarray::CopyOp>(
              op->getLoc(), base.getType(), base);
          rewriter.replaceAllUsesExcept(base, copyOp.getResult(), copyOp);
        }
      }
    });

    ::mlir::ConversionTarget target(ctxt);
    // ...into Linalg, Affine, Tensor, Arith
    target.addLegalDialect<
        ::mlir::linalg::LinalgDialect, ::mlir::arith::ArithDialect,
        ::mlir::memref::MemRefDialect, ::mlir::tensor::TensorDialect,
        ::mlir::bufferization::BufferizationDialect, ::mlir::func::FuncDialect,
        ::imex::region::RegionDialect>();
    target.addLegalOp<imex::ndarray::SubviewOp, imex::ndarray::InsertSliceOp,
                      mlir::UnrealizedConversionCastOp>();

    // We convert almost all NDArray stuff...
    target.addDynamicallyLegalDialect<::imex::ndarray::NDArrayDialect>(
        [&](mlir::Operation *op) {
          return mlir::isa<imex::ndarray::SubviewOp,
                           imex::ndarray::InsertSliceOp>(op);
        });
    ::mlir::RewritePatternSet patterns(&ctxt);
    patterns.insert<LinSpaceLowering, ReshapeLowering, CopyLowering,
                    DeleteLowering, CastElemTypeLowering>(&ctxt);

    if (::mlir::failed(::mlir::applyPartialConversion(getOperation(), target,
                                                      ::std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

/// Create a pass to convert NDArray to Linalg
std::unique_ptr<::mlir::Pass> createConvertNDArrayToLinalgPass() {
  return std::make_unique<ConvertNDArrayToLinalgPass>();
}

} // namespace imex
