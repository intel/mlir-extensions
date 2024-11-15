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
#include <imex/Dialect/Region/Transforms/RegionConversions.h>
#include <imex/Utils/ArithUtils.h>
#include <imex/Utils/PassUtils.h>

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
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

// /// @return type without a sign
// static mlir::Type makeSignlessType(mlir::Type type) {
//   if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
//     if (!intType.isSignless())
//       return mlir::IntegerType::get(intType.getContext(),
//       intType.getWidth());
//   }
//   return type;
// }

// /// @return operand cast to signless type if needed, val if not
// static mlir::Value doSignCast(mlir::OpBuilder &builder, mlir::Location &loc,
//                               mlir::Value val) {
//   auto origType = val.getType();
//   auto signlessType = makeSignlessType(origType);
//   if (signlessType != origType) {
//     val =
//         builder
//             .create<::mlir::UnrealizedConversionCastOp>(loc, signlessType,
//             val) .getResult(0);
//   }
//   return val;
// }

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
    auto srcArTyp =
        mlir::dyn_cast<::mlir::RankedTensorType>(op.getSource().getType());
    auto retArTyp = mlir::dyn_cast<::mlir::RankedTensorType>(op.getType());
    if (!(srcArTyp && retArTyp)) {
      return ::mlir::failure();
    }

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
          loc, env, std::nullopt, std::nullopt,
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
    auto retArTyp = mlir::dyn_cast<::mlir::RankedTensorType>(op.getType());
    auto srcArTyp =
        mlir::dyn_cast<::mlir::RankedTensorType>(op.getSource().getType());
    if (!(retArTyp && srcArTyp)) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto src = op.getSource();
    auto shape = op.getShape();

    if (op.getCopy().value_or(false)) {
      src = rewriter.create<::imex::ndarray::CopyOp>(loc, srcArTyp,
                                                     op.getSource());
    }

    auto shapeT = rewriter.create<::mlir::tensor::FromElementsOp>(loc, shape);
    rewriter.replaceOpWithNewOp<::mlir::tensor::ReshapeOp>(op, retArTyp, src,
                                                           shapeT);

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
    auto srcArType =
        mlir::dyn_cast_or_null<::mlir::ShapedType>(op.getSource().getType());
    auto resType = mlir::dyn_cast_or_null<::mlir::ShapedType>(op.getType());
    if (!resType || !srcArType)
      return mlir::failure();

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
            resType.getShape(), srcMRType, offsets, sizes, strides));

    auto sw = rewriter.create<::mlir::memref::SubViewOp>(
        loc, resMRType, srcMR, offsets, sizes, strides);

    // convert result to tensor
    auto res = rewriter.create<::mlir::bufferization::ToTensorOp>(
        loc, sw,
        /*restrict=*/true, /*writable=*/true);
    rewriter.replaceOp(op, res.getResult());

    return ::mlir::success();
  }
};

/// Convert NDArray's extract_slice to tensor.extract_slice.
struct ExtractSliceLowering
    : public ::mlir::OpRewritePattern<::imex::ndarray::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::ExtractSliceOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    auto srcTnsr = op.getSource();
    auto loc = op->getLoc();

    auto offsets = ::mlir::getMixedValues(op.getStaticOffsets(),
                                          op.getOffsets(), rewriter);
    auto sizes =
        ::mlir::getMixedValues(op.getStaticSizes(), op.getSizes(), rewriter);
    auto strides = ::mlir::getMixedValues(op.getStaticStrides(),
                                          op.getStrides(), rewriter);

    auto res = rewriter.create<::mlir::tensor::ExtractSliceOp>(
        loc, srcTnsr, offsets, sizes, strides);
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
    auto srcTyp = mlir::dyn_cast<::mlir::ShapedType>(src.getType());
    auto dstTyp = mlir::dyn_cast<::mlir::ShapedType>(dst.getType());
    if (!dstTyp || !srcTyp)
      return ::mlir::failure();

    auto srcMRTyp = getMemRefType(op.getContext(), srcTyp.getShape(),
                                  srcTyp.getElementType());
    auto dstMRTyp = getMemRefType(op.getContext(), dstTyp.getShape(),
                                  dstTyp.getElementType());
    mlir::Value srcMR = createToMemRef(loc, rewriter, src, srcMRTyp);
    auto dstMR = createToMemRef(loc, rewriter, dst, dstMRTyp);

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

/// Convert immutable_insert_slice to tensor
struct ImmutableInsertSliceLowering
    : public ::mlir::OpRewritePattern<::imex::ndarray::ImmutableInsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::ImmutableInsertSliceOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // get operators
    auto src = op.getSource();
    auto dst = op.getDestination();

    auto offsets = ::mlir::getMixedValues(op.getStaticOffsets(),
                                          op.getOffsets(), rewriter);
    auto sizes =
        ::mlir::getMixedValues(op.getStaticSizes(), op.getSizes(), rewriter);
    auto strides = ::mlir::getMixedValues(op.getStaticStrides(),
                                          op.getStrides(), rewriter);

    auto slice = rewriter.create<::mlir::tensor::InsertSliceOp>(
        loc, src, dst, offsets, sizes, strides);
    rewriter.replaceOp(op, slice.getResult());

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
    auto retArTyp = mlir::dyn_cast<::mlir::RankedTensorType>(op.getType());
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
    auto tensor = createEmptyTensor(rewriter, loc, elTyp, {count});

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

/// Convert NDArray's createOp and its return type to Linalg/tensor.
/// Also needs some arith and affine (for linalg::genericop).
struct CreateLowering
    : public ::mlir::OpRewritePattern<::imex::ndarray::CreateOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::CreateOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // check output type and get operands
    auto retArTyp = mlir::dyn_cast<::mlir::RankedTensorType>(op.getType());
    if (!retArTyp)
      return ::mlir::failure();
    auto value = op.getValue();

    // init tensor
    auto elTyp = retArTyp.getElementType();
    ::mlir::Value res = createEmptyTensor(rewriter, loc, elTyp, op.getShape());

    if (!ndarray::hasZeroSize(retArTyp.getShape()) && value) {
      res = createParFor(
                loc, rewriter, retArTyp.getRank(), res, ::mlir::ValueRange(),
                [&value](::mlir::OpBuilder &builder, ::mlir::Location loc,
                         ::mlir::ValueRange args) {
                  (void)builder.create<::mlir::linalg::YieldOp>(loc, value);
                })
                .getResult(0);
    }
    rewriter.replaceOp(op, res);

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
    // check output type and get operands
    auto inpArType =
        mlir::dyn_cast<::mlir::RankedTensorType>(op.getInput().getType());
    if (!inpArType) {
      return ::mlir::failure();
    }

    auto inp = op.getInput();
    auto inpMR =
        createToMemRef(op.getLoc(), rewriter, inp, getMemRefType(inpArType));
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
    auto srcArType =
        mlir::dyn_cast<::mlir::RankedTensorType>(op.getInput().getType());
    auto dstArType = mlir::dyn_cast<::mlir::RankedTensorType>(op.getType());
    if (!(srcArType && dstArType)) {
      return ::mlir::failure();
    }

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
    auto cast = rewriter.create<::mlir::linalg::GenericOp>(
        loc, dstArType, src, dst, ::mlir::ArrayRef({map, map}), iterators,
        [dstElType](::mlir::OpBuilder &b, ::mlir::Location loc,
                    ::mlir::ValueRange args) {
          auto val = createCast(loc, b, args[0], dstElType);
          b.create<::mlir::linalg::YieldOp>(loc, val);
        });
    rewriter.replaceOp(op, cast);

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
    auto &ctxt = getContext();
    // ::mlir::TypeConverter typeConverter;
    // // Convert unknown types to itself
    // auto convT2T = [](::mlir::Type type) { return type; };
    // // Convert NDArrayType to (tensorType)
    // auto convNDArray2RankedTensor =
    //     [](::imex::ndarray::NDArrayType type) -> std::optional<::mlir::Type>
    //     {
    //   return type.getTensorType();
    // };

    // typeConverter.addConversion(convT2T);
    // typeConverter.addConversion(convNDArray2RankedTensor);

    // auto materializeCast =
    //     [](::mlir::OpBuilder &builder, ::mlir::Type type,
    //        ::mlir::ValueRange inputs,
    //        ::mlir::Location loc) -> ::mlir::Value {
    //   if (inputs.size() == 1) {
    //     auto input = inputs[0];
    //     auto itype = input.getType();
    //     if (mlir::isa<::mlir::TensorType>(type) and
    //         mlir::isa<::mlir::TensorType>(itype)) {
    //       return builder.create<::mlir::tensor::CastOp>(loc, type, inputs)
    //           .getResult();
    //     }
    //     auto ttype = mlir::dyn_cast<::mlir::RankedTensorType>(itype);
    //     if (ttype && mlir::isa<::mlir::MemRefType>(type)) {
    //       return createToMemRef(loc, builder, input, type);
    //     }
    //     auto mrtype = mlir::dyn_cast<::mlir::MemRefType>(itype);
    //     if (mrtype && mlir::isa<::mlir::RankedTensorType>(type)) {
    //       return builder
    //           .create<::mlir::bufferization::ToTensorOp>(loc, type, input,
    //                                                      /*restrict=*/true)
    //           .getResult();
    //     }
    //   }
    //   return builder
    //       .create<::mlir::UnrealizedConversionCastOp>(loc, type, inputs)
    //       .getResult(0);
    // };
    // typeConverter.addSourceMaterialization(materializeCast);
    // typeConverter.addTargetMaterialization(materializeCast);

    // // At function boundaries we have actual memref semantics.
    // // We need to explicitly convert in/out arguments to memrefs.
    // // If we use tensors downstream passes will auto-convert to non-strided
    // // memrefs which will imply a copy (converting from strided to
    // non-strided
    // // requires a copy)
    // // We simply use a separate type-converter and materializations

    // ::mlir::TypeConverter typeConverter4Func;
    // // Convert NDArrayType to MemRefType
    // auto convNDArray2MemRef =
    //     [](::imex::ndarray::NDArrayType type) -> std::optional<::mlir::Type>
    //     {
    //   return type.getMemRefType();
    // };

    // typeConverter4Func.addConversion(convT2T);
    // typeConverter4Func.addConversion(convNDArray2MemRef);
    // typeConverter4Func.addSourceMaterialization(materializeCast);
    // typeConverter4Func.addTargetMaterialization(materializeCast);

    ::mlir::ConversionTarget target(ctxt);
    // We convert all NDArray stuff...
    target.addIllegalDialect<::imex::ndarray::NDArrayDialect>();
    // ...into Linalg, Affine, Tensor, Arith
    target.addLegalDialect<
        ::mlir::linalg::LinalgDialect, ::mlir::arith::ArithDialect,
        ::mlir::memref::MemRefDialect, ::mlir::tensor::TensorDialect,
        ::mlir::bufferization::BufferizationDialect,
        ::imex::region::RegionDialect>();
    // target.addLegalOp<::mlir::UnrealizedConversionCastOp>(); // FIXME

    // // make sure function boundaries use tensors (not NDArrays)
    // target.addDynamicallyLegalOp<::mlir::func::FuncOp>(
    //     [&](::mlir::func::FuncOp op) {
    //       return typeConverter4Func.isSignatureLegal(op.getFunctionType()) &&
    //              typeConverter4Func.isLegal(&op.getBody());
    //     });
    // target.addDynamicallyLegalOp<::mlir::func::ReturnOp, mlir::func::CallOp>(
    //     [&](mlir::Operation *op) { return typeConverter4Func.isLegal(op); });

    // target.addDynamicallyLegalOp<::imex::region::EnvironmentRegionOp,
    //                              ::imex::region::EnvironmentRegionYieldOp>(
    //     [&](mlir::Operation *op) { return typeConverter.isLegal(op); });

    ::mlir::RewritePatternSet patterns(&ctxt);
    patterns
        .insert<SubviewLowering, ExtractSliceLowering, InsertSliceLowering,
                ImmutableInsertSliceLowering, LinSpaceLowering, CreateLowering,
                CopyLowering, DeleteLowering, CastElemTypeLowering>(&ctxt);
    // ::imex::populateRegionTypeConversionPatterns(patterns, typeConverter);

    // // populate function boundaries using our special type converter
    // ::mlir::populateFunctionOpInterfaceTypeConversionPattern<
    //     ::mlir::func::FuncOp>(patterns, typeConverter4Func);
    // ::mlir::populateReturnOpTypeConversionPattern(patterns,
    // typeConverter4Func);
    // ::mlir::populateCallOpTypeConversionPattern(patterns,
    // typeConverter4Func);

    // ::mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
    //     typeConverter, patterns, target);

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
