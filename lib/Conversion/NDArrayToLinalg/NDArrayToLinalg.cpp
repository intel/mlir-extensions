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
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Dialect/NDArray/Transforms/Utils.h>
#include <imex/Dialect/Region/Transforms/RegionConversions.h>
#include <imex/Utils/ArithUtils.h>
#include <imex/Utils/PassUtils.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Pass/Pass.h>

#include <optional>

namespace imex {
#define GEN_PASS_DEF_CONVERTNDARRAYTOLINALG
#include "imex/Conversion/Passes.h.inc"
} // namespace imex

namespace imex {

/// @return type without a sign
static mlir::Type makeSignlessType(mlir::Type type) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (!intType.isSignless())
      return mlir::IntegerType::get(intType.getContext(), intType.getWidth());
  }
  return type;
}

/// @return operand cast to signless type if needed, val if not
static mlir::Value doSignCast(mlir::OpBuilder &builder, mlir::Location &loc,
                              mlir::Value val) {
  auto origType = val.getType();
  auto signlessType = makeSignlessType(origType);
  if (signlessType != origType) {
    val =
        builder
            .create<::mlir::UnrealizedConversionCastOp>(loc, signlessType, val)
            .getResult(0);
  }
  return val;
}

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

struct CastLowering
    : public ::mlir::OpConversionPattern<::imex::ndarray::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::CastOp op,
                  ::imex::ndarray::CastOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");
    auto src = adaptor.getSource();
    auto inTyp =
        mlir::dyn_cast<::mlir::RankedTensorType>(adaptor.getSource().getType());
    auto outTyp = mlir::dyn_cast<::mlir::RankedTensorType>(
        converter->convertType(op.getType()));

    if (!inTyp || !outTyp) {
      return ::mlir::failure();
    }

    if (outTyp == inTyp) {
      rewriter.replaceOp(op, src);
    } else {
      rewriter.replaceOpWithNewOp<::mlir::tensor::CastOp>(op, outTyp, src);
    }
    return ::mlir::success();
  }
};

/// Convert FromMemRefOp to bufferize.to_tensor
struct FromMemRefLowering
    : public ::mlir::OpConversionPattern<::imex::ndarray::FromMemRefOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::FromMemRefOp op,
                  ::imex::ndarray::FromMemRefOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<::mlir::bufferization::ToTensorOp>(
        op, adaptor.getInput(), /*restrict=*/true);

    return ::mlir::success();
  }
};

/// Lower to the input operand of the defining op.
struct ToTensorLowering
    : public ::mlir::OpConversionPattern<::imex::ndarray::ToTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::ToTensorOp op,
                  ::imex::ndarray::ToTensorOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOp(op, adaptor.getInput());

    return ::mlir::success();
  }
};

/// Convert NDArray's subview to memref::subview.
/// Adjusted from NTensor
struct SubviewLowering
    : public ::mlir::OpConversionPattern<::imex::ndarray::SubviewOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::SubviewOp op,
                  ::imex::ndarray::SubviewOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto srcTnsr = adaptor.getSource();
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

    auto offsets = ::mlir::getMixedValues(adaptor.getStaticOffsets(),
                                          adaptor.getOffsets(), rewriter);
    auto sizes = ::mlir::getMixedValues(adaptor.getStaticSizes(),
                                        adaptor.getSizes(), rewriter);
    auto strides = ::mlir::getMixedValues(adaptor.getStaticStrides(),
                                          adaptor.getStrides(), rewriter);

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
    : public ::mlir::OpConversionPattern<::imex::ndarray::ExtractSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::ExtractSliceOp op,
                  ::imex::ndarray::ExtractSliceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto srcTnsr = adaptor.getSource();
    auto loc = op->getLoc();

    auto offsets = ::mlir::getMixedValues(adaptor.getStaticOffsets(),
                                          adaptor.getOffsets(), rewriter);
    auto sizes = ::mlir::getMixedValues(adaptor.getStaticSizes(),
                                        adaptor.getSizes(), rewriter);
    auto strides = ::mlir::getMixedValues(adaptor.getStaticStrides(),
                                          adaptor.getStrides(), rewriter);

    auto res = rewriter.create<::mlir::tensor::ExtractSliceOp>(
        loc, srcTnsr, offsets, sizes, strides);
    rewriter.replaceOp(op, res.getResult());

    return ::mlir::success();
  }
};

/// Convert NDArray's DimOp to tensor::DimOp.
struct DimOpLowering : public mlir::OpConversionPattern<imex::ndarray::DimOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ndarray::DimOp op,
                  imex::ndarray::DimOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto srcTnsr = adaptor.getSource();
    auto srcType = mlir::dyn_cast<::mlir::TensorType>(srcTnsr.getType());
    if (!srcType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<::mlir::tensor::DimOp>(op, srcTnsr,
                                                       adaptor.getIndex());
    return mlir::success();
  }
};

/// Convert NDArray's LoadOp to tensor::ExtractOp.
/// Adjusted from NTensor
struct LoadOpLowering
    : public mlir::OpConversionPattern<imex::ndarray::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ndarray::LoadOp op,
                  imex::ndarray::LoadOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto srcArType =
        mlir::cast<imex::ndarray::NDArrayType>(op.getArray().getType());
    auto srcTnsr = adaptor.getArray();
    if (!mlir::isa<mlir::TensorType>(srcTnsr.getType()))
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");
    auto dstType = converter->convertType(op.getType());
    if (!dstType || dstType != srcArType.getElementType())
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::tensor::ExtractOp>(op, srcTnsr,
                                                         adaptor.getIndices());

    return mlir::success();
  }
};

/// Convert NDArray's insert_slice to memref
struct InsertSliceLowering
    : public ::mlir::OpConversionPattern<::imex::ndarray::InsertSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::InsertSliceOp op,
                  ::imex::ndarray::InsertSliceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // get operators
    auto src = adaptor.getSource();
    auto dst = adaptor.getDestination();
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

    auto slcOffs = ::mlir::getMixedValues(adaptor.getStaticOffsets(),
                                          adaptor.getOffsets(), rewriter);
    auto slcSizes = ::mlir::getMixedValues(adaptor.getStaticSizes(),
                                           adaptor.getSizes(), rewriter);
    auto slcStrides = ::mlir::getMixedValues(adaptor.getStaticStrides(),
                                             adaptor.getStrides(), rewriter);

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
    : public ::mlir::OpConversionPattern<
          ::imex::ndarray::ImmutableInsertSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::ImmutableInsertSliceOp op,
                  ::imex::ndarray::ImmutableInsertSliceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // get operators
    auto src = adaptor.getSource();
    auto dst = adaptor.getDestination();

    auto offsets = ::mlir::getMixedValues(adaptor.getStaticOffsets(),
                                          adaptor.getOffsets(), rewriter);
    auto sizes = ::mlir::getMixedValues(adaptor.getStaticSizes(),
                                        adaptor.getSizes(), rewriter);
    auto strides = ::mlir::getMixedValues(adaptor.getStaticStrides(),
                                          adaptor.getStrides(), rewriter);

    auto slice = rewriter.create<::mlir::tensor::InsertSliceOp>(
        loc, src, dst, offsets, sizes, strides);
    rewriter.replaceOp(op, slice.getResult());

    return ::mlir::success();
  }
};

/// Convert NDArray's linspace and its return type to Linalg/tensor.
/// Also needs some arith and affine (for linalg::genericop).
struct LinSpaceLowering
    : public ::mlir::OpConversionPattern<::imex::ndarray::LinSpaceOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::LinSpaceOp op,
                  ::imex::ndarray::LinSpaceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto start = adaptor.getStart();
    auto stop = adaptor.getStop();
    auto count = adaptor.getNum();
    bool endpoint = adaptor.getEndpoint();
    auto retArTyp = mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getType());
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
    : public ::mlir::OpConversionPattern<::imex::ndarray::CreateOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::CreateOp op,
                  ::imex::ndarray::CreateOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // check output type and get operands
    auto retArTyp = mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getType());
    if (!retArTyp)
      return ::mlir::failure();
    auto value = adaptor.getValue();

    // init tensor
    auto elTyp = retArTyp.getElementType();
    ::mlir::Value res =
        createEmptyTensor(rewriter, loc, elTyp, adaptor.getShape());

    if (!retArTyp.hasZeroSize() && value) {
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

/// Convert ndarray.copy and its return type to memref.alloc + memref.copy.
struct CopyLowering
    : public ::mlir::OpConversionPattern<::imex::ndarray::CopyOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::CopyOp op,
                  ::imex::ndarray::CopyOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // check output type and get operands
    auto srcArTyp =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getSource().getType());
    auto retArTyp = mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getType());
    if (!(srcArTyp && retArTyp)) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto src = adaptor.getSource();
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
    // and copy if not zero sized
    if (!retArTyp.hasZeroSize()) {
      auto srcMR =
          createToMemRef(loc, rewriter, src, srcArTyp.getMemRefType(src));
      // wrap copy in a region to mark it non-deletable or a gpu copy
      bool hasGPUEnv = ::imex::ndarray::hasGPUEnv(srcArTyp) ||
                       ::imex::ndarray::hasGPUEnv(retArTyp);
      std::string regName = hasGPUEnv ? "gpu_copy_op" : "protect_copy_op";
      auto env = rewriter.getStringAttr(regName);
      rewriter.create<::imex::region::EnvironmentRegionOp>(
          loc, env, std::nullopt, std::nullopt,
          [&srcMR, &mr](::mlir::OpBuilder &builder, ::mlir::Location loc) {
            (void)builder.create<::mlir::memref::CopyOp>(loc, srcMR, mr);
            (void)builder.create<::imex::region::EnvironmentRegionYieldOp>(loc);
          });
    }
    // convert memref to tensor
    auto res = rewriter.create<::mlir::bufferization::ToTensorOp>(
        loc, retArTyp.getTensorType(), mr, /*restrict=*/true,
        /*writable=*/true);
    rewriter.replaceOp(op, res);

    return ::mlir::success();
  }
};

/// Convert ndarray.delete and its return type to memref.dealloc.
struct DeleteLowering
    : public ::mlir::OpConversionPattern<::imex::ndarray::DeleteOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::DeleteOp op,
                  ::imex::ndarray::DeleteOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // check output type and get operands
    auto inpArType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getInput().getType());
    if (!inpArType) {
      return ::mlir::failure();
    }

    auto inp = adaptor.getInput();
    auto inpMR = createToMemRef(op.getLoc(), rewriter, inp,
                                inpArType.getMemRefType(inp));
    auto newOp =
        rewriter.replaceOpWithNewOp<::mlir::memref::DeallocOp>(op, inpMR);
    newOp->setAttrs(op->getAttrs());

    return ::mlir::success();
  }
};

/// Convert ndarray.cast_elemtype to linalg.generic with cast
struct CastElemTypeLowering
    : public ::mlir::OpConversionPattern<::imex::ndarray::CastElemTypeOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::CastElemTypeOp op,
                  ::imex::ndarray::CastElemTypeOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = adaptor.getInput();
    auto srcArType =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getInput().getType());
    auto dstArType = mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getType());
    if (!(srcArType && dstArType)) {
      return ::mlir::failure();
    }

    // verify identical shape
    assert(dstArType.getRank() == srcArType.getRank());
    assert(dstArType.getShape() == srcArType.getShape());

    auto srcType = srcArType.getTensorType();
    auto dstType = dstArType.getTensorType();
    auto dstElType = dstType.getElementType();

    auto rank = srcType.getRank();
    auto map = rewriter.getMultiDimIdentityMap(rank);
    ::mlir::SmallVector<mlir::utils::IteratorType> iterators(
        rank, ::mlir::utils::IteratorType::parallel);

    // identical types
    if (srcArType == dstArType) {
      if (adaptor.getCopy().value_or(false)) {
        // emit a copy op
        auto arSrc = op.getInput();
        auto copyOp =
            rewriter.create<::imex::ndarray::CopyOp>(loc, dstArType, arSrc);
        rewriter.replaceOp(op, copyOp.getResult());
        return ::mlir::success();
      } else {
        // eliminate cast op
        rewriter.replaceAllUsesWith(op.getResult(), op.getInput());
        rewriter.eraseOp(op);
        return ::mlir::success();
      }
    }

    auto dst = createEmptyTensor(rewriter, loc, dstType, src);
    auto cast = rewriter.create<::mlir::linalg::GenericOp>(
        loc, dstType, src, dst, ::mlir::ArrayRef({map, map}), iterators,
        [dstElType](::mlir::OpBuilder &b, ::mlir::Location loc,
                    ::mlir::ValueRange args) {
          auto val = createCast(loc, b, args[0], dstElType);
          b.create<::mlir::linalg::YieldOp>(loc, val);
        });
    rewriter.replaceOp(op, cast);

    return ::mlir::success();
  }
};

/// Convert NDArray's ReshapeOp and its return type to Linalg/tensor.
/// Optionally creates a copy first.
struct ReshapeLowering
    : public ::mlir::OpConversionPattern<::imex::ndarray::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::ReshapeOp op,
                  ::imex::ndarray::ReshapeOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // check output type and get operands
    auto retArTyp = mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getType());
    auto srcArTyp =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getSource().getType());
    if (!(retArTyp && srcArTyp)) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto src = adaptor.getSource();
    auto shape = adaptor.getShape();
    auto outTyp = retArTyp.getTensorType();

    if (adaptor.getCopy().value_or(false)) {
      auto arSrc = op.getSource();
      auto copyOp =
          rewriter.create<::imex::ndarray::CopyOp>(loc, srcArTyp, arSrc);
      auto toTensorOp =
          rewriter.create<::imex::ndarray::ToTensorOp>(loc, copyOp.getResult());
      src = toTensorOp.getResult();
    }

    auto shapeT = rewriter.create<::mlir::tensor::FromElementsOp>(loc, shape);
    rewriter.replaceOpWithNewOp<::mlir::tensor::ReshapeOp>(op, outTyp, src,
                                                           shapeT);

    return ::mlir::success();
  }
};

// function type for building body for linalg::generic
using BodyType = std::function<void(
    mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange args)>;

// any genericOp body needs to close with a yield
// we also add a cast op to "typ" if needed
template <typename T>
static void yield(mlir::OpBuilder &builder, ::mlir::Location loc,
                  ::mlir::Type typ, T val) {
  auto res = val;
  if (typ != res.getType()) {
    res = builder.create<::mlir::UnrealizedConversionCastOp>(loc, typ, res)
              .getResult(0);
  }
  (void)builder.create<mlir::linalg::YieldOp>(loc, res);
}

/// Trivial binop builders have simple equivalents in Arith.
/// The Arith ops are accepted as template arguments, one for ints and one for
/// floats. Currently only integers and floats are supported.
/// Currently unsigned int ops are not supported.
template <typename IOP, typename FOP = void>
static BodyType buildTrivialBinary(::mlir::Type typ) {
  return [typ](mlir::OpBuilder &builder, ::mlir::Location loc,
               ::mlir::ValueRange args) -> void {
    auto lhs = createCast(loc, builder, args[0], typ);
    auto rhs = createCast(loc, builder, args[1], typ);
    if (typ.isIntOrIndex()) {
      if constexpr (!std::is_same_v<IOP, void>) {
        yield(builder, loc, typ,
              builder.create<IOP>(loc, lhs, rhs).getResult());
        return;
      } else
        assert(0 &&
               "Found integer type but binary op not defined for integers");
    } else if (typ.isIntOrIndexOrFloat()) {
      if constexpr (!std::is_same_v<FOP, void>) {
        yield(builder, loc, typ,
              builder.create<FOP>(loc, lhs, rhs).getResult());
        return;
      } else
        assert(0 && "Found float type but binary op not defined for floats");
    } else {
      assert(0 && "Only integers and floats supported for binary ops");
    }
  };
}

static BodyType buildNegative(::mlir::Type typ) {
  return [typ](mlir::OpBuilder &builder, ::mlir::Location loc,
               ::mlir::ValueRange args) -> void {
    mlir::TypedAttr minus;
    if (typ.isUnsignedInteger()) {
      assert(0 && "Unsigned integers are not supported in negative op");
    } else if (typ.isIntOrIndex()) {
      minus = builder.getIntegerAttr(typ, -1);
    } else if (typ.isIntOrIndexOrFloat()) {
      minus = builder.getFloatAttr(typ, -1);
    } else {
      assert(0 && "Only integers and floats are supported");
    }
    // Emit a trivial multiply binop with a constant scalar -1
    auto scalar = builder.create<::mlir::arith::ConstantOp>(loc, typ, minus);
    auto mulOp =
        buildTrivialBinary<mlir::arith::MulIOp, mlir::arith::MulFOp>(typ);
    mulOp(builder, loc, {args[0], scalar});
  };
}

/// Trivial unary op builders have simple equivalents in Math.
/// The Math ops are accepted as template arguments, one for ints and one for
/// floats. Currently only integers and floats are supported.
/// Currently unsigned int ops are not supported.
template <typename IOP, typename FOP = void>
static BodyType buildTrivialUnary(::mlir::Type typ) {
  return [typ](mlir::OpBuilder &builder, ::mlir::Location loc,
               ::mlir::ValueRange args) -> void {
    auto srcTyp = args[0].getType();
    if (srcTyp.isIntOrIndex()) {
      if constexpr (!std::is_same_v<IOP, void>) {
        auto src = doSignCast(builder, loc, args[0]);
        yield(builder, loc, typ, builder.create<IOP>(loc, src).getResult());
        return;
      } else
        assert(0 &&
               "Found integer type but binary op not defined for integers");
    } else if (srcTyp.isIntOrIndexOrFloat()) {
      if constexpr (!std::is_same_v<FOP, void>) {
        yield(builder, loc, typ, builder.create<FOP>(loc, args[0]).getResult());
        return;
      } else
        assert(0 && "Found float type but binary op not defined for floats");
    } else {
      assert(0 && "Only integers and floats supported for binary ops");
    }
  };
}

/// get a body builder for given binary operation and result type.
/// Accepts a result type to insert a cast after the operation if needed
/// FIXME: add missing ops
static BodyType getBodyBuilder(::imex::ndarray::EWBinOpId binOp,
                               ::mlir::Type typ) {
  switch (binOp) {
  case ndarray::ADD:
    return buildTrivialBinary<mlir::arith::AddIOp, mlir::arith::AddFOp>(typ);
  case ndarray::ATAN2:
    return buildTrivialBinary<void, mlir::math::Atan2Op>(typ);
  case ndarray::FLOOR_DIVIDE:
    return buildTrivialBinary<mlir::arith::FloorDivSIOp>(typ);
  // case ndarray::LOGADDEXP] =
  // case ndarray::MATMUL] =
  case ndarray::MAXIMUM:
    return buildTrivialBinary<mlir::arith::MaxSIOp, mlir::arith::MaximumFOp>(
        typ);
  case ndarray::MINIMUM:
    return buildTrivialBinary<mlir::arith::MinSIOp, mlir::arith::MinimumFOp>(
        typ);
  case ndarray::MODULO:
    return buildTrivialBinary<mlir::arith::RemSIOp, mlir::arith::RemFOp>(typ);
  case ndarray::MULTIPLY:
    return buildTrivialBinary<mlir::arith::MulIOp, mlir::arith::MulFOp>(typ);
  case ndarray::POWER:
    return buildTrivialBinary<mlir::math::IPowIOp, mlir::math::PowFOp>(typ);
  case ndarray::SUBTRACT:
    return buildTrivialBinary<mlir::arith::SubIOp, mlir::arith::SubFOp>(typ);
  case ndarray::TRUE_DIVIDE:
    return buildTrivialBinary<::mlir::arith::DivSIOp, ::mlir::arith::DivFOp>(
        typ);
  // case ndarray::BITWISE_LEFT_SHIFT] =
  // case ndarray::BITWISE_RIGHT_SHIFT] =

  // case ndarray::EQUAL] =
  // case ndarray::GREATER] =
  // case ndarray::GREATER_EQUAL] =
  // case ndarray::LESS] =
  // case ndarray::LESS_EQUAL] =
  // case ndarray::NOT_EQUAL] =
  default:
    assert(0 && "unsupported elementwise binary operation");
  };
}

::mlir::Value createTosaOp(::mlir::Location loc,
                           ::imex::ndarray::EWBinOpId binOpId,
                           ::mlir::ConversionPatternRewriter &rewriter,
                           ::mlir::TensorType returnType, ::mlir::Value lhs,
                           ::mlir::Value rhs) {
  switch (binOpId) {
  case ndarray::BITWISE_AND:
    return rewriter
        .create<::mlir::tosa::BitwiseAndOp>(loc, returnType, lhs, rhs)
        .getResult();
  case ndarray::BITWISE_OR:
    return rewriter.create<::mlir::tosa::BitwiseOrOp>(loc, returnType, lhs, rhs)
        .getResult();
  case ndarray::BITWISE_XOR:
    return rewriter
        .create<::mlir::tosa::BitwiseXorOp>(loc, returnType, lhs, rhs)
        .getResult();
  case ndarray::LOGICAL_AND:
    return rewriter
        .create<::mlir::tosa::LogicalAndOp>(loc, returnType, lhs, rhs)
        .getResult();
  case ndarray::LOGICAL_OR:
    return rewriter.create<::mlir::tosa::LogicalOrOp>(loc, returnType, lhs, rhs)
        .getResult();
  case ndarray::LOGICAL_XOR:
    return rewriter
        .create<::mlir::tosa::LogicalXorOp>(loc, returnType, lhs, rhs)
        .getResult();
  default:
    break;
  };
  return ::mlir::Value();
}

/// Convert NDArray's elementwise binary operations and their return type to
/// Linalg/tensor. The given op's type is expected to convert to the appropriate
/// type (shape and element-type).
/// Also needs some arith and affine (for linalg::genericop).
struct EWBinOpLowering
    : public ::mlir::OpConversionPattern<::imex::ndarray::EWBinOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::EWBinOp op,
                  ::imex::ndarray::EWBinOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // We expect to lower NDArrays
    auto lhsArTyp =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getLhs().getType());
    auto rhsArTyp =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getRhs().getType());
    if (!lhsArTyp || !rhsArTyp) {
      return ::mlir::failure();
    }

    auto resType =
        mlir::cast<::imex::ndarray::NDArrayType>(op->getResult(0).getType())
            .getTensorType();
    // we assume the result type has been correctly promoted
    auto elTyp = resType.getElementType();

    // get the input as tensors
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto lhsTnsr = mlir::cast<::mlir::TensorType>(lhs.getType());
    auto rhsTnsr = mlir::cast<::mlir::TensorType>(rhs.getType());

    // we expect tensor types on operands
    auto lhsRank = lhsTnsr.getRank();
    auto rhsRank = rhsTnsr.getRank();

    auto rank = static_cast<unsigned>(std::max(lhsRank, rhsRank));

    const ::imex::ndarray::EWBinOpId binOpId =
        (::imex::ndarray::EWBinOpId)mlir::cast<::mlir::IntegerAttr>(
            adaptor.getOp())
            .getInt();

    ::mlir::Value newOp =
        createTosaOp(loc, binOpId, rewriter, resType, lhs, rhs);
    if (!newOp) {
      // generate linalg.generic loop

      // create output tensor with right dimensions
      auto tensor = createEmptyTensor(rewriter, loc, resType, {lhs, rhs});

      // we need affine maps for linalg::generic
      // as long as we have no proper support for rank-reduced sizes above
      // Linalg, we can handle only
      //   - explicitly rank-reduced inputs (such as explicit 0d tensors)
      //   - shapes with static dim-sizes of 1
      ::mlir::SmallVector<::mlir::AffineExpr> lhsExprs, rhsExprs, resExprs;
      for (int i = 0; i < lhsRank; ++i) {
        lhsExprs.emplace_back(lhsTnsr.getDimSize(i) == 1
                                  ? rewriter.getAffineConstantExpr(0)
                                  : rewriter.getAffineDimExpr(i));
      }
      for (int i = 0; i < rhsRank; ++i) {
        rhsExprs.emplace_back(rhsTnsr.getDimSize(i) == 1
                                  ? rewriter.getAffineConstantExpr(0)
                                  : rewriter.getAffineDimExpr(i));
      }
      for (unsigned i = 0; i < rank; ++i) {
        resExprs.emplace_back(rewriter.getAffineDimExpr(i));
      }
      auto lhsMap = ::mlir::AffineMap::get(resType.getRank(), /*symbolCount=*/0,
                                           lhsExprs, rewriter.getContext());
      auto rhsMap = ::mlir::AffineMap::get(resType.getRank(), /*symbolCount=*/0,
                                           rhsExprs, rewriter.getContext());
      auto resMap = rewriter.getMultiDimIdentityMap(resType.getRank());

      // we just make all dims parallel
      ::mlir::SmallVector<mlir::utils::IteratorType> iterators(
          rank, ::mlir::utils::IteratorType::parallel);

      // get the body builder for our binop and create genericop
      // FIXME: make createParFor ready for this
      auto bodyBuilder = getBodyBuilder(binOpId, elTyp);
      newOp =
          rewriter
              .create<::mlir::linalg::GenericOp>(
                  loc, tensor.getType(), ::mlir::ValueRange{lhs, rhs}, tensor,
                  ::mlir::ArrayRef<::mlir::AffineMap>{lhsMap, rhsMap, resMap},
                  iterators, bodyBuilder)
              .getResult(0);
    }
    rewriter.replaceOp(op, newOp);

    return ::mlir::success();
  }
};

/// get a body builder for given binary operation and result type.
/// Accepts a result type to insert a cast after the operation if needed
/// FIXME: add missing ops
static BodyType getBodyBuilder(::imex::ndarray::EWUnyOpId binOp,
                               ::mlir::Type typ) {
  switch (binOp) {
  case ndarray::ABS:
    return buildTrivialUnary<::mlir::math::AbsIOp, ::mlir::math::AbsFOp>(typ);
  case ndarray::ATAN:
    return buildTrivialUnary<void, ::mlir::math::AtanOp>(typ);
  case ndarray::CEIL:
    return buildTrivialUnary<void, ::mlir::math::CeilOp>(typ);
  case ndarray::COS:
    return buildTrivialUnary<void, ::mlir::math::CosOp>(typ);
  case ndarray::ERF:
    return buildTrivialUnary<void, ::mlir::math::ErfOp>(typ);
  case ndarray::EXP:
    return buildTrivialUnary<void, ::mlir::math::ExpOp>(typ);
  case ndarray::EXPM1:
    return buildTrivialUnary<void, ::mlir::math::ExpM1Op>(typ);
  case ndarray::FLOOR:
    return buildTrivialUnary<void, ::mlir::math::FloorOp>(typ);
  case ndarray::LOG:
    return buildTrivialUnary<void, ::mlir::math::LogOp>(typ);
  case ndarray::LOG1P:
    return buildTrivialUnary<void, ::mlir::math::Log1pOp>(typ);
  case ndarray::LOG2:
    return buildTrivialUnary<void, ::mlir::math::Log2Op>(typ);
  case ndarray::LOG10:
    return buildTrivialUnary<void, ::mlir::math::Log10Op>(typ);
  case ndarray::ROUND:
    return buildTrivialUnary<void, ::mlir::math::RoundOp>(typ);
  case ndarray::SIN:
    return buildTrivialUnary<void, ::mlir::math::SinOp>(typ);
  case ndarray::SQRT:
    return buildTrivialUnary<void, ::mlir::math::SqrtOp>(typ);
  case ndarray::TAN:
    return buildTrivialUnary<void, ::mlir::math::TanOp>(typ);
  case ndarray::TANH:
    return buildTrivialUnary<void, ::mlir::math::TanhOp>(typ);
  case ndarray::TRUNC:
    return buildTrivialUnary<void, ::mlir::math::TruncOp>(typ);
  case ndarray::NEGATIVE:
    return buildNegative(typ);
  default:
    assert(0 && "unsupported elementwise binary operation");
  };
}

/// Lower unary operations which are not natively provided in any of the MLIR
/// dialects.
/// @return resulting non-null value if the operation was lowered, null-value
/// otherwise
::mlir::Value createAggUnaryOp(::mlir::Location loc,
                               ::imex::ndarray::EWUnyOpId unyOpId,
                               ::mlir::ConversionPatternRewriter &rewriter,
                               ::imex::ndarray::NDArrayType returnType,
                               ::mlir::Value src) {
  switch (unyOpId) {
  case ndarray::SQUARE:
    return rewriter
        .create<::imex::ndarray::EWBinOp>(
            loc, returnType,
            getIntAttr(rewriter, ::imex::ndarray::MULTIPLY, 32), src, src)
        .getResult();
  default:
    break;
  };
  return ::mlir::Value();
}

/// Lower unary operations which are provided only by TOSA (and not by math or
/// arith).
/// @return resulting non-null value if the operation was lowered, null-value
/// otherwise
::mlir::Value createUnaryTosaOp(::mlir::Location loc,
                                ::imex::ndarray::EWUnyOpId unyOpId,
                                ::mlir::ConversionPatternRewriter &rewriter,
                                ::mlir::TensorType returnType,
                                ::mlir::Value src) {
  switch (unyOpId) {
  case ndarray::LOGICAL_NOT:
    return rewriter.create<mlir::tosa::LogicalNotOp>(loc, returnType, src)
        .getResult();
  default:
    break;
  };
  return ::mlir::Value();
}

/// Convert NDArray's elementwise unary operations and their return type to
/// Linalg/tensor. The given op's type is expected to convert to the appropriate
/// type (shape and element-type).
/// Also needs some arith and affine (for linalg::genericop).
struct EWUnyOpLowering
    : public ::mlir::OpConversionPattern<::imex::ndarray::EWUnyOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::EWUnyOp op,
                  ::imex::ndarray::EWUnyOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto arSrc = op.getSrc();
    // We expect to lower NDArrays
    auto srcArTyp =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(arSrc.getType());
    if (!srcArTyp) {
      // FIXME type casting
      return ::mlir::failure();
    }

    const ::imex::ndarray::EWUnyOpId unyOpId =
        (::imex::ndarray::EWUnyOpId)mlir::cast<::mlir::IntegerAttr>(
            adaptor.getOp())
            .getInt();
    if (unyOpId == ::imex::ndarray::POSITIVE) {
      // positive unary op is a no-op, remove it
      rewriter.replaceAllUsesWith(op.getResult(), op.getSrc());
      rewriter.eraseOp(op);
      return ::mlir::success();
    }

    auto resArType =
        mlir::cast<::imex::ndarray::NDArrayType>(op->getResult(0).getType());

    // generic lowering of non-MLIR-native ops
    ::mlir::Value newOp =
        createAggUnaryOp(loc, unyOpId, rewriter, resArType, arSrc);

    if (!newOp) { // not lowered yet
      // get the input/output tensor types
      auto src = adaptor.getSrc();
      auto srcTnsr = mlir::cast<::mlir::TensorType>(src.getType());
      auto resType = resArType.getTensorType();

      // we expect tensor types on operands
      auto elTyp = srcTnsr.getElementType();
      auto rank = srcTnsr.getRank();

      // try to lower to TOSA
      newOp = createUnaryTosaOp(loc, unyOpId, rewriter, resType, src);

      if (!newOp) { // still not lowered: generate linalg.generic loop
        // create output tensor with right dimensions
        auto tensor = createEmptyTensor(rewriter, loc, resType, {src});

        // we need affine maps for linalg::generic
        const ::mlir::AffineMap map = ::mlir::AffineMap::getMultiDimIdentityMap(
            rank, rewriter.getContext());
        ::mlir::SmallVector<::mlir::AffineMap> maps(2, map);
        // we just make all dims parallel
        ::mlir::SmallVector<mlir::utils::IteratorType> iterators(
            rank, ::mlir::utils::IteratorType::parallel);

        // get the body builder for our binop and create genericop
        // FIXME: make createParFor ready for this
        auto bodyBuilder = getBodyBuilder(unyOpId, elTyp);
        newOp = rewriter
                    .create<::mlir::linalg::GenericOp>(
                        loc, tensor.getType(), ::mlir::ValueRange{src}, tensor,
                        maps, iterators, bodyBuilder)
                    .getResult(0);
      }
    }

    rewriter.replaceOp(op, newOp);

    return ::mlir::success();
  }
};

// get a body builder for given binary operation and result type
// we accept a result type to insert a cast after the operation if needed
static BodyType getBodyBuilder(::imex::ndarray::ReduceOpId redOp,
                               ::mlir::Type typ) {
  switch (redOp) {
  case ::imex::ndarray::PROD:
    return getBodyBuilder(::imex::ndarray::MULTIPLY, typ);
  case ::imex::ndarray::SUM:
    return getBodyBuilder(::imex::ndarray::ADD, typ);
  case ::imex::ndarray::MAX:
    return getBodyBuilder(::imex::ndarray::MAXIMUM, typ);
  case ::imex::ndarray::MIN:
    return getBodyBuilder(::imex::ndarray::MINIMUM, typ);
  case ::imex::ndarray::MEAN:
  case ::imex::ndarray::STD:
  case ::imex::ndarray::VAR:
  default:
    assert(0 && "unsupported reduction operation");
  };
}

/// Convert NDArray's reduction operations and their return type to
/// Linalg/tensor. The given op's type is expected to convert to the appropriate
/// type (shape and element-type). Also needs some arith and affine (for
/// linalg::genericop).
// FIXME reduction over a subset of dimensionsstruct ReductionOpLowering
struct ReductionOpLowering
    : public ::mlir::OpConversionPattern<::imex::ndarray::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::ReductionOp op,
                  ::imex::ndarray::ReductionOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // We expect to lower NDArrays
    auto inpArTyp =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getInput().getType());
    if (!inpArTyp) {
      // fail if not, will be retried if operands get converted elsewhere
      return ::mlir::failure();
    }

    // we expect tensorType as operands
    auto inpTnsr = adaptor.getInput();
    auto inpTnsrTyp = mlir::cast<::mlir::TensorType>(inpTnsr.getType());

    // Get signless operands into vec
    ::mlir::SmallVector<mlir::Value, 1> oprnds = {inpTnsr};

    // determine resulting element type from converted op-type
    auto retArTyp =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getResult().getType());
    assert(retArTyp);
    auto retTyp = retArTyp.getTensorType();
    auto elTyp = retTyp.getElementType();
    auto sElTyp = makeSignlessType(elTyp);

    // build tensor using the resulting element type and shape
    // FIXME support reduction dimensions
    auto rank = static_cast<unsigned>(retTyp.getRank());
    assert(rank == 0);
    auto zeroI = createIndex(loc, rewriter, 0);
    ::imex::ValVec shapeVVec(rank, zeroI);
    // create new tensor
    auto zero = createInt(loc, rewriter, 0);
    auto tensor = createEmptyTensor(rewriter, loc, sElTyp, shapeVVec);
    auto tnsr = rewriter.create<::mlir::linalg::FillOp>(loc, zero, tensor);

    // rank/num-dims of input
    auto inpRank = static_cast<unsigned>(inpTnsrTyp.getRank());
    // input maps are identity maps
    auto inpMap = ::mlir::AffineMap::getMultiDimIdentityMap(
        inpRank, rewriter.getContext());
    // output map is "*->()"
    auto omap = ::mlir::AffineMap::get(inpRank, 0, rewriter.getContext());
    const ::mlir::AffineMap maps[] = {inpMap, omap};
    ::mlir::SmallVector<mlir::utils::IteratorType> iterators(
        inpRank, mlir::utils::IteratorType::reduction);

    // create reduction op as linalg::generic
    const ::imex::ndarray::ReduceOpId ropid =
        (::imex::ndarray::ReduceOpId)mlir::cast<::mlir::IntegerAttr>(
            adaptor.getOp())
            .getInt();
    auto bodyBuilder = getBodyBuilder(ropid, sElTyp);
    auto resTnsr = rewriter.create<::mlir::linalg::GenericOp>(
        loc, tnsr.getType(0), oprnds, tnsr.getResult(0), maps, iterators,
        bodyBuilder);
    rewriter.replaceOp(op, resTnsr.getResult(0));

    return ::mlir::success();
  }
};

/// Convert NDArray's permute_dims operations and their return type to
/// Linalg/tensor.
struct PermuteDimsOpLowering
    : public ::mlir::OpConversionPattern<::imex::ndarray::PermuteDimsOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::PermuteDimsOp op,
                  ::imex::ndarray::PermuteDimsOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto loc = op->getLoc();
    auto srcTnsr = adaptor.getSource();

    // convert src array to memref
    auto srcArType = mlir::dyn_cast_or_null<::imex::ndarray::NDArrayType>(
        op.getSource().getType());
    if (!srcArType)
      return mlir::failure();
    auto srcMRType = srcArType.getMemRefType(srcTnsr);
    auto srcMR = createToMemRef(loc, rewriter, srcTnsr, srcMRType, true);

    auto perm = ::mlir::AffineMapAttr::get(::mlir::AffineMap::getPermutationMap(
        adaptor.getAxes(), rewriter.getContext()));
    mlir::memref::TransposeOp transposeOp =
        rewriter.create<mlir::memref::TransposeOp>(loc, srcMR, perm);

    rewriter.replaceOp(op, transposeOp.getResult());

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
    ::mlir::TypeConverter typeConverter;
    // Convert unknown types to itself
    auto convT2T = [](::mlir::Type type) { return type; };
    // Convert NDArrayType to (tensorType)
    auto convNDArray2RankedTensor =
        [](::imex::ndarray::NDArrayType type) -> std::optional<::mlir::Type> {
      return type.getTensorType();
    };

    typeConverter.addConversion(convT2T);
    typeConverter.addConversion(convNDArray2RankedTensor);

    auto materializeCast =
        [](::mlir::OpBuilder &builder, ::mlir::Type type,
           ::mlir::ValueRange inputs,
           ::mlir::Location loc) -> ::mlir::Value {
      if (inputs.size() == 1) {
        auto input = inputs[0];
        auto itype = input.getType();
        if (mlir::isa<::mlir::TensorType>(type) and
            mlir::isa<::mlir::TensorType>(itype)) {
          return builder.create<::mlir::tensor::CastOp>(loc, type, inputs)
              .getResult();
        }
        auto ttype = mlir::dyn_cast<::mlir::RankedTensorType>(itype);
        if (ttype && mlir::isa<::mlir::MemRefType>(type)) {
          return createToMemRef(loc, builder, input, type);
        }
        auto mrtype = mlir::dyn_cast<::mlir::MemRefType>(itype);
        if (mrtype && mlir::isa<::mlir::RankedTensorType>(type)) {
          return builder
              .create<::mlir::bufferization::ToTensorOp>(loc, type, input,
                                                         /*restrict=*/true)
              .getResult();
        }
      }
      return builder
          .create<::mlir::UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };
    typeConverter.addSourceMaterialization(materializeCast);
    typeConverter.addTargetMaterialization(materializeCast);

    // At function boundaries we have actual memref semantics.
    // We need to explicitly convert in/out arguments to memrefs.
    // If we use tensors downstream passes will auto-convert to non-strided
    // memrefs which will imply a copy (converting from strided to non-strided
    // requires a copy)
    // We simply use a separate type-converter and materializations

    ::mlir::TypeConverter typeConverter4Func;
    // Convert NDArrayType to MemRefType
    auto convNDArray2MemRef =
        [](::imex::ndarray::NDArrayType type) -> std::optional<::mlir::Type> {
      return type.getMemRefType();
    };

    typeConverter4Func.addConversion(convT2T);
    typeConverter4Func.addConversion(convNDArray2MemRef);
    typeConverter4Func.addSourceMaterialization(materializeCast);
    typeConverter4Func.addTargetMaterialization(materializeCast);

    ::mlir::ConversionTarget target(ctxt);
    // We convert all NDArray stuff...
    target.addIllegalDialect<::imex::ndarray::NDArrayDialect>();
    // ...into Linalg, Affine, Tensor, Arith
    target.addLegalDialect<
        ::mlir::linalg::LinalgDialect, ::mlir::affine::AffineDialect,
        ::mlir::arith::ArithDialect, ::mlir::math::MathDialect,
        ::mlir::memref::MemRefDialect, ::mlir::tensor::TensorDialect,
        ::mlir::tosa::TosaDialect, ::mlir::shape::ShapeDialect,
        ::mlir::bufferization::BufferizationDialect,
        ::imex::region::RegionDialect>();
    target.addLegalOp<::mlir::UnrealizedConversionCastOp>(); // FIXME

    // make sure function boundaries use tensors (not NDArrays)
    target.addDynamicallyLegalOp<::mlir::func::FuncOp>(
        [&](::mlir::func::FuncOp op) {
          return typeConverter4Func.isSignatureLegal(op.getFunctionType()) &&
                 typeConverter4Func.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<::mlir::func::ReturnOp, mlir::func::CallOp>(
        [&](mlir::Operation *op) { return typeConverter4Func.isLegal(op); });

    target.addDynamicallyLegalOp<::imex::region::EnvironmentRegionOp,
                                 ::imex::region::EnvironmentRegionYieldOp>(
        [&](mlir::Operation *op) { return typeConverter.isLegal(op); });

    ::mlir::RewritePatternSet patterns(&ctxt);
    patterns.insert<ToTensorLowering, SubviewLowering, ExtractSliceLowering,
                    InsertSliceLowering, ImmutableInsertSliceLowering,
                    LinSpaceLowering, LoadOpLowering, CreateLowering,
                    EWBinOpLowering, DimOpLowering, EWUnyOpLowering,
                    ReductionOpLowering, ReshapeLowering, CastLowering,
                    CopyLowering, DeleteLowering, CastElemTypeLowering,
                    FromMemRefLowering, PermuteDimsOpLowering>(typeConverter,
                                                               &ctxt);
    ::imex::populateRegionTypeConversionPatterns(patterns, typeConverter);

    // populate function boundaries using our special type converter
    ::mlir::populateFunctionOpInterfaceTypeConversionPattern<
        ::mlir::func::FuncOp>(patterns, typeConverter4Func);
    ::mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter4Func);
    ::mlir::populateCallOpTypeConversionPattern(patterns, typeConverter4Func);

    ::mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        typeConverter, patterns, target);

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
