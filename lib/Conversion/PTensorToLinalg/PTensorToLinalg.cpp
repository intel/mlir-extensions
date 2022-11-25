//===- PTensorToLinalg.cpp - PTensorToLinalg conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the PTensorToLinalg conversion, converting the PTensor
/// dialect to the Linalg and helper dialects.
///
/// Any tensor of PTensorType is expected to be initialized by MkPTensorOp.
/// Lowering a MkPtensorOp results in a unrealized_conversion_cast. After
/// complete conversion the resulting value should have no use. However, during
/// conversion its operands will serve for extracting the members (such as
/// ExtractMemRefOp): we chase the unrealized_conversion_cast as the rooting op
/// and return the corresponding operand.
///
/// Currently we do not support propagating device data across function
/// boundaries.
///
/// FIXME: same for device by adding regions.
///
/// The pass is based on a ConversionTarget, TypeConverters, legality checks and
/// conversion patterns.
///
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/PTensorToLinalg/PTensorToLinalg.h>
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Dialect/PTensor/Transforms/Utils.h>
#include <imex/Utils/ArithUtils.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
// #include <mlir/Dialect/memref/IR/MemRef.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <iostream>

#include "../PassDetail.h"

namespace imex {

/// @return type without a sign
static mlir::Type makeSignlessType(mlir::Type type) {
  if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
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

// *******************************
// ***** Individual patterns *****
// *******************************

namespace {

/// Lower MkPTensorOp into a UnrealizedConversionCastOp, using the type
/// converter to determine the target type. Operations extracting members
/// (tensor, device etc) are expected to chase the tuple creation back to here
/// and get the respective operand of the cast.
// FIXME Is there a better/cleaner way to do this?
// FIXME Right now we simply convert to the tensor, we need proper function
// boundary handling
struct MkPTensorLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::MkPTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::MkPTensorOp op,
                  ::imex::ptensor::MkPTensorOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // auto & converter = *getTypeConverter();
    // (void)rewriter.replaceOpWithNewOp<::mlir::UnrealizedConversionCastOp>(
    // op, converter.conveMRType(op.getType()), adaptor.getOperands());
    rewriter.replaceOp(op, adaptor.getTensor());
    return ::mlir::success();
  }
};

/// Lower to the input operand of the defining op. We assume this to ultimately
/// be the UnrealizedConversionCast created by MkPTensorLowering.
struct ExtractTensorLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::ExtractMemRefOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ExtractMemRefOp op,
                  ::imex::ptensor::ExtractMemRefOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // std::cerr << "op: "; op.dump(); std::cerr << std::endl;
    // std::cerr << "oinput: "; op.getInput().dump(); std::cerr << std::endl;
    // std::cerr << "ainput: "; adaptor.getInput().dump(); std::cerr <<
    // std::endl; std::cerr << "odefop: "; if(op.getInput().getDefiningOp())
    // op.getInput().getDefiningOp()->dump(); std::cerr << std::endl; std::cerr
    // << "adefop: "; if(adaptor.getInput().getDefiningOp())
    // adaptor.getInput().getDefiningOp()->dump(); std::cerr << std::endl;
    auto inpOp =
        adaptor.getInput().getDefiningOp<::mlir::UnrealizedConversionCastOp>();
    if (!inpOp) { // block arg or similar
      rewriter.replaceOp(op, adaptor.getInput());
    } else {
      // This can be a chain of casts, originating from type conversion like
      // type materialization for function arguments. This requires chasing the
      // chain of casts. We cannot chase casts with more than one operand
      // without getting into realms of unclear semantics.
      while (inpOp && inpOp.getOperands().size() == 1 &&
             inpOp.getOperands().front().getType().isa<::mlir::MemRefType>()) {
        inpOp = inpOp.getOperands()
                    .front()
                    .getDefiningOp<::mlir::UnrealizedConversionCastOp>();
      }
      assert(inpOp);
      // assert(inpOp.getOperands().size() == 4);
      assert(inpOp.getOperands().front().getType().isa<::mlir::MemRefType>());
      // std::cerr << "repl: "; inpOp.dump(); std::cerr << " ";
      // inpOp.getOperands()[0].dump(); std::cerr << std::endl;
      rewriter.replaceOp(op, inpOp.getOperands()[0]);
    }
    return ::mlir::success();
  }
};

/// Convert PTensor's extract_slice to tensor::extract_slice.
struct ExtractSliceLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::ExtractSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ExtractSliceOp op,
                  ::imex::ptensor::ExtractSliceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // auto loc = op.getLoc();

    // source and result are expected to be of PTensorType
    auto srcTnsrTyp =
        adaptor.getSource().getType().dyn_cast<::mlir::MemRefType>();
    // op.getSource().getType().dyn_cast<::imex::ptensor::PTensorType>();
    if (!srcTnsrTyp)
      return ::mlir::failure();
    // auto srcTnsrTyp = srcPtTyp.getMemRefType();
    // auto dstTnsrTyp =
    // op.getDestination().getType().dyn_cast<::mlir::MemRefType>(); if
    // (!dstTnsrTyp) return ::mlir::failure();
    // auto source = adaptor.getSource();
    // rewriter.create<::imex::ptensor::ExtractMemRefOp>(
    // loc, srcTnsrTyp, op.getSource());
    // and replace with tensor::extract_slice
    // auto viewTensorType =
    // ::mlir::tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
    // retPtTyp.getRank(), srcTnsrTyp, offsets, sizes, strides);
    // mlir::Value view = builder.create<mlir::tensor::ExtractSliceOp>(
    // loc, viewTensorType, srcTensor, offsets, sizes, strides);
    rewriter.replaceOpWithNewOp<::mlir::memref::SubViewOp>(
        op, adaptor.getSource(), adaptor.getOffsets(), adaptor.getSizes(),
        adaptor.getStrides());

    return ::mlir::success();
  }
};

/// Convert PTensor's insert_slice to memref
struct InsertSliceLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::InsertSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::InsertSliceOp op,
                  ::imex::ptensor::InsertSliceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // get operators
    auto src = adaptor.getSource();
    auto dst = adaptor.getDestination();
    // source and result are expected to be of MemRefType
    auto srcMRTyp = src.getType().dyn_cast<::mlir::MemRefType>();
    auto dstMRTyp = dst.getType().dyn_cast<::mlir::MemRefType>();
    if (!dstMRTyp || !srcMRTyp)
      return ::mlir::failure();

    auto view = rewriter.create<::mlir::memref::SubViewOp>(
        loc, dst, adaptor.getOffsets(), adaptor.getSizes(),
        adaptor.getStrides());
    // auto viewTnsrTyp = view.getType().dyn_cast<::mlir::MemRefType>();

    // need memrefs to copy
    // linalg::makeMemRefCopyOp(rewriter, loc, src, dst);
    // auto srcMRTyp = ::mlir::MemRefType::get(srcTnsrTyp.getShape(),
    // srcTnsrTyp.getElementType()); auto viewMRTyp =
    // ::mlir::MemRefType::get(viewTnsrTyp.getShape(),
    // viewTnsrTyp.getElementType()); auto srcMR =
    // rewriter.create<::mlir::bufferization::ToMemrefOp>(loc, srcMRTyp, src);
    // auto viewMR = rewriter.create<::mlir::bufferization::ToMemrefOp>(loc,
    // viewMRTyp, view);
    rewriter.replaceOp(
        op, ::mlir::linalg::makeMemRefCopyOp(rewriter, loc, src, view)
                .getResults());
    return ::mlir::success();
  }
};

/// Convert PTensor's arange and its return type to Linalg/tensor.
/// Also needs some arith and affine (for linalg::genericop).
struct ARangeLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::ARangeOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ARangeOp op,
                  ::imex::ptensor::ARangeOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Get Operands
    auto start = ::imex::EasyInt(loc, rewriter, adaptor.getStart(), true);
    auto stop = ::imex::EasyInt(loc, rewriter, adaptor.getStop(), true);
    auto step = ::imex::EasyInt(loc, rewriter, adaptor.getStep(), true);
    auto retPtTyp = op.getType().dyn_cast<::imex::ptensor::PTensorType>();
    assert(retPtTyp);

    // get arange count
    auto count = createCountARange(rewriter, loc, start, stop, step);

    // init tensor
    auto elTyp = retPtTyp.getElementType();
    auto tensor =
        createEmptyTensor(rewriter, loc, elTyp, {count.get()}).getResult();

    // fill with arange values
    // map needed for output only (we have no input tensor)
    const ::mlir::AffineMap maps[] = {
        ::mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext())};
    llvm::SmallVector<::mlir::StringRef> iterators(1, "parallel");

    // The body; accepting no input, the lambda simply captures start and step
    auto body = [&start, &step, &elTyp](::mlir::OpBuilder &builder,
                                        ::mlir::Location loc,
                                        ::mlir::ValueRange args) {
      auto dim = getIntAttr<64>(builder, 0);
      auto idx = ::imex::EasyInt(
          loc, builder, builder.create<::mlir::linalg::IndexOp>(loc, dim));
      auto val = start + (step * idx);
      // auto _val = builder.create<mlir::arith::SIToFPOp>(loc, elTyp, val);
      (void)builder.create<::mlir::linalg::YieldOp>(
          loc, createIndexCast(loc, builder, val.get(), elTyp));
    };

    auto resTnsr = rewriter.create<::mlir::linalg::GenericOp>(
        loc, retPtTyp.getTensorType(), ::llvm::None, tensor, maps, iterators,
        body);
    rewriter.replaceOpWithNewOp<::mlir::bufferization::ToMemrefOp>(
        op, retPtTyp.getMemRefType(), resTnsr.getResult(0));

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
/// floats. Currently only integers and floats are supported. Currently unsigned
/// int ops are not supported.
template <typename IOP, typename FOP = void>
static BodyType buildTrivial(::mlir::Type typ) {
  return [typ](mlir::OpBuilder &builder, ::mlir::Location loc,
               ::mlir::ValueRange args) -> void {
    auto lhsTyp = args[0].getType();
    if (lhsTyp.isIntOrIndex()) {
      if constexpr (!std::is_same_v<IOP, void>) {
        auto lhs = doSignCast(builder, loc, args[0]);
        auto rhs = doSignCast(builder, loc, args[1]);
        yield(builder, loc, typ,
              builder.create<IOP>(loc, lhs, rhs).getResult());
        return;
      } else
        assert(0 &&
               "Found integer type but binary op not defined for integers");
    } else if (lhsTyp.isIntOrIndexOrFloat()) {
      if constexpr (!std::is_same_v<FOP, void>) {
        yield(builder, loc, typ,
              builder.create<FOP>(loc, args[0], args[1]).getResult());
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
static BodyType getBodyBuilder(::imex::ptensor::EWBinOpId binOp,
                               ::mlir::Type typ) {
  switch (binOp) {
  case ptensor::ADD:
    return buildTrivial<mlir::arith::AddIOp, mlir::arith::AddFOp>(typ);
  // case ptensor::ATAN2] =
  case ptensor::FLOOR_DIVIDE:
    return buildTrivial<mlir::arith::FloorDivSIOp>(typ);
  // case ptensor::LOGADDEXP] =
  // case ptensor::LSHIFT] =
  // case ptensor::MATMUL] =
  case ptensor::MAXIMUM:
    return buildTrivial<mlir::arith::MaxSIOp, mlir::arith::MaxFOp>(typ);
  case ptensor::MINIMUM:
    return buildTrivial<mlir::arith::MinSIOp, mlir::arith::MinFOp>(typ);
  case ptensor::MODULO:
    return buildTrivial<mlir::arith::RemSIOp, mlir::arith::RemFOp>(typ);
  case ptensor::MULTIPLY:
    return buildTrivial<mlir::arith::MulIOp, mlir::arith::MulFOp>(typ);
  // case ptensor::POW] =
  case ptensor::SUBTRACT:
    return buildTrivial<mlir::arith::SubIOp, mlir::arith::SubFOp>(typ);
  // case ptensor::TRUE_DIVIDE] =
  // case ptensor::BITWISE_AND] =
  // case ptensor::BITWISE_LEFT_SHIFT] =
  // case ptensor::BITWISE_OR] =
  // case ptensor::BITWISE_RIGHT_SHIFT] =
  // case ptensor::BITWISE_XOR] =

  // case ptensor::EQUAL] =
  // case ptensor::GREATER] =
  // case ptensor::GREATER_EQUAL] =
  // case ptensor::LESS] =
  // case ptensor::LESS_EQUAL] =
  // case ptensor::LOGICAL_AND] =
  // case ptensor::LOGICAL_OR] =
  // case ptensor::LOGICAL_XOR] =
  // case ptensor::NOT_EQUAL] =
  default:
    assert(0 && "unsupported elementwise binary operation");
  };
}

/// Convert PTensor's elementwise binary operations and their return type to
/// Linalg/tensor. The given op's type is expected to convert to the appropriate
/// type (shape and element-type).
/// Also needs some arith and affine (for linalg::genericop).
struct EWBinOpLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::EWBinOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::EWBinOp op,
                  ::imex::ptensor::EWBinOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // We expect to lower PTensors
    auto lhsPtTyp =
        op.getLhs().getType().dyn_cast<::imex::ptensor::PTensorType>();
    auto rhsPtTyp =
        op.getRhs().getType().dyn_cast<::imex::ptensor::PTensorType>();
    if (!lhsPtTyp || !rhsPtTyp) {
      // fail if not, will be retried if operands get converted elsewhere
      return ::mlir::failure();
    }
    // we expect tensorType as operands
    auto lhsTnsrTyp = lhsPtTyp.getMemRefType();
    auto rhsTnsrTyp = rhsPtTyp.getMemRefType();
    // input tensors might have compatible but different types
    assert(adaptor.getLhs().getType() == adaptor.getRhs().getType());

    // the element type of a binop depends on the input arguments and the
    // operation itself we assume this had beeen taken care of and simply use
    // the op's converted type
    auto retPtTyp =
        op.getResult().getType().dyn_cast<::imex::ptensor::PTensorType>();
    assert(retPtTyp);
    auto retTyp = retPtTyp.getMemRefType();
    auto elTyp = retTyp.getElementType();

    // get the input as tensors
    auto lhsTnsr = rewriter.create<::imex::ptensor::ExtractMemRefOp>(
        loc, lhsTnsrTyp, op.getLhs());
    auto rhsTnsr = rewriter.create<::imex::ptensor::ExtractMemRefOp>(
        loc, rhsTnsrTyp, op.getRhs());

    // build tensor using the resulting element type
    // the shape is not statically known, we need to retrieve it (it's the same
    // as the input shapes)
    // FIXME shape broadcasting: input tensors might have compatible but
    // different shapes
    auto rank = static_cast<unsigned>(retTyp.getRank());
    llvm::SmallVector<::mlir::Value> shapeVVec(rank);
    llvm::SmallVector<mlir::StringRef> iterators(rank);
    for (auto i : llvm::seq<unsigned>(0u, rank)) {
      shapeVVec[i] =
          rewriter.create<::mlir::memref::DimOp>(loc, lhsTnsr, i).getResult();
      // iterate in parallel
      iterators[i] = "parallel";
    }

    // init tensor
    auto tensor =
        createEmptyTensor(rewriter, loc, elTyp, shapeVVec).getResult();

    // Get signless operands into vec
    llvm::SmallVector<mlir::Value, 2> oprnds = {
        rewriter.create<::mlir::bufferization::ToTensorOp>(
            loc, lhsPtTyp.getTensorType(), lhsTnsr),
        rewriter.create<::mlir::bufferization::ToTensorOp>(
            loc, rhsPtTyp.getTensorType(), rhsTnsr)};

    // all maps are identity maps
    auto inpMap =
        ::mlir::AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    const ::mlir::AffineMap maps[] = {inpMap, inpMap, inpMap};

    // create binop as linalg::generic
    const ::imex::ptensor::EWBinOpId binOpId =
        (::imex::ptensor::EWBinOpId)adaptor.getOp()
            .cast<::mlir::IntegerAttr>()
            .getInt();
    auto bodyBuilder = getBodyBuilder(binOpId, elTyp);
    auto resTnsr = rewriter.create<::mlir::linalg::GenericOp>(
        loc, tensor.getType(), oprnds, tensor, maps, iterators, bodyBuilder);
    rewriter.replaceOpWithNewOp<::mlir::bufferization::ToMemrefOp>(
        op, retPtTyp.getMemRefType(), resTnsr.getResult(0));

    return ::mlir::success();
  }
};

// get a body builder for given binary operation and result type
// we accept a result type to insert a cast after the operation if needed
static BodyType getBodyBuilder(::imex::ptensor::ReduceOpId redOp,
                               ::mlir::Type typ) {
  switch (redOp) {
  case ::imex::ptensor::PROD:
    return getBodyBuilder(::imex::ptensor::MULTIPLY, typ);
  case ::imex::ptensor::SUM:
    return getBodyBuilder(::imex::ptensor::ADD, typ);
  case ::imex::ptensor::MAX:
    return getBodyBuilder(::imex::ptensor::MAXIMUM, typ);
  case ::imex::ptensor::MIN:
    return getBodyBuilder(::imex::ptensor::MINIMUM, typ);
  case ::imex::ptensor::MEAN:
  case ::imex::ptensor::STD:
  case ::imex::ptensor::VAR:
  default:
    assert(0 && "unsupported reduction operation");
  };
}

/// Convert PTensor's reduction operations and their return type to
/// Linalg/tensor. The given op's type is expected to convert to the appropriate
/// type (shape and element-type). Also needs some arith and affine (for
/// linalg::genericop).
// FIXME reduction over a subset of dimensionsstruct ReductionOpLowering
struct ReductionOpLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ReductionOp op,
                  ::imex::ptensor::ReductionOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // We expect to lower PTensors
    auto inpPtTyp =
        op.getInput().getType().dyn_cast<::imex::ptensor::PTensorType>();
    if (!inpPtTyp) {
      // fail if not, will be retried if operands get converted elsewhere
      return ::mlir::failure();
    }

    // we expect tensorType as operands
    auto inpTnsrTyp = inpPtTyp.getMemRefType();
    auto inpTnsr = rewriter.create<::imex::ptensor::ExtractMemRefOp>(
        loc, inpTnsrTyp, op.getInput());

    // Get signless operands into vec
    llvm::SmallVector<mlir::Value, 1> oprnds = {
        rewriter.create<::mlir::bufferization::ToTensorOp>(
            loc, inpPtTyp.getTensorType(), inpTnsr)};

    // determine resulting element type from converted op-type
    auto retPtTyp =
        op.getResult().getType().dyn_cast<::imex::ptensor::PTensorType>();
    assert(retPtTyp);
    auto retTyp = retPtTyp.getMemRefType();
    auto elTyp = retTyp.getElementType();
    auto sElTyp = makeSignlessType(elTyp);

    // build tensor using the resulting element type and shape
    // FIXME support reduction dimensions
    auto rank = static_cast<unsigned>(retTyp.getRank());
    assert(rank == 0);
    auto zeroI = createIndex(loc, rewriter, 0);
    llvm::SmallVector<::mlir::Value> shapeVVec(rank, zeroI);
    // create new tensor
    auto zero = createInt(loc, rewriter, 0);
    auto tensor =
        createEmptyTensor(rewriter, loc, sElTyp, shapeVVec).getResult();
    auto tnsr = rewriter.create<::mlir::linalg::FillOp>(loc, zero, tensor);

    // rank/num-dims of input
    auto inpRank = static_cast<unsigned>(inpTnsrTyp.getRank());
    // input maps are identity maps
    auto inpMap = ::mlir::AffineMap::getMultiDimIdentityMap(
        inpRank, rewriter.getContext());
    // output map is "*->()"
    auto omap = ::mlir::AffineMap::get(inpRank, 0, rewriter.getContext());
    const ::mlir::AffineMap maps[] = {inpMap, omap};
    llvm::SmallVector<mlir::StringRef> iterators(inpRank, "reduction");

    // create reduction op as linalg::generic
    const ::imex::ptensor::ReduceOpId ropid =
        (::imex::ptensor::ReduceOpId)adaptor.getOp()
            .cast<::mlir::IntegerAttr>()
            .getInt();
    auto bodyBuilder = getBodyBuilder(ropid, sElTyp);
    auto resTnsr = rewriter.create<::mlir::linalg::GenericOp>(
        loc, tnsr.getType(0), oprnds, tnsr.getResult(0), maps, iterators,
        bodyBuilder);
    rewriter.replaceOpWithNewOp<::mlir::bufferization::ToMemrefOp>(
        op, retPtTyp.getMemRefType(), resTnsr.getResult(0));

    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

/// Convert PTensor to Linalg.
/// After success, no more PTensor should be left, replaced by Linalg & Affine &
/// Arith. Use a type converter to get rid of PTensorType.
struct ConvertPTensorToLinalgPass
    : public ::imex::ConvertPTensorToLinalgBase<ConvertPTensorToLinalgPass> {

  ConvertPTensorToLinalgPass() = default;

  void runOnOperation() override {
    auto &ctxt = getContext();
    ::mlir::ConversionTarget target(ctxt);
    ::mlir::TypeConverter typeConverter;
    // Convert unknown types to itself
    auto convT2T = [](::mlir::Type type) { return type; };
    // Convert PTensorType to (tensorType, device, team, handle)
    auto convPt2Rt = [&ctxt](::imex::ptensor::PTensorType type)
        -> ::mlir::Optional<::mlir::Type> { return type.getMemRefType(); };

    typeConverter.addConversion(convT2T);
    typeConverter.addConversion(convPt2Rt);

    auto materializeCast =
        [](::mlir::OpBuilder &builder, ::mlir::Type type,
           ::mlir::ValueRange inputs,
           ::mlir::Location loc) -> ::llvm::Optional<::mlir::Value> {
      return builder
          .create<::mlir::UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };
    typeConverter.addSourceMaterialization(materializeCast);
    typeConverter.addTargetMaterialization(materializeCast);

    // We convert all PTensor stuff...
    target.addIllegalDialect<::imex::ptensor::PTensorDialect>();
    // ...into Linalg, Affine, Tensor, Arith
    target.addLegalDialect<::mlir::linalg::LinalgDialect>();
    target.addLegalDialect<::mlir::AffineDialect>();
    target.addLegalDialect<::mlir::arith::ArithDialect>();
    target.addLegalDialect<::mlir::memref::MemRefDialect>();
    target.addLegalDialect<::mlir::tensor::TensorDialect>();
    target.addLegalDialect<::mlir::bufferization::BufferizationDialect>();
    target.addLegalOp<::mlir::UnrealizedConversionCastOp>(); // FIXME
    // make sure function boundaries use tensors (not PTensors)
    target.addDynamicallyLegalOp<::mlir::func::FuncOp>(
        [&](::mlir::func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                 typeConverter.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<::mlir::func::ReturnOp>(
        [&](::mlir::func::ReturnOp op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<mlir::func::CallOp>(
        [&](mlir::func::CallOp op) { return typeConverter.isLegal(op); });

    ::mlir::RewritePatternSet patterns(&ctxt);
    patterns.insert<MkPTensorLowering, ExtractTensorLowering,
                    ExtractSliceLowering, InsertSliceLowering, ARangeLowering,
                    EWBinOpLowering, ReductionOpLowering>(typeConverter, &ctxt);
    ::mlir::populateFunctionOpInterfaceTypeConversionPattern<
        ::mlir::func::FuncOp>(patterns, typeConverter);
    ::mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
    ::mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);

    if (::mlir::failed(::mlir::applyPartialConversion(getOperation(), target,
                                                      ::std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

/// Create a pass to convert PTensor to Linalg
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertPTensorToLinalgPass() {
  return std::make_unique<ConvertPTensorToLinalgPass>();
}

} // namespace imex
