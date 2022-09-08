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
/// dialect to the Linalg and Dist dialects.
///
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include <imex/Conversion/PTensorToLinalg/PTensorToLinalg.h>
#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <iostream>

namespace imex {

/// @return type without a sign
static mlir::Type makeSignlessType(mlir::Type type) {
  if (auto retRtTyp = type.dyn_cast<::mlir::ShapedType>()) {
    auto origElemType = retRtTyp.getElementType();
    return makeSignlessType(origElemType);
  } else if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
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

/// Initialize a distributed Tensor:
/// 1. register tensor with runtime
/// 2. get local shape
/// 3. init local tensor
/// @return pair of tensor and id as assigned by runtime
/// If not distributed, simply init tensor.
static auto
initDTensor(mlir::Location &loc, ::mlir::ConversionPatternRewriter &rewriter,
            bool dist, uint64_t rank,
            llvm::SmallVector<mlir::Value> shapeVVec_vals,
            ::mlir::Value shapeVVec_tnsr, ::mlir::Type eltyp,
            ::llvm::SmallVector<mlir::Value> &lShapeVVec /* out */) {
  if (dist) {
    auto intTyp = rewriter.getI64Type();
    auto idxTyp = rewriter.getIndexType();
    auto shapeVVectyp =
        mlir::RankedTensorType::get(llvm::SmallVector<int64_t>(rank), idxTyp);

    // Register with runtime
    ::mlir::Value id = rewriter.create<::imex::dist::RegisterPTensorOp>(
        loc, intTyp, shapeVVec_tnsr);
    // and get local shape
    auto lShapeVVec_mr =
        rewriter.create<::imex::dist::LocalShapeOp>(loc, shapeVVectyp, id);

    // get shape as SmallVector<mlir::Value>
    // why can't we just use the existing tensor?
    lShapeVVec.resize(rank);
    for (auto i : ::llvm::seq(0LU, rank)) {
      auto idx = rewriter.create<::mlir::arith::ConstantOp>(
          loc, rewriter.getIndexAttr(i));
      lShapeVVec[i] = rewriter.create<::mlir::tensor::ExtractOp>(
          loc, idxTyp, lShapeVVec_mr, ::mlir::ValueRange({idx}));
    }
    // create a 1d tensor of local shape
    auto lTnsr =
        rewriter.create<::mlir::linalg::InitTensorOp>(loc, lShapeVVec, eltyp);
    return std::make_pair(lTnsr.getResult(), id);
  } else { // not distributed, simply init
    auto lTnsr = rewriter.create<::mlir::linalg::InitTensorOp>(
        loc, shapeVVec_vals, eltyp);
    return std::make_pair(lTnsr.getResult(), ::mlir::Value());
  }
}

// *******************************
// ***** Individual patterns *****
// *******************************

namespace {
/// Lower MkPTensorOp into a UnrealizedConversionCastOp, using the type
/// converter to determine the target type. Operations extracting members
/// (rtensor, device etc) are expected to chase the tuple creation back to here
/// and get the respective operand of the cast.
// FIXME Is there a better/clener way to do this?
struct MkPTensorLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::MkPTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::MkPTensorOp op,
                  ::imex::ptensor::MkPTensorOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = *getTypeConverter();
    (void)rewriter.replaceOpWithNewOp<::mlir::UnrealizedConversionCastOp>(
        op, converter.convertType(op.getType()), adaptor.getOperands());
    return ::mlir::success();
  }
};

/// Lower to the input operand of the defining op. We assume this to ultimately
/// be the UnrealizedConversionCast created by MkPTensorLowering.
struct ExtractRTensorLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::ExtractRTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ExtractRTensorOp op,
                  ::imex::ptensor::ExtractRTensorOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto inpOp =
        adaptor.input().getDefiningOp<::mlir::UnrealizedConversionCastOp>();
    // This can be a chain of casts, originating from type conversion like type
    // materialization for function arguments. This requires chasing the chain
    // of casts. We cannot chase casts with more than one operand without
    // getting into realms of unclear semantics.
    while (inpOp && inpOp.getOperands().size() == 1 &&
           !inpOp.getOperands()
                .front()
                .getType()
                .isa<::mlir::RankedTensorType>()) {
      inpOp = inpOp.getOperands()
                  .front()
                  .getDefiningOp<::mlir::UnrealizedConversionCastOp>();
    }
    assert(inpOp);
    assert(inpOp.getOperands().size() == 4);
    assert(
        inpOp.getOperands().front().getType().isa<::mlir::RankedTensorType>());
    rewriter.replaceOp(op, inpOp.getOperands()[0]);
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
    auto converter = *getTypeConverter();

    // Get Operands
    auto start = adaptor.start();
    auto stop = adaptor.stop();
    auto step = adaptor.step();
    auto retPtTyp = op.getType().dyn_cast<::imex::ptensor::PTensorType>();
    assert(retPtTyp);

    // we operate on signless integers
    auto intTyp = rewriter.getI64Type();
    if (start.getType() != intTyp) {
      start =
          rewriter
              .create<::mlir::UnrealizedConversionCastOp>(loc, intTyp, start)
              .getResult(0);
    }
    if (stop.getType() != intTyp) {
      stop =
          rewriter.create<::mlir::UnrealizedConversionCastOp>(loc, intTyp, stop)
              .getResult(0);
    }
    if (step.getType() != intTyp) {
      step =
          rewriter.create<::mlir::UnrealizedConversionCastOp>(loc, intTyp, step)
              .getResult(0);
    }

    // Create constants 0, 1, -1 for later
    auto zero = createSignlessInt<64>(rewriter, loc, 0);
    auto one = createSignlessInt<64>(rewriter, loc, 1);
    auto mone = createSignlessInt<64>(rewriter, loc, -1);

    // Compute number of elements as
    //   (stop - start + step + (step < 0 ? 1 : -1)) / step
    auto cond = rewriter.create<mlir::arith::CmpIOp>(
        loc, ::mlir::arith::CmpIPredicate::ult, step, zero);
    auto increment =
        rewriter.create<mlir::arith::SelectOp>(loc, cond, one, mone);
    auto tmp1 = rewriter.create<mlir::arith::AddIOp>(loc, stop, step);
    auto tmp2 = rewriter.create<mlir::arith::AddIOp>(loc, tmp1, increment);
    auto tmp3 = rewriter.create<mlir::arith::SubIOp>(loc, tmp2, start);
    auto count =
        rewriter.create<mlir::arith::DivUIOp>(loc, tmp3, step).getResult();
    count = rewriter
                .create<::mlir::arith::IndexCastOp>(
                    loc, rewriter.getIndexType(), count)
                .getResult();

    // create shape vector
    auto retRtTyp = retPtTyp.getRtensor();
    assert(retRtTyp);
    auto elTyp = retRtTyp.getElementType();
    llvm::SmallVector<mlir::Value> shapeVVec(1, count);

    // register and init tensor
    llvm::SmallVector<mlir::Value> lShapeVVec(1);
    auto tmpTnsr = rewriter.create<::mlir::linalg::InitTensorOp>(
        loc, ::mlir::ValueRange({count}), elTyp);
    auto shape = rewriter.create<::mlir::shape::ShapeOfOp>(loc, tmpTnsr);
    bool isDist = retPtTyp.getDist();
    auto tensorId = initDTensor(loc, rewriter, isDist, 1, shapeVVec, shape,
                                elTyp, lShapeVVec);

    // compute start index of local partition
    if (isDist) {
      auto offTyp = rewriter.getIndexType();
      auto offsets = rewriter.create<::imex::dist::LocalOffsetsOp>(
          loc, offTyp, tensorId.second);
      // auto _off = rewriter.create<::mlir::memref::DimOp>(loc, offsets, 0 );
      auto off =
          rewriter.create<::mlir::arith::IndexCastOp>(loc, intTyp, offsets);
      auto tmp = // off * step
          rewriter.create<::mlir::arith::MulIOp>(loc, off, step);
      start = // start + (off * stride)
          rewriter.create<::mlir::arith::AddIOp>(loc, start, tmp);
    }

    // fill with arange values
    // map needed for output only (we have no input tensor)
    const ::mlir::AffineMap maps[] = {
        ::mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext())};
    llvm::SmallVector<::mlir::StringRef> iterators(1, "parallel");

    // The body; accepting no input, the lambda simply captures start and step
    auto body = [&start, &step, &elTyp, &intTyp](::mlir::OpBuilder &builder,
                                                 ::mlir::Location loc,
                                                 ::mlir::ValueRange args) {
      auto dim = builder.getI64IntegerAttr(0);
      auto idx = builder.create<::mlir::linalg::IndexOp>(loc, dim);
      auto _idx = builder.create<::mlir::arith::IndexCastOp>(loc, intTyp, idx);
      auto tmp = builder.create<::mlir::arith::MulIOp>(loc, step, _idx);
      auto val = builder.create<::mlir::arith::AddIOp>(loc, start, tmp);
      auto ret = builder.create<::mlir::UnrealizedConversionCastOp>(
          loc, elTyp, val.getResult());
      // auto _val = builder.create<mlir::arith::SIToFPOp>(loc, elTyp, val);
      (void)builder.create<::mlir::linalg::YieldOp>(loc, ret.getResult(0));
    };

    auto arange = rewriter.create<::mlir::linalg::GenericOp>(
        loc, retRtTyp, ::llvm::None, tensorId.first, maps, iterators, body);
    (void)rewriter.replaceOpWithNewOp<::imex::ptensor::MkPTensorOp>(
        op, arange.getResult(0));
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
        assert("Found integer type but binary op not defined for integers" ==
               nullptr);
    } else if (lhsTyp.isIntOrIndexOrFloat()) {
      if constexpr (!std::is_same_v<FOP, void>) {
        yield(builder, loc, typ,
              builder.create<FOP>(loc, args[0], args[1]).getResult());
        return;
      } else
        assert("Found float type but binary op not defined for floats" ==
               nullptr);
    } else {
      assert("Only integers and floats supported for binary ops" == nullptr);
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
    assert("unsupported elementwise binary operation" == nullptr);
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
    auto lhsPtTyp = op.lhs().getType().dyn_cast<::imex::ptensor::PTensorType>();
    auto rhsPtTyp = op.rhs().getType().dyn_cast<::imex::ptensor::PTensorType>();
    if (!lhsPtTyp || !rhsPtTyp) {
      // fail if not, will be retried if operands get converted elsewhere
      return ::mlir::failure();
    }
    auto converter = *getTypeConverter();
    // we expect RankedTensorType as operands
    auto lhsRtTyp = lhsPtTyp.getRtensor();
    auto rhsRtTyp = rhsPtTyp.getRtensor();
    // input tensors might have compatible but different types
    assert(adaptor.lhs().getType() == adaptor.rhs().getType());

    // the element type of a binop depends on the input arguments and the
    // operation itself we assume this had beeen taken care of and simply use
    // the op's converted type
    auto retPtTyp =
        op.getResult().getType().dyn_cast<::imex::ptensor::PTensorType>();
    assert(retPtTyp);
    auto retRtTyp = retPtTyp.getRtensor();
    auto elTyp = retRtTyp.getElementType();

    // get the input as RankedTensors
    auto lhsTnsr = rewriter.create<::imex::ptensor::ExtractRTensorOp>(
        loc, lhsRtTyp, op.lhs());
    auto rhsTnsr = rewriter.create<::imex::ptensor::ExtractRTensorOp>(
        loc, rhsRtTyp, op.rhs());

    // build tensor using the resulting element type
    // the shape is not statically known, we need to retrieve it (it's the same
    // as the input shapes)
    // FIXME shape broadcasting: input tensors might have compatible but
    // different shapes
    auto rank = static_cast<unsigned>(retRtTyp.getRank());
    llvm::SmallVector<mlir::Value> shapeVVec(rank);
    llvm::SmallVector<mlir::StringRef> iterators(rank);
    for (auto i : llvm::seq(0u, rank)) {
      shapeVVec[i] = rewriter.create<::mlir::tensor::DimOp>(loc, lhsTnsr, i);
      // iterate in parallel
      iterators[i] = "parallel";
    }

    // register and init tensor
    llvm::SmallVector<mlir::Value> lShapeVVec(rank);
    auto tmpTnsr =
        rewriter.create<::mlir::linalg::InitTensorOp>(loc, shapeVVec, elTyp);
    auto shape = rewriter.create<::mlir::shape::ShapeOfOp>(loc, tmpTnsr);
    auto tensorId = initDTensor(loc, rewriter, lhsPtTyp.getDist(), rank,
                                shapeVVec, shape, elTyp, lShapeVVec);

    // Get signless operands into vec
    llvm::SmallVector<mlir::Value, 2> oprnds = {lhsTnsr, rhsTnsr};

    // all maps are identity maps
    auto inpMap =
        ::mlir::AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    const ::mlir::AffineMap maps[] = {inpMap, inpMap, inpMap};

    // create binop as linalg::generic
    const ::imex::ptensor::EWBinOpId binOpId =
        (::imex::ptensor::EWBinOpId)adaptor.op()
            .cast<::mlir::IntegerAttr>()
            .getInt();
    auto bodyBuilder = getBodyBuilder(binOpId, elTyp);
    auto retTnsr = rewriter
                       .create<::mlir::linalg::GenericOp>(
                           loc, tensorId.first.getType(), oprnds,
                           tensorId.first, maps, iterators, bodyBuilder)
                       .getResult(0);

    rewriter.replaceOpWithNewOp<::imex::ptensor::MkPTensorOp>(op, retTnsr);
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
    assert("unsupported reduction operation" == nullptr);
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
        op.input().getType().dyn_cast<::imex::ptensor::PTensorType>();
    if (!inpPtTyp) {
      // fail if not, will be retried if operands get converted elsewhere
      return ::mlir::failure();
    }

    auto converter = *getTypeConverter();
    // we expect RankedTensorType as operands
    auto inpRtTyp = inpPtTyp.getRtensor();
    auto inpTnsr = rewriter.create<::imex::ptensor::ExtractRTensorOp>(
        loc, inpRtTyp, op.input());

    // Get signless operands into vec
    llvm::SmallVector<mlir::Value, 1> oprnds = {inpTnsr};

    // determine resulting element type from converted op-type
    auto retPtTyp =
        op.getResult().getType().dyn_cast<::imex::ptensor::PTensorType>();
    assert(retPtTyp);
    auto retRtTyp = retPtTyp.getRtensor();
    auto elTyp = retRtTyp.getElementType();
    auto sElTyp = makeSignlessType(elTyp);

    // build tensor using the resulting element type and shape
    // FIXME support reduction dimensions
    auto rank = static_cast<unsigned>(retRtTyp.getRank());
    assert(rank == 0);
    llvm::SmallVector<mlir::Value> shapeVVec(
        rank); //::::mlir::ShapedType::kDynamicSize;
    // create new tensor
    auto zero =
        rewriter
            .create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0))
            .getResult();
    llvm::SmallVector<mlir::Value> lShapeVVec(rank);
    auto tmpTnsr =
        rewriter.create<::mlir::linalg::InitTensorOp>(loc, shapeVVec, elTyp);
    auto shape = rewriter.create<::mlir::shape::ShapeOfOp>(loc, tmpTnsr);
    bool isDist = inpPtTyp.getDist();
    auto tensorId = initDTensor(loc, rewriter, isDist, rank, shapeVVec, shape,
                                sElTyp, lShapeVVec);
    auto tnsr =
        rewriter.create<::mlir::linalg::FillOp>(loc, zero, tensorId.first);

    // rank/num-dims of input
    auto inpRank = static_cast<unsigned>(inpRtTyp.getRank());
    // input maps are identity maps
    auto inpMap = ::mlir::AffineMap::getMultiDimIdentityMap(
        inpRank, rewriter.getContext());
    // output map is "*->()"
    auto omap = ::mlir::AffineMap::get(inpRank, 0, rewriter.getContext());
    const ::mlir::AffineMap maps[] = {inpMap, omap};
    llvm::SmallVector<mlir::StringRef> iterators(inpRank, "reduction");

    // create reduction op as linalg::generic
    const ::imex::ptensor::ReduceOpId ropid =
        (::imex::ptensor::ReduceOpId)adaptor.op()
            .cast<::mlir::IntegerAttr>()
            .getInt();
    auto bodyBuilder = getBodyBuilder(ropid, sElTyp);
    auto retTnsr = rewriter
                       .create<::mlir::linalg::GenericOp>(
                           loc, tnsr.getType(0), oprnds, tnsr.getResult(0),
                           maps, iterators, bodyBuilder)
                       .getResult(0);

    // we reduced the local part, now we reduce across processes
    if (isDist) {
      retTnsr = rewriter
                    .create<::imex::dist::AllReduceOp>(loc, tnsr.getType(0),
                                                       adaptor.op(), retTnsr)
                    .getResult();
    }

    rewriter.replaceOpWithNewOp<::imex::ptensor::MkPTensorOp>(op, retTnsr);
    return ::mlir::success();
  }
};

/// Convert return operands of type ptensor to multiple operands (rtensor,
/// device, team, handle)
// Don't know how to use :mlir::populateReturnOpTypeConversionPattern because we
// have to pass individual operands (e.g. not one ptensor, but rtensor, device,
// team and handle).
struct ReturnOpConversion
    : public ::mlir::OpConversionPattern<::mlir::func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::func::ReturnOp op,
                  ::mlir::func::ReturnOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto orgOprnd = op.operands().begin();
    auto oldOprnds = adaptor.operands();
    ::mlir::SmallVector<::mlir::Value> newOprnds;
    for (auto o : oldOprnds) {
      // we assume the number of operands is identical in op and adaptor
      // -> we can check for original ptensor type which is safer than TupleType
      // in adaptor
      if ((*orgOprnd).getType().isa<::imex::ptensor::PTensorType>()) {
        auto defOp = o.getDefiningOp<::mlir::UnrealizedConversionCastOp>();
        assert(defOp);
        // ptensor operands get expanded
        for (auto i : defOp.getInputs()) {
          newOprnds.push_back(i);
        }
      } else {
        // all other types get added as-is
        newOprnds.push_back(o);
      }
      ++orgOprnd; // explicitly move iterator for orig type
    }
    rewriter.replaceOpWithNewOp<::mlir::func::ReturnOp>(op, newOprnds);
    return ::mlir::success();
  }
};

/// Convert callops returning PTensors to a callop followed by a MkPTensorOp.
/// Other result types and operands are converted using the provioded
/// typeConverter. Currently only calls to functions with a single result are
/// supported.
struct CallOpConversion
    : public ::mlir::OpConversionPattern<::mlir::func::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::func::CallOp op,
                  ::mlir::func::CallOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = *getTypeConverter();
    auto funcTyp = op.getCalleeType(); // result types are defined by the callee
    assert(funcTyp.getNumResults() == 1);
    auto orgResTyp = funcTyp.getResult(0);
    ::mlir::SmallVector<::mlir::Type> convResTyps;
    if (::mlir::failed(converter.convertType(orgResTyp, convResTyps))) {
      return ::mlir::failure();
    }
    auto resPTTyp = orgResTyp.dyn_cast<::imex::ptensor::PTensorType>();
    if (resPTTyp) {
      auto convFunc = rewriter.create<::mlir::func::CallOp>(
          op.getLoc(), adaptor.getCallee(), ::mlir::TypeRange(convResTyps),
          adaptor.operands());
      // if return type is a ptensor, we need to convert multiple return vals
      // back into a ptensor which should fold into a tuple
      rewriter.replaceOpWithNewOp<::imex::ptensor::MkPTensorOp>(
          op, resPTTyp.getOnDevice(), resPTTyp.getDist(), convFunc.getResult(0),
          convFunc.getResult(1), convFunc.getResult(2), convFunc.getResult(3));
    } else {
      // only type-conversion needed if the return type is not a ptensor
      rewriter.replaceOpWithNewOp<::mlir::func::CallOp>(
          op, adaptor.getCallee(), ::mlir::TypeRange(convResTyps),
          adaptor.operands());
    }

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
    // Convert PTensorType to (RankedTensorType, device, team, handle)
    auto convPT2Tuple = [&ctxt](::imex::ptensor::PTensorType type)
        -> ::mlir::Optional<::mlir::Type> {
      return ::mlir::TupleType::get(
          &ctxt, {type.getRtensor(), ::mlir::IntegerType::get(&ctxt, 1),
                  ::mlir::IntegerType::get(&ctxt, 1),
                  ::mlir::IntegerType::get(&ctxt, 1)});
    };
    auto convPT2Multiple =
        [&ctxt](::imex::ptensor::PTensorType type,
                ::mlir::SmallVectorImpl<::mlir::Type> &target)
        -> ::mlir::Optional<::mlir::LogicalResult> {
      //  for(auto t : type.getTypes()) target.push_back(t);
      target = ::mlir::SmallVector<::mlir::Type>(
          {type.getRtensor(), ::mlir::IntegerType::get(&ctxt, 1),
           ::mlir::IntegerType::get(&ctxt, 1),
           ::mlir::IntegerType::get(&ctxt, 1)});
      return ::mlir::success();
    };
    typeConverter.addConversion(convT2T);
    typeConverter.addConversion(convPT2Tuple);
    // typeConverter.addConversion(convPT2Multiple);

    auto materializeCast =
        [](::mlir::OpBuilder &builder, ::mlir::Type type,
           ::mlir::ValueRange inputs,
           ::mlir::Location loc) -> ::llvm::Optional<::mlir::Value> {
      return builder
          .create<::mlir::UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };
    typeConverter.addSourceMaterialization(materializeCast);

    // We convert all PTensor stuff...
    target.addIllegalDialect<::imex::ptensor::PTensorDialect>();
    // ...into Linalg, Affine, Tensor, Arith, Dist
    target.addLegalDialect<::imex::dist::DistDialect>();
    target.addLegalDialect<::mlir::linalg::LinalgDialect>();
    target.addLegalDialect<::mlir::AffineDialect>();
    target.addLegalDialect<::mlir::tensor::TensorDialect>();
    target.addLegalDialect<::mlir::arith::ArithmeticDialect>();
    target.addLegalDialect<::mlir::shape::ShapeDialect>();
    target.addLegalOp<::mlir::UnrealizedConversionCastOp>(); // FIXME
    target.addDynamicallyLegalOp<::mlir::func::FuncOp>(
        [&](::mlir::func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                 typeConverter.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<::mlir::func::ReturnOp>(
        [&](::mlir::func::ReturnOp op) {
          for (auto o : op.operands()) {
            if (o.getType().isa<::mlir::TupleType>())
              return false;
          }
          return typeConverter.isLegal(op);
        });

    target.addDynamicallyLegalOp<mlir::func::CallOp>(
        [&](mlir::func::CallOp op) { return typeConverter.isLegal(op); });

    ::mlir::RewritePatternSet patterns(&ctxt);
    patterns.insert<MkPTensorLowering, ExtractRTensorLowering, ARangeLowering,
                    EWBinOpLowering, ReductionOpLowering, ReturnOpConversion>(
        typeConverter, &ctxt);

    // We use a separate type converter for converting type signatures
    // because we do not want tuples in the signature and use separate args
    // for the ptensor members rtensor, device, team, and handle.
    ::mlir::TypeConverter tc2;
    tc2.addConversion(convT2T);
    tc2.addConversion(convPT2Multiple);
    tc2.addArgumentMaterialization(materializeCast);
    ::mlir::populateFunctionOpInterfaceTypeConversionPattern<
        ::mlir::func::FuncOp>(patterns, tc2);
    // ::mlir::populateCallOpTypeConversionPattern(patterns, tc2);
    patterns.insert<CallOpConversion>(tc2, &ctxt);

    // no :mlir::populateReturnOpTypeConversionPattern but above
    // ReturnOpConversion

    if (::mlir::failed(::mlir::applyPartialConversion(getOperation(), target,
                                                      ::std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertPTensorToLinalgPass() {
  return std::make_unique<ConvertPTensorToLinalgPass>();
}

} // namespace imex
