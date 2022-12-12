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
/// Any tensor of PTensorType is expected to be initialized by MkPTensorOp.
/// Lowering a MkPtenorTypeOp results in a unrealized_conversion_cast to
/// tuple<RankedTensorType, devicetype, teamtype, handletype>. Since MLIR
/// provides no Ops on tuples, extra Ops are provided to extract the members
/// (such as ExtractRTensorOp). These extraction ops chase
/// unrealized_conversion_cast to find the tuple-defining op and return the
/// corresponding operand.
///
/// In a similar way, RTensorTypes get converted to multiple arguments on
/// function boundaries.
///
/// Ops of the array-API get lowered mostly to Linalg. If input types are
/// distributed (PTEnsorType.getDist()) necessary ops of the Dist dialect are
/// created.
/// FIXME: same for device by adding regions.
///
/// The pass is based on a ConversionTarget, TypeConverters. legality checks and
/// conversion patterns.
///
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/PTensorToLinalg/PTensorToLinalg.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Dialect/PTensor/Transforms/Utils.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
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

#include "../PassDetail.h"

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
             inpOp.getOperands()
                 .front()
                 .getType()
                 .isa<::mlir::RankedTensorType>()) {
        inpOp = inpOp.getOperands()
                    .front()
                    .getDefiningOp<::mlir::UnrealizedConversionCastOp>();
      }
      assert(inpOp);
      // assert(inpOp.getOperands().size() == 4);
      assert(inpOp.getOperands()
                 .front()
                 .getType()
                 .isa<::mlir::RankedTensorType>());
      // std::cerr << "repl: "; inpOp.dump(); std::cerr << " ";
      // inpOp.getOperands()[0].dump(); std::cerr << std::endl;
      rewriter.replaceOp(op, inpOp.getOperands()[0]);
    }
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
    auto start = adaptor.getStart();
    auto stop = adaptor.getStop();
    auto step = adaptor.getStep();
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

    auto count = createCountARange(rewriter, loc, start, stop, step);

    // create shape vector
    auto retRtTyp = retPtTyp.getRtensor();
    assert(retRtTyp);
    auto elTyp = retRtTyp.getElementType();
    // init tensor
    auto tensor =
        rewriter
            .create<::mlir::tensor::EmptyOp>(
                loc, ::mlir::ArrayRef<::mlir::OpFoldResult>({count}), elTyp)
            .getResult();

    // fill with arange values
    // map needed for output only (we have no input tensor)
    const ::mlir::AffineMap maps[] = {
        ::mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext())};
    llvm::SmallVector<mlir::utils::IteratorType> iterators(
        1, mlir::utils::IteratorType::parallel);

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

    rewriter.replaceOpWithNewOp<::mlir::linalg::GenericOp>(
        op, retRtTyp, ::std::nullopt, tensor, maps, iterators, body);

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
    auto converter = *getTypeConverter();
    // we expect RankedTensorType as operands
    auto lhsRtTyp = lhsPtTyp.getRtensor();
    auto rhsRtTyp = rhsPtTyp.getRtensor();
    // input tensors might have compatible but different types
    assert(adaptor.getLhs().getType() == adaptor.getRhs().getType());

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
        loc, lhsRtTyp, op.getLhs());
    auto rhsTnsr = rewriter.create<::imex::ptensor::ExtractRTensorOp>(
        loc, rhsRtTyp, op.getRhs());

    // build tensor using the resulting element type
    // the shape is not statically known, we need to retrieve it (it's the same
    // as the input shapes)
    // FIXME shape broadcasting: input tensors might have compatible but
    // different shapes
    auto rank = static_cast<unsigned>(retRtTyp.getRank());
    llvm::SmallVector<::mlir::OpFoldResult> shapeVVec(rank);
    llvm::SmallVector<mlir::utils::IteratorType> iterators(rank);
    for (auto i : llvm::seq<unsigned>(0u, rank)) {
      shapeVVec[i] =
          rewriter.create<::mlir::tensor::DimOp>(loc, lhsTnsr, i).getResult();
      // iterate in parallel
      iterators[i] = mlir::utils::IteratorType::parallel;
    }

    // init tensor
    auto tensor =
        rewriter.create<::mlir::tensor::EmptyOp>(loc, shapeVVec, elTyp)
            .getResult();

    // Get signless operands into vec
    llvm::SmallVector<mlir::Value, 2> oprnds = {lhsTnsr, rhsTnsr};

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
    rewriter
        .replaceOpWithNewOp<::mlir::linalg::GenericOp>(
            op, tensor.getType(), oprnds, tensor, maps, iterators, bodyBuilder)
        .getResult(0);

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

    auto converter = *getTypeConverter();
    // we expect RankedTensorType as operands
    auto inpRtTyp = inpPtTyp.getRtensor();
    auto inpTnsr = rewriter.create<::imex::ptensor::ExtractRTensorOp>(
        loc, inpRtTyp, op.getInput());

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
    auto attr = rewriter.getIndexAttr(0);
    auto zeroI =
        rewriter.create<::mlir::arith::ConstantOp>(loc, attr).getResult();
    llvm::SmallVector<::mlir::OpFoldResult> shapeVVec(rank, zeroI);
    // create new tensor
    auto zero =
        rewriter
            .create<mlir::arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0))
            .getResult();
    auto tensor =
        rewriter.create<::mlir::tensor::EmptyOp>(loc, shapeVVec, sElTyp)
            .getResult();
    auto tnsr = rewriter.create<::mlir::linalg::FillOp>(loc, zero, tensor);

    // rank/num-dims of input
    auto inpRank = static_cast<unsigned>(inpRtTyp.getRank());
    // input maps are identity maps
    auto inpMap = ::mlir::AffineMap::getMultiDimIdentityMap(
        inpRank, rewriter.getContext());
    // output map is "*->()"
    auto omap = ::mlir::AffineMap::get(inpRank, 0, rewriter.getContext());
    const ::mlir::AffineMap maps[] = {inpMap, omap};
    llvm::SmallVector<mlir::utils::IteratorType> iterators(
        inpRank, mlir::utils::IteratorType::reduction);

    // create reduction op as linalg::generic
    const ::imex::ptensor::ReduceOpId ropid =
        (::imex::ptensor::ReduceOpId)adaptor.getOp()
            .cast<::mlir::IntegerAttr>()
            .getInt();
    auto bodyBuilder = getBodyBuilder(ropid, sElTyp);
    rewriter
        .replaceOpWithNewOp<::mlir::linalg::GenericOp>(
            op, tnsr.getType(0), oprnds, tnsr.getResult(0), maps, iterators,
            bodyBuilder)
        .getResult(0);

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
        -> ::mlir::Optional<::mlir::Type> { return type.getRtensor(); };

    typeConverter.addConversion(convT2T);
    typeConverter.addConversion(convPT2Tuple);

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
    // ...into Linalg, Affine, Tensor, Arith
    target.addLegalDialect<::mlir::linalg::LinalgDialect>();
    target.addLegalDialect<::mlir::AffineDialect>();
    target.addLegalDialect<::mlir::tensor::TensorDialect>();
    target.addLegalDialect<::mlir::arith::ArithDialect>();
    target.addLegalDialect<::mlir::shape::ShapeDialect>();
    target.addLegalDialect<::mlir::memref::MemRefDialect>();
    target.addLegalDialect<::mlir::bufferization::BufferizationDialect>();
    target.addLegalOp<::mlir::UnrealizedConversionCastOp>(); // FIXME
    // make sure function boundaries use RankedTensors (not PTensors)
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
    patterns.insert<MkPTensorLowering, ExtractRTensorLowering, ARangeLowering,
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

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertPTensorToLinalgPass() {
  return std::make_unique<ConvertPTensorToLinalgPass>();
}

} // namespace imex
