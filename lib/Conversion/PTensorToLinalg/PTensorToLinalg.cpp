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

namespace imex {

// return type without a sign
// copied from py_linalg_resolver.cpp
static mlir::Type makeSignlessType(mlir::Type type) {
  if (auto shaped = type.dyn_cast<mlir::ShapedType>()) {
    auto origElemType = shaped.getElementType();
    return makeSignlessType(origElemType);
  } else if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    if (!intType.isSignless())
      return mlir::IntegerType::get(intType.getContext(), intType.getWidth());
  }
  return type;
}

// creating operand cast to signless type if needed
// copied from py_linalg_resolver.cpp
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

#if 0
// creating operand cast to given type if needed
// copied from py_linalg_resolver.cpp
static mlir::Value doSignCast(mlir::OpBuilder &builder, mlir::Location &loc,
                              mlir::Value val, mlir::Type dstType) {
  auto origType = val.getType();
  if (dstType != origType) {
      val = builder.create<::mlir::UnrealizedConversionCastOp>(loc, dstType, val).getResult(0);
  }
  return val;
}
#endif

// Initialze a distributed Tensor:
// 1. register tensor with runtime
// 2. get local shape
// 3. init local tensor
// returns pair of tensor and id as assigned by runtime
// If not distributed, simply init tensor
static auto initDTensor(mlir::Location &loc,
                        ::mlir::ConversionPatternRewriter &rewriter, bool dist,
                        uint64_t rank, llvm::SmallVector<mlir::Value> shp_vals,
                        ::mlir::Value shp_tnsr, ::mlir::Type eltyp,
                        ::llvm::SmallVector<mlir::Value> &lshp /* out */) {
  if (dist) {
    auto ityp = rewriter.getI64Type();
    auto idxtyp = rewriter.getIndexType();
    auto shptyp =
        mlir::RankedTensorType::get(llvm::SmallVector<int64_t>(rank), idxtyp);

    // Register with runtime
    ::mlir::Value id =
        rewriter.create<::imex::dist::RegisterPTensorOp>(loc, ityp, shp_tnsr);
    // and get local shape
    auto lshp_mr = rewriter.create<::imex::dist::LocalShapeOp>(loc, shptyp, id);

    // get shape as SmallVector<mlir::Value>
    // why can't we just use the existing tensor?
    lshp.resize(rank);
    for (auto i : ::llvm::seq(0LU, rank)) {
      auto ia = rewriter.getIndexAttr(i);
      auto idx = rewriter.create<::mlir::arith::ConstantOp>(loc, ia);
      lshp[i] = rewriter.create<::mlir::tensor::ExtractOp>(
          loc, idxtyp, lshp_mr, ::mlir::ValueRange({idx}));
    }
    // create a 1d tensor of local shape
    auto ltnsr =
        rewriter.create<::mlir::linalg::InitTensorOp>(loc, lshp, eltyp);
    return std::make_pair(ltnsr.getResult(), id);
  } else { // not distributed, simply init
    auto ltnsr =
        rewriter.create<::mlir::linalg::InitTensorOp>(loc, shp_vals, eltyp);
    return std::make_pair(ltnsr.getResult(), ::mlir::Value());
  }
}

// *******************************
// ***** Individual patterns *****
// *******************************

namespace {
// convert PTensor's arange and its return type to Linalg/tensor
// we also need some arith and affine (for linalg::genericop)
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
    auto orgrtyp = op.getType().dyn_cast<::imex::ptensor::PTensorType>();
    assert(orgrtyp);

    // we operate on signless integers
    auto ityp = rewriter.getI64Type();
    if (start.getType() != ityp) {
      start =
          rewriter.create<::mlir::UnrealizedConversionCastOp>(loc, ityp, start)
              .getResult(0);
    }
    if (stop.getType() != ityp) {
      stop =
          rewriter.create<::mlir::UnrealizedConversionCastOp>(loc, ityp, stop)
              .getResult(0);
    }
    if (step.getType() != ityp) {
      step =
          rewriter.create<::mlir::UnrealizedConversionCastOp>(loc, ityp, step)
              .getResult(0);
    }

    // Create constants 0, 1, -1 for later
    auto zattr = rewriter.getI64IntegerAttr(0);
    auto zero =
        rewriter.create<mlir::arith::ConstantOp>(loc, zattr).getResult();
    auto oattr = rewriter.getI64IntegerAttr(1);
    auto one = rewriter.create<mlir::arith::ConstantOp>(loc, oattr).getResult();
    auto mattr = rewriter.getI64IntegerAttr(-1);
    auto mone =
        rewriter.create<mlir::arith::ConstantOp>(loc, mattr).getResult();

    // Compute number of elements as
    //   (stop - start + step + (step < 0 ? 1 : -1)) / step
    auto cnd = rewriter.create<mlir::arith::CmpIOp>(
        loc, ::mlir::arith::CmpIPredicate::ult, step, zero);
    auto inc = rewriter.create<mlir::arith::SelectOp>(loc, cnd, one, mone);
    auto tmp1 = rewriter.create<mlir::arith::AddIOp>(loc, stop, step);
    auto tmp2 = rewriter.create<mlir::arith::AddIOp>(loc, tmp1, inc);
    auto tmp3 = rewriter.create<mlir::arith::SubIOp>(loc, tmp2, start);
    auto cnt =
        rewriter.create<mlir::arith::DivUIOp>(loc, tmp3, step).getResult();
    cnt = rewriter
              .create<::mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                  cnt)
              .getResult();

    // create shape vector
    auto ttyp = converter.convertType(op.getType())
                    .dyn_cast<::mlir::RankedTensorType>();
    assert(ttyp);
    auto typ = ttyp.getElementType();
    llvm::SmallVector<mlir::Value> shp(1, cnt);

    // register and init tensor
    llvm::SmallVector<mlir::Value> lshp(1);
    auto tmp_tnsr = rewriter.create<::mlir::linalg::InitTensorOp>(
        loc, ::mlir::ValueRange({cnt}), typ);
    auto shape = rewriter.create<::mlir::shape::ShapeOfOp>(loc, tmp_tnsr);
    auto tnsr_id =
        initDTensor(loc, rewriter, orgrtyp.getDist(), 1, shp, shape, typ, lshp);

    // compute start index of local partition
    if (orgrtyp.getDist()) {
      auto offtyp =
          rewriter
              .getIndexType(); // mlir::MemRefType::get(llvm::SmallVector<int64_t>(1,
                               // mlir::ShapedType::kDynamicSize), ityp);
      auto offs = rewriter.create<::imex::dist::LocalOffsetsOp>(loc, offtyp,
                                                                tnsr_id.second);
      // auto _off = rewriter.create<::mlir::memref::DimOp>(loc, offs, 0);
      auto off = rewriter.create<mlir::arith::IndexCastOp>(loc, ityp, offs);
      auto tmp =
          rewriter.create<mlir::arith::MulIOp>(loc, off, step); // off * step
      start =
          rewriter.create<mlir::arith::AddIOp>(loc, start,
                                               tmp); // start + (off * stride)
    }

    // fill with arange values
    // map needed for output only (we have no input tensor)
    const ::mlir::AffineMap maps[] = {
        ::mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext())};
    llvm::SmallVector<mlir::StringRef> iterators(1, "parallel");

    // The body; accepting no input, the lambda simply captures start and step
    auto body = [&start, &step, &typ, &ityp](mlir::OpBuilder &builder,
                                             ::mlir::Location loc,
                                             ::mlir::ValueRange args) {
      auto dim = builder.getI64IntegerAttr(0);
      auto idx = builder.create<mlir::linalg::IndexOp>(loc, dim);
      auto _idx = builder.create<mlir::arith::IndexCastOp>(loc, ityp, idx);
      auto tmp = builder.create<mlir::arith::MulIOp>(loc, step, _idx);
      auto val = builder.create<mlir::arith::AddIOp>(loc, start, tmp);
      auto ret = builder.create<::mlir::UnrealizedConversionCastOp>(
          loc, typ, val.getResult());
      // auto _val = builder.create<mlir::arith::SIToFPOp>(loc, typ, val);
      (void)builder.create<mlir::linalg::YieldOp>(loc, ret.getResult(0));
    };

    (void)rewriter.replaceOpWithNewOp<mlir::linalg::GenericOp>(
        op, ttyp, llvm::None, tnsr_id.first, maps, iterators, body);
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

// trivial builders have simple arith equivalents
// the arith ops are template arguments, one for ints and one for floats
// currently only integers and floats are supported
// currently unsigned int ops are not supported
template <typename IOP, typename FOP = void>
static BodyType buildTrivial(::mlir::Type typ) {
  return [typ](mlir::OpBuilder &builder, ::mlir::Location loc,
               ::mlir::ValueRange args) -> void {
    auto lhs_typ = args[0].getType();
    if (lhs_typ.isIntOrIndex()) {
      if constexpr (!std::is_same_v<IOP, void>) {
        auto lhs = doSignCast(builder, loc, args[0]);
        auto rhs = doSignCast(builder, loc, args[1]);
        yield(builder, loc, typ,
              builder.create<IOP>(loc, lhs, rhs).getResult());
        return;
      } else
        assert("Found integer type but binary op not defined for integers" ==
               nullptr);
    } else if (lhs_typ.isIntOrIndexOrFloat()) {
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

// get a body builder for given binary operation and result type
// we accept a result type to insert a cast after the operation if needed
static BodyType getBodyBuilder(::imex::ptensor::EWBinOpId bop,
                               ::mlir::Type typ) {
  switch (bop) {
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

// convert PTensor's elementwise binary operations and their return type to
// Linalg/tensor the given op's type is expected to convert to the apprioprate
// type (shape and element-type) we also need some arith and affine (for
// linalg::genericop)
// Convert PTensor's elementwise binary operations to Linalg
struct EWBinOpLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::EWBinOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::EWBinOp op,
                  ::imex::ptensor::EWBinOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // We expect to lower PTensors
    auto lhsorgtyp =
        op.getLhs().getType().dyn_cast<::imex::ptensor::PTensorType>();
    auto rhsorgtyp =
        op.getRhs().getType().dyn_cast<::imex::ptensor::PTensorType>();
    // we expect RankedTensorType as operands
    auto lhstyp =
        adaptor.getLhs().getType().dyn_cast<::mlir::RankedTensorType>();
    auto rhstyp =
        adaptor.getRhs().getType().dyn_cast<::mlir::RankedTensorType>();
    if (!lhstyp || !rhstyp || !lhsorgtyp || !rhsorgtyp) {
      // fail if not, will be retried if operands get converted elsewhere
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto converter = *getTypeConverter();

    // input tensors might have compatible but different types
    assert(adaptor.getLhs().getType() == adaptor.getRhs().getType());
    assert(adaptor.getLhs().getType() == adaptor.getRhs().getType());

    // the element type of a binop depends on the input arguments and the
    // operation itself we assume this had beeen taken care of and simply use
    // the op's converted type
    auto _t = converter.convertType(op.getType());
    auto shaped = _t.dyn_cast<::mlir::RankedTensorType>();
    assert(shaped);
    auto typ = shaped.getElementType();

    // build tensor using the resulting element type
    // the shape is not statically known, we need to retrieve it (it's the same
    // as the input shapes)
    // FIXME shape broadcasting: input tensors might have compatible but
    // different shapes
    auto lhs = adaptor.getLhs();
    auto rank = static_cast<unsigned>(shaped.getRank());
    llvm::SmallVector<mlir::Value> shp(rank);
    llvm::SmallVector<mlir::StringRef> iterators(rank);
    for (auto i : llvm::seq(0u, rank)) {
      shp[i] = rewriter.create<::mlir::tensor::DimOp>(loc, lhs, i);
      // iterate in parallel
      iterators[i] = "parallel";
    }

    // register and init tensor
    llvm::SmallVector<mlir::Value> lshp(rank);
    auto tmp_tnsr =
        rewriter.create<::mlir::linalg::InitTensorOp>(loc, shp, typ);
    auto shape = rewriter.create<::mlir::shape::ShapeOfOp>(loc, tmp_tnsr);
    auto tnsr_id = initDTensor(loc, rewriter, lhsorgtyp.getDist(), rank, shp,
                               shape, typ, lshp);

    // Get signless operands into vec
    llvm::SmallVector<mlir::Value, 2> oprnds = {adaptor.getLhs(),
                                                adaptor.getRhs()};

    // all maps are identity maps
    auto imap =
        ::mlir::AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    const ::mlir::AffineMap maps[] = {imap, imap, imap};

    // create binop as linalg::generic
    const ::imex::ptensor::EWBinOpId bopid =
        (::imex::ptensor::EWBinOpId)adaptor.getOp()
            .cast<::mlir::IntegerAttr>()
            .getInt();
    auto bodyBuilder = getBodyBuilder(bopid, typ);
    (void)rewriter
        .replaceOpWithNewOp<::mlir::linalg::GenericOp>(
            op, tnsr_id.first.getType(), oprnds, tnsr_id.first, maps, iterators,
            bodyBuilder)
        .getResult(0);
    return ::mlir::success();
  }
};

// get a body builder for given binary operation and result type
// we accept a result type to insert a cast after the operation if needed
static BodyType getBodyBuilder(::imex::ptensor::ReduceOpId rop,
                               ::mlir::Type typ) {
  switch (rop) {
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

// convert PTensor's reduction operations and their return type to Linalg/tensor
// the given op's type is expected to convert to the appropriate type (shape and
// element-type) we also need some arith and affine (for linalg::genericop)
// FIXME reduction over a subset of dimensionsstruct ReductionOpLowering
struct ReductionOpLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ReductionOp op,
                  ::imex::ptensor::ReductionOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // we expect RankedTensorType as operands
    auto inptyp =
        adaptor.getInput().getType().dyn_cast<::mlir::RankedTensorType>();
    auto orginptyp =
        op.getInput().getType().dyn_cast<::imex::ptensor::PTensorType>();
    if (!inptyp || !orginptyp) {
      // fail if not, will be retried if operands get converted elsewhere
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto converter = *getTypeConverter();

    // Get signless operands into vec
    llvm::SmallVector<mlir::Value, 1> oprnds = {adaptor.getInput()};

    // determine resulting element type from converted op-type
    auto _t = converter.convertType(op.getType());
    auto shaped = _t.dyn_cast<::mlir::RankedTensorType>();
    assert(shaped);
    auto typ = shaped.getElementType();
    auto sltyp = makeSignlessType(typ);

    // build tensor using the resulting element type and shape
    // FIXME support reduction dimensions
    auto rank = static_cast<unsigned>(shaped.getRank());
    assert(rank == 0);
    llvm::SmallVector<mlir::Value> shp(
        rank); //::mlir::ShapedType::kDynamicSize;
    // create new tensor
    auto zattr = rewriter.getI64IntegerAttr(0);
    auto zero =
        rewriter.create<mlir::arith::ConstantOp>(loc, zattr).getResult();
    llvm::SmallVector<mlir::Value> lshp(rank);
    auto tmp_tnsr =
        rewriter.create<::mlir::linalg::InitTensorOp>(loc, shp, typ);
    auto shape = rewriter.create<::mlir::shape::ShapeOfOp>(loc, tmp_tnsr);
    auto tnsr_id = initDTensor(loc, rewriter, orginptyp.getDist(), rank, shp,
                               shape, sltyp, lshp);
    auto tnsr =
        rewriter.create<::mlir::linalg::FillOp>(loc, zero, tnsr_id.first);

    // rank/num-dims of input
    auto irank = static_cast<unsigned>(inptyp.getRank());
    // input maps are identity maps
    auto imap =
        ::mlir::AffineMap::getMultiDimIdentityMap(irank, rewriter.getContext());
    // output map is "*->()"
    auto omap = ::mlir::AffineMap::get(irank, 0, rewriter.getContext());
    const ::mlir::AffineMap maps[] = {imap, omap};
    llvm::SmallVector<mlir::StringRef> iterators(irank, "reduction");

    // create reduction op as linalg::generic
    const ::imex::ptensor::ReduceOpId ropid =
        (::imex::ptensor::ReduceOpId)adaptor.getOp()
            .cast<::mlir::IntegerAttr>()
            .getInt();
    auto bodyBuilder = getBodyBuilder(ropid, sltyp);
    auto rtnsr = rewriter
                     .create<::mlir::linalg::GenericOp>(
                         loc, tnsr.getType(0), oprnds, tnsr.getResult(0), maps,
                         iterators, bodyBuilder)
                     .getResult(0);

    // we reduced the local part, now we reduce across processes
    if (orginptyp.getDist()) {
      rtnsr = rewriter.create<::imex::dist::AllReduceOp>(
          loc, tnsr.getType(0), adaptor.getOp(), rtnsr);
    }

    // For now we only support reduction over all dims and return a scalar
    auto rval = rewriter.create<::mlir::tensor::ExtractOp>(
        loc, sltyp, rtnsr, ::mlir::ValueRange());
    (void)rewriter.replaceOpWithNewOp<::mlir::UnrealizedConversionCastOp>(
        op, typ, rval.getResult());

    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Converting PTensor to Linalg
// After success, no more PTensor should be left, replaced by Linalg & Affine &
// Arith We use a type converter to get rid of PTensorType
struct ConvertPTensorToLinalgPass
    : public ::imex::ConvertPTensorToLinalgBase<ConvertPTensorToLinalgPass> {

  ConvertPTensorToLinalgPass() = default;

  void runOnOperation() override {
    auto &ctxt = getContext();
    ::mlir::ConversionTarget target(ctxt);
    ::mlir::TypeConverter typeConverter;
    // Convert unknown types to itself
    typeConverter.addConversion([](::mlir::Type type) { return type; });
    // Convert PTensorType to its RankedTensorType
    typeConverter.addConversion(
        [&typeConverter](::imex::ptensor::PTensorType type)
            -> llvm::Optional<::mlir::Type> { return type.getRtensor(); });

#if 1
    // In theory we should not need any materialization
    // if we use a hybrid conversion (plier->ptensor->linalg and direct
    // plier->linalg) we might need it, though
    auto materializeCast =
        [](::mlir::OpBuilder &builder, ::mlir::Type type,
           ::mlir::ValueRange inputs,
           ::mlir::Location loc) -> ::llvm::Optional<::mlir::Value> {
      if (inputs.size() == 1) {
        return builder
            .create<::mlir::UnrealizedConversionCastOp>(loc, type,
                                                        inputs.front())
            .getResult(0);
      }
      return ::llvm::None;
    };
    // typeConverter.addArgumentMaterialization(materializeCast);
    typeConverter.addSourceMaterialization(materializeCast);
    // typeConverter.addTargetMaterialization(materializeCast);
#endif
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
        [&](::mlir::func::ReturnOp op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<mlir::func::CallOp>(
        [&](mlir::func::CallOp op) { return typeConverter.isLegal(op); });

    ::mlir::RewritePatternSet patterns(&ctxt);
    patterns.insert<ARangeLowering, EWBinOpLowering, ReductionOpLowering>(
        typeConverter, &ctxt);

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
