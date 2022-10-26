//===- DistToStandard.cpp - DistToStandard conversion  ----------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the DistToStandard conversion, converting the Dist
/// dialect to standard dialects.
/// Some operations get converted to runtime calls, others with standard
/// MLIR operations from dialects like arith and tensor.
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/DistToStandard/DistToStandard.h>
#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/internal/PassUtils.h>
#include <imex/internal/PassWrapper.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>

#include <iostream>

#include "../PassDetail.h"

namespace imex {
namespace {

// create function prototype fo given function name, arg-types and
// return-types
inline void requireFunc(::mlir::Location &loc, ::mlir::OpBuilder &builder,
                        ::mlir::ModuleOp module, const char *fname,
                        ::mlir::TypeRange args, ::mlir::TypeRange results) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  // Insert before module terminator.
  builder.setInsertionPoint(module.getBody(),
                            std::prev(module.getBody()->end()));
  auto funcType = builder.getFunctionType(args, results);
  auto func = builder.create<::mlir::func::FuncOp>(loc, fname, funcType);
  func.setPrivate();
}

// *******************************
// ***** Individual patterns *****
// *******************************

// RuntimePrototypesOp -> func.func ops
struct RuntimePrototypesOpConverter
    : public mlir::OpConversionPattern<::imex::dist::RuntimePrototypesOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::RuntimePrototypesOp op,
                  ::imex::dist::RuntimePrototypesOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // get guid and rank and call runtime function
    auto loc = op.getLoc();
    auto mod = op->getParentOp();
    assert(::mlir::isa<mlir::ModuleOp>(mod));
    ::mlir::ModuleOp module = ::mlir::cast<mlir::ModuleOp>(mod);
    auto dtype = rewriter.getI64Type();
    auto i64Type = rewriter.getI64Type();
    auto dtypeType = rewriter.getIntegerType(sizeof(int) * 8);
    auto opType =
        rewriter.getIntegerType(sizeof(::imex::ptensor::ReduceOpId) * 8);
#if 0
    requireFunc(loc, rewriter, module, "_idtr_init_dtensor",
                {::mlir::RankedTensorType::get({-1}, dtype), i64Type},
                {i64Type});
    requireFunc(loc, rewriter, module, "_idtr_local_shape",
                {i64Type, ::mlir::RankedTensorType::get({-1}, dtype), i64Type},
                {});
    requireFunc(loc, rewriter, module, "_idtr_local_offsets",
                {i64Type, ::mlir::RankedTensorType::get({-1}, dtype), i64Type},
                {});
#endif // if 0
    requireFunc(loc, rewriter, module, "_idtr_nprocs", {i64Type}, {i64Type});
    requireFunc(loc, rewriter, module, "_idtr_prank", {}, {i64Type});
    requireFunc(loc, rewriter, module, "_idtr_reduce_all",
                {::mlir::RankedTensorType::get({}, dtype), dtypeType, opType},
                {});
    rewriter.eraseOp(op);
    return ::mlir::success();
  }
};

/// Convert ::imex::dist::NProcsOp into runtime call to _idtr_nprocs
struct NProcsOpConverter
    : public mlir::OpConversionPattern<::imex::dist::NProcsOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::NProcsOp op,
                  ::imex::dist::NProcsOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<::mlir::func::CallOp>(
        op, "_idtr_nprocs", rewriter.getI64Type(), adaptor.getTeam());
    return ::mlir::success();
  }
};

// Convert ::imex::dist::PRankOp into runtime call to _idtr_prank
struct PRankOpConverter
    : public mlir::OpConversionPattern<::imex::dist::PRankOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::PRankOp op,
                  ::imex::dist::PRankOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<::mlir::func::CallOp>(op, "_idtr_prank",
                                                      rewriter.getI64Type());
    return ::mlir::success();
  }
};

/// Convert ::imex::dist::InitDistTensorOp into a UnrealizedConversionCastOp,
/// using the type converter to determine the target type. Operations extracting
/// members (info, ptensor) are expected to chase the creation back to here and
/// get the respective operand of the cast. After full conversion the cast is
/// expected ot have no use.
// FIXME Is there a better/cleaner way to do this?
struct InitDistTensorOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::InitDistTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::InitDistTensorOp op,
                  ::imex::dist::InitDistTensorOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = *getTypeConverter();
    (void)rewriter.replaceOpWithNewOp<::mlir::UnrealizedConversionCastOp>(
        op, converter.convertType(op.getType()), adaptor.getOperands());
    return ::mlir::success();
  }
};

/// Common implementation for getting defining op of given DistTensor
/// (assumed to be a InitDistTensorOp) and returning one of its operands
/// Currently the operand is provided as 2 template arguments:
/// extracting operation and operand-index-of-defining-op
template <typename OP, int i>
struct GetFromInitDTConverter : public ::mlir::OpConversionPattern<OP> {
  using ::mlir::OpConversionPattern<OP>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(OP op, typename OP::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto defOp =
        adaptor.getDTensor()
            .template getDefiningOp<::mlir::UnrealizedConversionCastOp>();
    if (!defOp)
      return ::mlir::failure();
    rewriter.replaceOp(op, defOp.getOperands()[i]);
    return ::mlir::success();
  }
};

/// Convert ::imex::dist::ExtractFromInfoOp into respective operand of defining
/// op. DistInfo is expected to be converted to a unrealized_conversion_cast.
struct ExtractFromInfoOpConverter
    : public mlir::OpConversionPattern<::imex::dist::ExtractFromInfoOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::ExtractFromInfoOp op,
                  ::imex::dist::ExtractFromInfoOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto defOp =
        adaptor.getInfo().getDefiningOp<::mlir::UnrealizedConversionCastOp>();
    assert(defOp);
    rewriter.replaceOp(op, defOp.getOperands()[adaptor.getWhat()]);
    return ::mlir::success();
  }
};

/// Convert ::imex::dist::DistInfoOp into a unrealized_conversion_cast
/// casting all operands to none.
/// After full conversion the cast is expected ot have no use.
struct DistInfoOpConverter
    : public mlir::OpConversionPattern<::imex::dist::DistInfoOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::DistInfoOp op,
                  ::imex::dist::DistInfoOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = *getTypeConverter();
    auto loc = op.getLoc();
    auto gshape = op.getShape();
    auto team = op.getTeam();
    auto rank = op.getRankAttr();
    auto np = rewriter.create<::imex::dist::NProcsOp>(loc, team);
    auto prank = rewriter.create<::imex::dist::PRankOp>(loc, team);
    auto lshape = rewriter.create<::imex::dist::LocalShapeOp>(
        loc, rank.getInt(), np, prank, gshape);
    auto loffsets = rewriter.create<::imex::dist::LocalOffsetsOp>(
        loc, rank.getInt(), np, prank, gshape);
    ::mlir::SmallVector<::mlir::Value> vals(::imex::dist::INFO_LAST);
    // vals[::imex::dist::RANK] =
    // rewriter.create<::mlir::arith::ConstantOp>(loc, rank);
    // vals[::imex::dist::GSHAPE] = gshape;
    vals[::imex::dist::LSHAPE] = lshape;
    vals[::imex::dist::LOFFSETS] = loffsets;
    vals[::imex::dist::TEAM] = team;
    rewriter.replaceOpWithNewOp<::mlir::UnrealizedConversionCastOp>(
        op, converter.convertType(op.getType()), vals);
    return ::mlir::success();
  }
};

/// compute tile-size from global shape and #procs
static ::mlir::Value createTileSize(const ::mlir::Location &loc,
                                    ::mlir::OpBuilder &builder,
                                    ::mlir::Value sz, ::mlir::Value np) {
  return builder.create<mlir::arith::DivUIOp>(
      loc,
      builder.create<mlir::arith::AddIOp>(
          loc, sz,
          builder.create<mlir::arith::SubIOp>(loc, np,
                                              createInt(loc, builder, 1))),
      np);
}

/// Convert ::imex::dist::LocalOffsetsOp into shape and arith calls.
/// We currently assume evenly split data.
struct LocalOffsetsOpConverter
    : public mlir::OpConversionPattern<::imex::dist::LocalOffsetsOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::LocalOffsetsOp op,
                  ::imex::dist::LocalOffsetsOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // FIXME: non-even partitions, ndims
    auto loc = op.getLoc();
    auto converter = *getTypeConverter();

    auto sz0_ = rewriter.create<::mlir::shape::GetExtentOp>(
        loc, adaptor.getGshape(), createIndex(loc, rewriter, 0));
    auto sz0 = rewriter.create<::mlir::arith::IndexCastOp>(
        loc, rewriter.getI64Type(), sz0_);
    auto tsz = createTileSize(loc, rewriter, sz0, adaptor.getNumProcs());
    auto off =
        rewriter.create<mlir::arith::MulIOp>(loc, adaptor.getPrank(), tsz);
    rewriter.replaceOpWithNewOp<::mlir::tensor::FromElementsOp>(
        op, converter.convertType(op.getType()), off.getResult());

    return ::mlir::success();
  }
};

/// Convert ::imex::dist::LocalShapeOp into shape and arith calls.
/// We currently assume evenly split data.
struct LocalShapeOpConverter
    : public mlir::OpConversionPattern<::imex::dist::LocalShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::LocalShapeOp op,
                  ::imex::dist::LocalShapeOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // FIXME: non-even partitions, ndims
    auto loc = op.getLoc();
    auto converter = *getTypeConverter();

    auto sz0_ = rewriter.create<::mlir::shape::GetExtentOp>(
        loc, adaptor.getGshape(), createIndex(loc, rewriter, 0));
    auto sz0 = rewriter.create<::mlir::arith::IndexCastOp>(
        loc, rewriter.getI64Type(), sz0_);
    auto tsz = createTileSize(loc, rewriter, sz0, adaptor.getNumProcs());
    rewriter.replaceOpWithNewOp<::mlir::tensor::FromElementsOp>(
        op, converter.convertType(op.getType()), tsz);

    return ::mlir::success();
  }
};

/// Convert ::imex::dist::AllReduceOp into runtime call to "_idtr_reduce_all".
/// Pass local RankedTensor as argument.
/// Replaces op with new DistTensor.
struct AllReduceOpConverter
    : public mlir::OpConversionPattern<::imex::dist::AllReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::AllReduceOp op,
                  ::imex::dist::AllReduceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // get guid and rank and call runtime function
    auto loc = op.getLoc();
    auto opV = rewriter.create<::mlir::arith::ConstantOp>(loc, op.getOp());
    auto rTnsr = adaptor.getTensor();
    auto dtype = createInt<sizeof(int) * 8>(loc, rewriter, 5); // FIXME getDType
    auto fsa = rewriter.getStringAttr("_idtr_reduce_all");
    rewriter.create<::mlir::func::CallOp>(
        loc, fsa, ::mlir::TypeRange(), ::mlir::ValueRange({rTnsr, dtype, opV}));
    rewriter.replaceOp(op, rTnsr);
    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Full Pass
struct ConvertDistToStandardPass
    : public ::imex::ConvertDistToStandardBase<ConvertDistToStandardPass> {
  ConvertDistToStandardPass() = default;

  void runOnOperation() override {
    auto &ctxt = getContext();
    ::mlir::ConversionTarget target(ctxt);
    ::mlir::TypeConverter typeConverter;
    // Convert unknown types to itself
    auto convT2T = [](::mlir::Type type) { return type; };
    // Convert DistInfo and DistTensor to none
    auto convInfo =
        [&ctxt](
            ::imex::dist::DistInfoType type) -> ::mlir::Optional<::mlir::Type> {
      return ::mlir::NoneType::get(&ctxt);
    };
    auto convDTensor = [&ctxt](::imex::dist::DistTensorType type)
        -> ::mlir::Optional<::mlir::Type> {
      return ::mlir::NoneType::get(&ctxt);
    };

    typeConverter.addConversion(convT2T);
    typeConverter.addConversion(convInfo);
    typeConverter.addConversion(convDTensor);

    auto materializeCast =
        [](::mlir::OpBuilder &builder, ::mlir::Type type,
           ::mlir::ValueRange inputs,
           ::mlir::Location loc) -> ::llvm::Optional<::mlir::Value> {
      return builder
          .create<::mlir::UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };
    typeConverter.addSourceMaterialization(materializeCast);

    target.addDynamicallyLegalOp<::mlir::func::FuncOp>(
        [&](::mlir::func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                 typeConverter.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<::mlir::func::ReturnOp>(
        [&](::mlir::func::ReturnOp op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<mlir::func::CallOp>(
        [&](mlir::func::CallOp op) { return typeConverter.isLegal(op); });

    // We convert all Dist stuff...
    target.addIllegalDialect<::imex::dist::DistDialect>();
    target.addLegalDialect<::mlir::linalg::LinalgDialect>();
    target.addLegalDialect<::mlir::tensor::TensorDialect>();
    target.addLegalDialect<::mlir::arith::ArithDialect>();
    target.addLegalDialect<::mlir::shape::ShapeDialect>();
    target.addLegalDialect<::imex::ptensor::PTensorDialect>();
    target.addLegalOp<::mlir::UnrealizedConversionCastOp>(); // FIXME

    ::mlir::RewritePatternSet patterns(&ctxt);
    patterns.insert<RuntimePrototypesOpConverter, NProcsOpConverter,
                    PRankOpConverter, InitDistTensorOpConverter,
                    GetFromInitDTConverter<::imex::dist::GetInfoOp, 1>,
                    GetFromInitDTConverter<::imex::dist::GetPTensorOp, 0>,
                    ExtractFromInfoOpConverter, DistInfoOpConverter,
                    LocalOffsetsOpConverter, LocalShapeOpConverter,
                    AllReduceOpConverter>(typeConverter, &ctxt);
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

/// Populate the given list with patterns that convert Dist to Standard
void populateDistToStandardConversionPatterns(
    ::mlir::LLVMTypeConverter &converter, ::mlir::RewritePatternSet &patterns) {
  assert(false);
}

/// Create a pass that convert Dist to Standard
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertDistToStandardPass() {
  return std::make_unique<ConvertDistToStandardPass>();
}

} // namespace imex
