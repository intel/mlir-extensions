// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/ScalarOpsConversion.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Transforms/DialectConversion.h>

namespace {
template <typename Op>
struct ConvertUnaryOp : public mlir::OpConversionPattern<Op> {
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    assert(operands.size() == 1);
    rewriter.replaceOpWithNewOp<Op>(op, operands.front());
    return mlir::success();
  }
};

template <typename Op>
struct ConvertBinaryOp : public mlir::OpConversionPattern<Op> {
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    assert(operands.size() == 2);
    rewriter.replaceOpWithNewOp<Op>(op, operands[0], operands[1]);
    return mlir::success();
  }
};

template <typename Op>
struct ConvertTernaryOp : public mlir::OpConversionPattern<Op> {
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    assert(operands.size() == 3);
    rewriter.replaceOpWithNewOp<Op>(op, operands[0], operands[1], operands[2]);
    return mlir::success();
  }
};

template <typename Op>
struct ConvertUnaryFMOp : public mlir::OpConversionPattern<Op> {
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    assert(operands.size() == 1);
    rewriter.replaceOpWithNewOp<Op>(op, operands.front(),
                                    adaptor.getFastmath());
    return mlir::success();
  }
};

template <typename Op>
struct ConvertBinaryFMOp : public mlir::OpConversionPattern<Op> {
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    assert(operands.size() == 2);
    rewriter.replaceOpWithNewOp<Op>(op, operands[0], operands[1],
                                    adaptor.getFastmath());
    return mlir::success();
  }
};

template <typename Op>
struct ConvertCastOp : public mlir::OpConversionPattern<Op> {
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = this->getTypeConverter();
    assert(converter && "Invalid type converter");

    auto resType = converter->convertType(op.getResult().getType());
    if (!resType)
      return mlir::failure();

    auto operands = adaptor.getOperands();
    assert(operands.size() == 1);

    auto arg = operands.front();
    if (arg.getType() == resType) {
      rewriter.replaceOp(op, arg);
    } else {
      rewriter.replaceOpWithNewOp<Op>(op, resType, arg);
    }

    return mlir::success();
  }
};

template <typename Op>
struct ConvertCmpOp : public mlir::OpConversionPattern<Op> {
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<Op>(op, adaptor.getPredicate(),
                                    adaptor.getLhs(), adaptor.getRhs());
    return mlir::success();
  }
};

struct ConvertSelectOp
    : public mlir::OpConversionPattern<mlir::arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op,
                  mlir::arith::SelectOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(
        op, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return mlir::success();
  }
};

struct ConvertConstantOp
    : public mlir::OpConversionPattern<mlir::arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op,
                  mlir::arith::ConstantOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = this->getTypeConverter();
    assert(converter && "Invalid type converter");

    auto resType = converter->convertType(op.getResult().getType());
    if (!resType)
      return mlir::failure();

    auto oldAttr = adaptor.getValue();
    mlir::TypedAttr newAttr;
    if (auto fpAttr = oldAttr.dyn_cast<mlir::FloatAttr>()) {
      newAttr = rewriter.getFloatAttr(resType, fpAttr.getValueAsDouble());
    } else if (auto intAttr = oldAttr.dyn_cast<mlir::IntegerAttr>()) {
      newAttr = rewriter.getIntegerAttr(resType, intAttr.getValue());
    } else {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, newAttr);
    return mlir::success();
  }
};
} // namespace

void imex::populateArithConversionRewritesAndTarget(
    mlir::TypeConverter &converter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {
  target.addDynamicallyLegalDialect<mlir::arith::ArithDialect>(
      [&converter](mlir::Operation *op) -> llvm::Optional<bool> {
        if (converter.isLegal(op))
          return true;

        return std::nullopt;
      });

  patterns.insert<
      // clang-format off
      ConvertBinaryOp<mlir::arith::AddIOp>,
      ConvertBinaryOp<mlir::arith::AndIOp>,
      ConvertBinaryOp<mlir::arith::CeilDivSIOp>,
      ConvertBinaryOp<mlir::arith::CeilDivUIOp>,
      ConvertBinaryOp<mlir::arith::DivSIOp>,
      ConvertBinaryOp<mlir::arith::DivUIOp>,
      ConvertBinaryOp<mlir::arith::FloorDivSIOp>,
      ConvertBinaryOp<mlir::arith::MaxSIOp>,
      ConvertBinaryOp<mlir::arith::MaxUIOp>,
      ConvertBinaryOp<mlir::arith::MinSIOp>,
      ConvertBinaryOp<mlir::arith::MinUIOp>,
      ConvertBinaryOp<mlir::arith::MulIOp>,
      ConvertBinaryOp<mlir::arith::OrIOp>,
      ConvertBinaryOp<mlir::arith::RemSIOp>,
      ConvertBinaryOp<mlir::arith::RemUIOp>,
      ConvertBinaryOp<mlir::arith::ShLIOp>,
      ConvertBinaryOp<mlir::arith::ShRSIOp>,
      ConvertBinaryOp<mlir::arith::ShRUIOp>,
      ConvertBinaryOp<mlir::arith::SubIOp>,
      ConvertBinaryOp<mlir::arith::XOrIOp>,

      ConvertBinaryFMOp<mlir::arith::AddFOp>,
      ConvertBinaryFMOp<mlir::arith::DivFOp>,
      ConvertBinaryFMOp<mlir::arith::MaxFOp>,
      ConvertBinaryFMOp<mlir::arith::MinFOp>,
      ConvertBinaryFMOp<mlir::arith::MulFOp>,
      ConvertBinaryFMOp<mlir::arith::RemFOp>,
      ConvertBinaryFMOp<mlir::arith::SubFOp>,

      ConvertCastOp<mlir::arith::BitcastOp>,
      ConvertCastOp<mlir::arith::ExtFOp>,
      ConvertCastOp<mlir::arith::ExtSIOp>,
      ConvertCastOp<mlir::arith::ExtUIOp>,
      ConvertCastOp<mlir::arith::FPToSIOp>,
      ConvertCastOp<mlir::arith::FPToUIOp>,
      ConvertCastOp<mlir::arith::IndexCastOp>,
      ConvertCastOp<mlir::arith::IndexCastUIOp>,
      ConvertCastOp<mlir::arith::SIToFPOp>,
      ConvertCastOp<mlir::arith::UIToFPOp>,
      ConvertCastOp<mlir::arith::TruncFOp>,
      ConvertCastOp<mlir::arith::TruncIOp>,

      ConvertCmpOp<mlir::arith::CmpIOp>,
      ConvertCmpOp<mlir::arith::CmpFOp>,

      ConvertUnaryFMOp<mlir::arith::NegFOp>,

      ConvertConstantOp,
      ConvertSelectOp
      // clang-format on
      >(converter, patterns.getContext());
}

void imex::populateMathConversionRewritesAndTarget(
    mlir::TypeConverter &converter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {
  target.addDynamicallyLegalDialect<mlir::math::MathDialect>(
      [&converter](mlir::Operation *op) -> llvm::Optional<bool> {
        if (converter.isLegal(op))
          return true;

        return std::nullopt;
      });

  patterns.insert<
      // clang-format off
      ConvertUnaryOp<mlir::math::AbsFOp>,
      ConvertUnaryOp<mlir::math::AbsIOp>,
      ConvertUnaryOp<mlir::math::AtanOp>,
      ConvertUnaryOp<mlir::math::CeilOp>,
      ConvertUnaryOp<mlir::math::CosOp>,
      ConvertUnaryOp<mlir::math::CountLeadingZerosOp>,
      ConvertUnaryOp<mlir::math::CountTrailingZerosOp>,
      ConvertUnaryOp<mlir::math::CtPopOp>,
      ConvertUnaryOp<mlir::math::ErfOp>,
      ConvertUnaryOp<mlir::math::Exp2Op>,
      ConvertUnaryOp<mlir::math::ExpM1Op>,
      ConvertUnaryOp<mlir::math::ExpOp>,
      ConvertUnaryOp<mlir::math::Log10Op>,
      ConvertUnaryOp<mlir::math::Log1pOp>,
      ConvertUnaryOp<mlir::math::Log2Op>,
      ConvertUnaryOp<mlir::math::LogOp>,
      ConvertUnaryOp<mlir::math::RsqrtOp>,
      ConvertUnaryOp<mlir::math::SinOp>,
      ConvertUnaryOp<mlir::math::SqrtOp>,
      ConvertUnaryOp<mlir::math::TanOp>,
      ConvertUnaryOp<mlir::math::TanhOp>,
      ConvertUnaryOp<mlir::math::TruncOp>,
      ConvertUnaryOp<mlir::math::RoundOp>,
      ConvertUnaryOp<mlir::math::RoundEvenOp>,

      ConvertBinaryOp<mlir::math::CopySignOp>,
      ConvertBinaryOp<mlir::math::Atan2Op>,
      ConvertBinaryOp<mlir::math::FPowIOp>,
      ConvertBinaryOp<mlir::math::IPowIOp>,
      ConvertBinaryOp<mlir::math::PowFOp>,

      ConvertTernaryOp<mlir::math::FmaOp>
      // clang-format on
      >(converter, patterns.getContext());
}
