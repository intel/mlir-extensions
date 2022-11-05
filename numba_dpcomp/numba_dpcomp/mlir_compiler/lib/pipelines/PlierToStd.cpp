// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _USE_MATH_DEFINES
#include <cmath>

#include "pipelines/PlierToScf.hpp"
#include "pipelines/PlierToStd.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "imex/Dialect/imex_util/Dialect.hpp"
#include "imex/Dialect/plier/Dialect.hpp"

#include "imex/Compiler/PipelineRegistry.hpp"
#include "imex/Transforms/CallLowering.hpp"
#include "imex/Transforms/CastUtils.hpp"
#include "imex/Transforms/ConstUtils.hpp"
#include "imex/Transforms/InlineUtils.hpp"
#include "imex/Transforms/PipelineUtils.hpp"
#include "imex/Transforms/RewriteWrapper.hpp"
#include "imex/Transforms/TypeConversion.hpp"

#include "BasePipeline.hpp"
#include "LoopUtils.hpp"
#include "Mangle.hpp"
#include "PyFuncResolver.hpp"
#include "PyLinalgResolver.hpp"

namespace {
static bool isSupportedType(mlir::Type type) {
  assert(type);
  return type.isIntOrFloat();
}

static bool isInt(mlir::Type type) {
  assert(type);
  return type.isa<mlir::IntegerType>();
}

static bool isFloat(mlir::Type type) {
  assert(type);
  return type.isa<mlir::FloatType>();
}

struct ConstOpLowering : public mlir::OpConversionPattern<plier::ConstOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::ConstOp op, plier::ConstOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter = *getTypeConverter();
    auto expectedType = converter.convertType(op.getType());
    if (!expectedType)
      return mlir::failure();

    auto value = adaptor.getVal();
    auto typeAttr = value.dyn_cast_or_null<mlir::TypedAttr>();
    if (typeAttr && isSupportedType(typeAttr.getType())) {
      if (auto intAttr = value.dyn_cast<mlir::IntegerAttr>()) {
        auto type = intAttr.getType().cast<mlir::IntegerType>();
        if (!type.isSignless()) {
          auto loc = op.getLoc();
          auto intVal = intAttr.getValue().getSExtValue();
          auto constVal = rewriter.create<mlir::arith::ConstantIntOp>(
              loc, intVal, type.getWidth());
          mlir::Value res =
              rewriter.create<imex::util::SignCastOp>(loc, type, constVal);
          if (res.getType() != expectedType)
            res = rewriter.create<plier::CastOp>(loc, expectedType, res);

          rewriter.replaceOp(op, res);
          return mlir::success();
        }
      }
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, value);
      return mlir::success();
    }

    if (expectedType.isa<mlir::NoneType>()) {
      rewriter.replaceOpWithNewOp<imex::util::UndefOp>(op, expectedType);
      return mlir::success();
    }
    return mlir::failure();
  }
};

static bool isOmittedType(mlir::Type type) {
  return type.isa<plier::OmittedType>();
}

static mlir::Attribute makeSignlessAttr(mlir::Attribute val) {
  auto type = val.cast<mlir::TypedAttr>().getType();
  if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    if (!intType.isSignless()) {
      auto newType = imex::makeSignlessType(intType);
      return mlir::IntegerAttr::get(
          newType, imex::getIntAttrValue(val.cast<mlir::IntegerAttr>()));
    }
  }
  return val;
}

template <typename Op>
struct LiteralLowering : public mlir::OpConversionPattern<Op> {
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = op.getType();
    auto &converter = *(this->getTypeConverter());
    auto convertedType = converter.convertType(type);
    if (!convertedType)
      return mlir::failure();

    if (convertedType.template isa<mlir::NoneType>()) {
      rewriter.replaceOpWithNewOp<imex::util::UndefOp>(op, convertedType);
      return mlir::success();
    }

    if (auto typevar =
            convertedType.template dyn_cast<imex::util::TypeVarType>()) {
      rewriter.replaceOpWithNewOp<imex::util::UndefOp>(op, typevar);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct OmittedLowering : public mlir::OpConversionPattern<plier::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::CastOp op, plier::CastOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = op.getType();
    auto &converter = *(this->getTypeConverter());
    auto convertedType = converter.convertType(type);
    if (!convertedType)
      return mlir::failure();

    auto getOmittedValue = [&](mlir::Type type,
                               mlir::Type dstType) -> mlir::Attribute {
      if (auto attr = type.dyn_cast<plier::OmittedType>())
        return attr.getValue();

      return {};
    };

    if (auto omittedAttr =
            getOmittedValue(adaptor.getValue().getType(), convertedType)) {
      auto loc = op.getLoc();
      auto dstType = omittedAttr.cast<mlir::TypedAttr>().getType();
      auto val = makeSignlessAttr(omittedAttr);
      auto newVal =
          rewriter.create<mlir::arith::ConstantOp>(loc, val).getResult();
      if (dstType != val.cast<mlir::TypedAttr>().getType())
        newVal = rewriter.create<imex::util::SignCastOp>(loc, dstType, newVal);

      rewriter.replaceOp(op, newVal);
      return mlir::success();
    }
    return mlir::failure();
  }
};

static mlir::Value lowerConst(mlir::Location loc, mlir::OpBuilder &builder,
                              double value) {
  auto type = builder.getF64Type();
  return builder.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(value),
                                                      type);
}

static mlir::Value lowerPi(mlir::Location loc, mlir::OpBuilder &builder) {
  return lowerConst(loc, builder, M_PI);
}

static mlir::Value lowerE(mlir::Location loc, mlir::OpBuilder &builder) {
  return lowerConst(loc, builder, M_E);
}

// TODO: unhardcode
struct LowerGlobals : public mlir::OpConversionPattern<plier::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::GlobalOp op, plier::GlobalOp::Adaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    using lower_f = mlir::Value (*)(mlir::Location, mlir::OpBuilder &);
    const std::pair<llvm::StringRef, lower_f> handlers[] = {
        // clang-format off
        {"math.pi", &lowerPi},
        {"math.e", &lowerE},
        // clang-format on
    };

    mlir::Value res;
    auto name = op.getName();
    auto loc = op.getLoc();
    for (auto h : handlers) {
      if (h.first == name) {
        res = h.second(loc, rewriter);
        break;
      }
    }

    if (!res)
      return mlir::failure();

    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};

struct UndefOpLowering : public mlir::OpConversionPattern<imex::util::UndefOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::UndefOp op,
                  imex::util::UndefOp::Adaptor /*adapator*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto oldType = op.getType();
    auto &converter = *getTypeConverter();
    auto type = converter.convertType(oldType);
    if (!type || oldType == type)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<imex::util::UndefOp>(op, type);
    return mlir::success();
  }
};

static mlir::Type coerce(mlir::Type type0, mlir::Type type1) {
  // TODO: proper rules
  assert(type0 != type1);
  auto get_bits_count = [](mlir::Type type) -> unsigned {
    assert(type);
    if (type.isa<mlir::IntegerType>())
      return type.cast<mlir::IntegerType>().getWidth();

    if (type.isa<mlir::Float16Type>())
      return 11;

    if (type.isa<mlir::Float32Type>())
      return 24;

    if (type.isa<mlir::Float64Type>())
      return 53;

    llvm_unreachable("Unhandled type");
  };
  auto f0 = isFloat(type0);
  auto f1 = isFloat(type1);
  if (f0 && !f1)
    return type0;

  if (!f0 && f1)
    return type1;

  return get_bits_count(type0) < get_bits_count(type1) ? type1 : type0;
}

static mlir::Value invalidReplaceOp(mlir::PatternRewriter & /*rewriter*/,
                                    mlir::Location /*loc*/,
                                    mlir::ValueRange /*operands*/,
                                    mlir::Type /*newType*/) {
  llvm_unreachable("invalidReplaceOp");
}

template <typename T>
static mlir::Value replaceOp(mlir::PatternRewriter &rewriter,
                             mlir::Location loc, mlir::ValueRange operands,
                             mlir::Type newType) {
  auto signlessType = imex::makeSignlessType(newType);
  llvm::SmallVector<mlir::Value> newOperands(operands.size());
  for (auto it : llvm::enumerate(operands))
    newOperands[it.index()] =
        imex::doConvert(rewriter, loc, it.value(), signlessType);

  auto res = rewriter.createOrFold<T>(loc, newOperands);
  return imex::doConvert(rewriter, loc, res, newType);
}

mlir::Value replaceIpowOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::ValueRange operands, mlir::Type newType) {
  auto f64Type = rewriter.getF64Type();
  auto a = imex::doConvert(rewriter, loc, operands[0], f64Type);
  auto b = imex::doConvert(rewriter, loc, operands[1], f64Type);
  auto fres = rewriter.create<mlir::math::PowFOp>(loc, a, b).getResult();
  return imex::doConvert(rewriter, loc, fres, newType);
}

mlir::Value replaceItruedivOp(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::ValueRange operands,
                              mlir::Type newType) {
  assert(newType.isa<mlir::FloatType>());
  auto lhs = imex::doConvert(rewriter, loc, operands[0], newType);
  auto rhs = imex::doConvert(rewriter, loc, operands[1], newType);
  return rewriter.createOrFold<mlir::arith::DivFOp>(loc, lhs, rhs);
}

mlir::Value replaceIfloordivOp(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, mlir::ValueRange operands,
                               mlir::Type newType) {
  auto newIntType = newType.cast<mlir::IntegerType>();
  auto signlessType = imex::makeSignlessType(newIntType);
  auto lhs = imex::doConvert(rewriter, loc, operands[0], signlessType);
  auto rhs = imex::doConvert(rewriter, loc, operands[1], signlessType);
  mlir::Value res;
  if (newIntType.isSigned()) {
    res = rewriter.createOrFold<mlir::arith::FloorDivSIOp>(loc, lhs, rhs);
  } else {
    res = rewriter.createOrFold<mlir::arith::DivUIOp>(loc, lhs, rhs);
  }
  return imex::doConvert(rewriter, loc, res, newType);
}

mlir::Value replaceFfloordivOp(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, mlir::ValueRange operands,
                               mlir::Type newType) {
  assert(newType.isa<mlir::FloatType>());
  auto lhs = imex::doConvert(rewriter, loc, operands[0], newType);
  auto rhs = imex::doConvert(rewriter, loc, operands[1], newType);
  auto res = rewriter.createOrFold<mlir::arith::DivFOp>(loc, lhs, rhs);
  return rewriter.createOrFold<mlir::math::FloorOp>(loc, res);
}

mlir::Value replaceImodOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::ValueRange operands, mlir::Type newType) {
  auto signlessType = imex::makeSignlessType(operands[0].getType());
  auto a = imex::doConvert(rewriter, loc, operands[0], signlessType);
  auto b = imex::doConvert(rewriter, loc, operands[1], signlessType);
  auto v1 = rewriter.create<mlir::arith::RemSIOp>(loc, a, b).getResult();
  auto v2 = rewriter.create<mlir::arith::AddIOp>(loc, v1, b).getResult();
  auto res = rewriter.create<mlir::arith::RemSIOp>(loc, v2, b).getResult();
  return imex::doConvert(rewriter, loc, res, newType);
}

mlir::Value replaceFmodOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::ValueRange operands, mlir::Type /*newType*/) {
  auto a = operands[0];
  auto b = operands[1];
  auto v1 = rewriter.create<mlir::arith::RemFOp>(loc, a, b).getResult();
  auto v2 = rewriter.create<mlir::arith::AddFOp>(loc, v1, b).getResult();
  return rewriter.create<mlir::arith::RemFOp>(loc, v2, b).getResult();
}

template <mlir::arith::CmpIPredicate SignedPred,
          mlir::arith::CmpIPredicate UnsignedPred = SignedPred>
mlir::Value replaceCmpiOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::ValueRange operands, mlir::Type /*newType*/) {
  assert(operands.size() == 2);
  assert(operands[0].getType() == operands[1].getType());
  auto type = operands[0].getType().cast<mlir::IntegerType>();
  auto signlessType = imex::makeSignlessType(type);
  auto a = imex::doConvert(rewriter, loc, operands[0], signlessType);
  auto b = imex::doConvert(rewriter, loc, operands[1], signlessType);
  if (SignedPred == UnsignedPred || type.isSigned()) {
    return rewriter.createOrFold<mlir::arith::CmpIOp>(loc, SignedPred, a, b);
  } else {
    return rewriter.createOrFold<mlir::arith::CmpIOp>(loc, UnsignedPred, a, b);
  }
}

template <mlir::arith::CmpFPredicate Pred>
mlir::Value replaceCmpfOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::ValueRange operands, mlir::Type /*newType*/) {
  auto signlessType = imex::makeSignlessType(operands[0].getType());
  auto a = imex::doConvert(rewriter, loc, operands[0], signlessType);
  auto b = imex::doConvert(rewriter, loc, operands[1], signlessType);
  return rewriter.createOrFold<mlir::arith::CmpFOp>(loc, Pred, a, b);
}

struct BinOpLowering : public mlir::OpConversionPattern<plier::BinOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::BinOp op, plier::BinOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter = *getTypeConverter();
    auto operands = adaptor.getOperands();
    assert(operands.size() == 2);
    auto type0 = operands[0].getType();
    auto type1 = operands[1].getType();
    if (!isSupportedType(type0) || !isSupportedType(type1))
      return mlir::failure();

    auto resType = converter.convertType(op.getType());
    if (!resType || !isSupportedType(resType))
      return mlir::failure();

    auto loc = op.getLoc();
    auto literalCast = [&](mlir::Value val, mlir::Type dstType) -> mlir::Value {
      if (dstType != val.getType())
        return rewriter.createOrFold<plier::CastOp>(loc, dstType, val);

      return val;
    };

    std::array<mlir::Value, 2> convertedOperands = {
        literalCast(operands[0], type0), literalCast(operands[1], type1)};
    mlir::Type finalType;
    if (type0 != type1) {
      finalType = coerce(type0, type1);
      convertedOperands = {
          imex::doConvert(rewriter, loc, convertedOperands[0], finalType),
          imex::doConvert(rewriter, loc, convertedOperands[1], finalType)};
    } else {
      finalType = type0;
    }
    assert(finalType);
    assert(convertedOperands[0]);
    assert(convertedOperands[1]);

    using func_t = mlir::Value (*)(mlir::PatternRewriter &, mlir::Location,
                                   mlir::ValueRange, mlir::Type);
    struct OpDesc {
      llvm::StringRef type;
      func_t iop;
      func_t fop;
    };

    const OpDesc handlers[] = {
        {"+", &replaceOp<mlir::arith::AddIOp>, &replaceOp<mlir::arith::AddFOp>},
        {"-", &replaceOp<mlir::arith::SubIOp>, &replaceOp<mlir::arith::SubFOp>},
        {"*", &replaceOp<mlir::arith::MulIOp>, &replaceOp<mlir::arith::MulFOp>},
        {"**", &replaceIpowOp, &replaceOp<mlir::math::PowFOp>},
        {"/", &replaceItruedivOp, &replaceOp<mlir::arith::DivFOp>},
        {"//", &replaceIfloordivOp, &replaceFfloordivOp},
        {"%", &replaceImodOp, &replaceFmodOp},
        {"&", &replaceOp<mlir::arith::AndIOp>, &invalidReplaceOp},
        {"|", &replaceOp<mlir::arith::OrIOp>, &invalidReplaceOp},
        {"^", &replaceOp<mlir::arith::XOrIOp>, &invalidReplaceOp},
        {">>", &replaceOp<mlir::arith::ShRSIOp>, &invalidReplaceOp},
        {"<<", &replaceOp<mlir::arith::ShLIOp>, &invalidReplaceOp},

        {">",
         &replaceCmpiOp<mlir::arith::CmpIPredicate::sgt,
                        mlir::arith::CmpIPredicate::ugt>,
         &replaceCmpfOp<mlir::arith::CmpFPredicate::OGT>},
        {">=",
         &replaceCmpiOp<mlir::arith::CmpIPredicate::sge,
                        mlir::arith::CmpIPredicate::uge>,
         &replaceCmpfOp<mlir::arith::CmpFPredicate::OGE>},
        {"<",
         &replaceCmpiOp<mlir::arith::CmpIPredicate::slt,
                        mlir::arith::CmpIPredicate::ult>,
         &replaceCmpfOp<mlir::arith::CmpFPredicate::OLT>},
        {"<=",
         &replaceCmpiOp<mlir::arith::CmpIPredicate::sle,
                        mlir::arith::CmpIPredicate::ule>,
         &replaceCmpfOp<mlir::arith::CmpFPredicate::OLE>},
        {"!=", &replaceCmpiOp<mlir::arith::CmpIPredicate::ne>,
         &replaceCmpfOp<mlir::arith::CmpFPredicate::ONE>},
        {"==", &replaceCmpiOp<mlir::arith::CmpIPredicate::eq>,
         &replaceCmpfOp<mlir::arith::CmpFPredicate::OEQ>},
    };

    using membptr_t = func_t OpDesc::*;
    auto callHandler = [&](membptr_t mem) {
      for (auto &h : handlers) {
        if (h.type == op.getOp()) {
          auto res = (h.*mem)(rewriter, loc, convertedOperands, resType);
          if (res.getType() != resType)
            res = rewriter.createOrFold<imex::util::SignCastOp>(loc, resType,
                                                                res);
          rewriter.replaceOp(op, res);
          return mlir::success();
        }
      }
      return mlir::failure();
    };

    if (isInt(finalType)) {
      return callHandler(&OpDesc::iop);
    } else if (isFloat(finalType)) {
      return callHandler(&OpDesc::fop);
    }
    return mlir::failure();
  }
};

struct BinOpTupleLowering : public mlir::OpConversionPattern<plier::BinOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::BinOp op, plier::BinOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto lhsType = lhs.getType().dyn_cast<mlir::TupleType>();
    if (!lhsType)
      return mlir::failure();

    auto loc = op->getLoc();
    if (adaptor.getOp() == "+") {
      auto rhsType = rhs.getType().dyn_cast<mlir::TupleType>();
      if (!rhsType)
        return mlir::failure();

      auto count = lhsType.size() + rhsType.size();
      llvm::SmallVector<mlir::Value> newArgs;
      llvm::SmallVector<mlir::Type> newTypes;
      newArgs.reserve(count);
      newTypes.reserve(count);

      for (auto &arg : {lhs, rhs}) {
        auto type = arg.getType().cast<mlir::TupleType>();
        for (auto i : llvm::seq<size_t>(0, type.size())) {
          auto elemType = type.getType(i);
          auto ind = rewriter.create<mlir::arith::ConstantIndexOp>(
              loc, static_cast<int64_t>(i));
          auto elem = rewriter.create<imex::util::TupleExtractOp>(loc, elemType,
                                                                  arg, ind);
          newArgs.emplace_back(elem);
          newTypes.emplace_back(elemType);
        }
      }

      auto newTupleType = mlir::TupleType::get(getContext(), newTypes);
      rewriter.replaceOpWithNewOp<imex::util::BuildTupleOp>(op, newTupleType,
                                                            newArgs);
      return mlir::success();
    }

    return mlir::failure();
  }
};

static mlir::Value negate(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Value val, mlir::Type resType) {
  val = imex::doConvert(rewriter, loc, val, resType);
  if (auto itype = resType.dyn_cast<mlir::IntegerType>()) {
    auto signless = imex::makeSignlessType(resType);
    if (signless != itype)
      val = rewriter.create<imex::util::SignCastOp>(loc, signless, val);

    // TODO: no int negation?
    auto zero = rewriter.create<mlir::arith::ConstantOp>(
        loc, mlir::IntegerAttr::get(signless, 0));
    auto res = rewriter.create<mlir::arith::SubIOp>(loc, zero, val).getResult();
    if (signless != itype)
      res = rewriter.create<imex::util::SignCastOp>(loc, itype, res);

    return res;
  }
  if (resType.isa<mlir::FloatType>())
    return rewriter.create<mlir::arith::NegFOp>(loc, val);

  llvm_unreachable("negate: unsupported type");
}

static mlir::Value unaryPlus(mlir::PatternRewriter &rewriter,
                             mlir::Location loc, mlir::Value arg,
                             mlir::Type resType) {
  return imex::doConvert(rewriter, loc, arg, resType);
}

static mlir::Value unaryMinus(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::Value arg,
                              mlir::Type resType) {
  return negate(rewriter, loc, arg, resType);
}

static mlir::Value unaryNot(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::Value arg, mlir::Type resType) {
  auto i1 = rewriter.getIntegerType(1);
  auto casted = imex::doConvert(rewriter, loc, arg, i1);
  auto one = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, i1);
  return rewriter.create<mlir::arith::SubIOp>(loc, one, casted);
}

static mlir::Value unaryInvert(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, mlir::Value arg,
                               mlir::Type resType) {
  auto intType = arg.getType().dyn_cast<mlir::IntegerType>();
  if (!intType)
    return {};

  mlir::Type signlessType;
  if (intType.getWidth() == 1) {
    intType = rewriter.getIntegerType(64);
    signlessType = intType;
    arg = rewriter.create<mlir::arith::ExtUIOp>(loc, intType, arg);
  } else {
    signlessType = imex::makeSignlessType(intType);
    if (intType != signlessType)
      arg = rewriter.create<imex::util::SignCastOp>(loc, signlessType, arg);
  }

  auto all = rewriter.create<mlir::arith::ConstantIntOp>(loc, -1, signlessType);

  arg = rewriter.create<mlir::arith::XOrIOp>(loc, all, arg);

  if (intType != signlessType)
    arg = rewriter.create<imex::util::SignCastOp>(loc, intType, arg);

  if (resType != arg.getType())
    arg = imex::doConvert(rewriter, loc, arg, resType);

  return arg;
}

struct UnaryOpLowering : public mlir::OpConversionPattern<plier::UnaryOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::UnaryOp op, plier::UnaryOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter = *getTypeConverter();
    auto arg = adaptor.getValue();
    auto type = arg.getType();
    if (!isSupportedType(type))
      return mlir::failure();

    auto resType = converter.convertType(op.getType());
    if (!resType)
      return mlir::failure();

    using func_t = mlir::Value (*)(mlir::PatternRewriter &, mlir::Location,
                                   mlir::Value, mlir::Type);

    using Handler = std::pair<llvm::StringRef, func_t>;
    const Handler handlers[] = {
        {"+", &unaryPlus},
        {"-", &unaryMinus},
        {"not", &unaryNot},
        {"~", &unaryInvert},
    };

    auto opname = op.getOp();
    for (auto &h : handlers) {
      if (h.first == opname) {
        auto loc = op.getLoc();
        auto res = h.second(rewriter, loc, arg, resType);
        if (!res)
          return mlir::failure();

        rewriter.replaceOp(op, res);
        return mlir::success();
      }
    }

    return mlir::failure();
  }
};

struct LowerCasts : public mlir::OpConversionPattern<plier::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::CastOp op, plier::CastOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter = *getTypeConverter();
    auto src = adaptor.getValue();
    auto dstType = converter.convertType(op.getType());
    if (!dstType)
      return mlir::failure();

    auto srcType = src.getType();
    if (srcType == dstType) {
      rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(op, dstType,
                                                                    src);
      return mlir::success();
    }

    auto res = imex::doConvert(rewriter, op.getLoc(), src, dstType);
    if (!res)
      return mlir::failure();

    rewriter.replaceOp(op, res);

    return mlir::success();
  }
};

static void rerunScfPipeline(mlir::Operation *op) {
  assert(nullptr != op);
  auto marker =
      mlir::StringAttr::get(op->getContext(), plierToScfPipelineName());
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  assert(nullptr != mod);
  imex::addPipelineJumpMarker(mod, marker);
}

static mlir::LogicalResult
lowerSlice(plier::PyCallOp op, mlir::ValueRange operands,
           llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs,
           mlir::PatternRewriter &rewriter) {
  if (!kwargs.empty())
    return mlir::failure();

  if (operands.size() != 2 && operands.size() != 3)
    return mlir::failure();

  if (llvm::any_of(operands, [](mlir::Value op) {
        return !op.getType()
                    .isa<mlir::IntegerType, mlir::IndexType, mlir::NoneType>();
      }))
    return mlir::failure();

  auto begin = operands[0];
  auto end = operands[1];
  auto stride = [&]() -> mlir::Value {
    if (operands.size() == 3)
      return operands[2];

    return rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 1);
  }();

  rerunScfPipeline(op);
  rewriter.replaceOpWithNewOp<plier::BuildSliceOp>(op, begin, end, stride);
  return mlir::success();
}

static mlir::LogicalResult
lowerRangeImpl(plier::PyCallOp op, mlir::ValueRange operands,
               llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs,
               mlir::PatternRewriter &rewriter) {
  auto parent = op->getParentOp();
  auto res = imex::lowerRange(op, operands, kwargs, rewriter);
  if (mlir::succeeded(res))
    rerunScfPipeline(parent);

  return res;
}

using kwargs_t = llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>>;
using func_t = mlir::LogicalResult (*)(plier::PyCallOp, mlir::ValueRange,
                                       kwargs_t, mlir::PatternRewriter &);
static const std::pair<llvm::StringRef, func_t> builtinFuncsHandlers[] = {
    // clang-format off
    {"range", &lowerRangeImpl},
    {"slice", &lowerSlice},
    // clang-format on
};

struct BuiltinCallsLowering final : public imex::CallOpLowering {
  BuiltinCallsLowering(mlir::MLIRContext *context)
      : CallOpLowering(context),
        resolver("numba_dpcomp.mlir.builtin.funcs", "registry") {}

protected:
  virtual mlir::LogicalResult
  resolveCall(plier::PyCallOp op, mlir::StringRef name, mlir::Location loc,
              mlir::PatternRewriter &rewriter, mlir::ValueRange args,
              KWargs kwargs) const override {
    for (auto &handler : builtinFuncsHandlers)
      if (handler.first == name)
        return handler.second(op, args, kwargs, rewriter);

    auto res = resolver.rewriteFunc(name, loc, rewriter, args, kwargs);
    if (!res)
      return mlir::failure();

    auto results = std::move(res).value();
    assert(results.size() == op->getNumResults());
    for (auto it : llvm::enumerate(results)) {
      auto i = it.index();
      auto r = it.value();
      auto dstType = op->getResultTypes()[i];
      if (dstType != r.getType())
        results[i] = rewriter.create<plier::CastOp>(loc, dstType, r);
    }

    rerunScfPipeline(op);
    rewriter.replaceOp(op, results);
    return mlir::success();
  }

private:
  PyLinalgResolver resolver;
};

struct ExternalCallsLowering final : public imex::CallOpLowering {
  using CallOpLowering::CallOpLowering;

protected:
  virtual mlir::LogicalResult
  resolveCall(plier::PyCallOp op, mlir::StringRef name, mlir::Location loc,
              mlir::PatternRewriter &rewriter, mlir::ValueRange args,
              KWargs kwargs) const override {
    if (!kwargs.empty())
      return mlir::failure(); // TODO: kwargs support

    auto types = args.getTypes();
    auto mangledName = mangle(name, types);
    if (mangledName.empty())
      return mlir::failure();

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    assert(mod);
    auto externalFunc = mod.lookupSymbol<mlir::func::FuncOp>(mangledName);
    if (!externalFunc) {
      externalFunc = resolver.getFunc(name, types);
      if (externalFunc) {
        externalFunc.setPrivate();
        externalFunc.setName(mangledName);
      }
    }
    if (!externalFunc)
      return mlir::failure();

    assert(externalFunc.getFunctionType().getNumResults() ==
           op->getNumResults());

    llvm::SmallVector<mlir::Value> castedArgs(args.size());
    auto funcTypes = externalFunc.getFunctionType().getInputs();
    for (auto it : llvm::enumerate(args)) {
      auto arg = it.value();
      auto i = it.index();
      auto dstType = funcTypes[i];
      if (arg.getType() != dstType)
        castedArgs[i] = rewriter.createOrFold<plier::CastOp>(loc, dstType, arg);
      else
        castedArgs[i] = arg;
    }

    auto newFuncCall =
        rewriter.create<mlir::func::CallOp>(loc, externalFunc, castedArgs);

    auto results = newFuncCall.getResults();
    llvm::SmallVector<mlir::Value> castedResults(results.size());

    for (auto it : llvm::enumerate(results)) {
      auto i = static_cast<unsigned>(it.index());
      auto res = it.value();
      auto oldResType = op->getResult(i).getType();
      if (res.getType() != oldResType)
        castedResults[i] =
            rewriter.createOrFold<plier::CastOp>(loc, oldResType, res);
      else
        castedResults[i] = res;
    }

    rerunScfPipeline(op);
    rewriter.replaceOp(op, castedResults);
    return mlir::success();
  }

private:
  PyFuncResolver resolver;
};

struct BuiltinCallsLoweringPass
    : public imex::RewriteWrapperPass<
          BuiltinCallsLoweringPass, void, void, BuiltinCallsLowering,
          imex::ExpandCallVarargs, ExternalCallsLowering> {};

struct PlierToStdPass
    : public mlir::PassWrapper<PlierToStdPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlierToStdPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::util::ImexUtilDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<plier::PlierDialect>();
  }

  void runOnOperation() override;
};

struct BuildTupleConversionPattern
    : public mlir::OpConversionPattern<plier::BuildTupleOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::BuildTupleOp op, plier::BuildTupleOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto converter = getTypeConverter();
    assert(converter);
    auto retType = converter->convertType(op.getResult().getType());
    if (!retType.isa_and_nonnull<mlir::TupleType>())
      return mlir::failure();

    rewriter.replaceOpWithNewOp<imex::util::BuildTupleOp>(op, retType,
                                                          adaptor.getArgs());
    return mlir::success();
  }
};

struct GetItemTupleConversionPattern
    : public mlir::OpConversionPattern<plier::GetItemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::GetItemOp op, plier::GetItemOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto container = adaptor.getValue();
    auto containerType = container.getType().dyn_cast<mlir::TupleType>();
    if (!containerType)
      return mlir::failure();

    auto &converter = *getTypeConverter();

    auto retType = converter.convertType(op.getType());
    if (!retType)
      return mlir::failure();

    auto index = imex::indexCast(rewriter, op->getLoc(), adaptor.getIndex());

    rewriter.replaceOpWithNewOp<imex::util::TupleExtractOp>(op, retType,
                                                            container, index);
    return mlir::success();
  }
};

void PlierToStdPass::runOnOperation() {
  mlir::TypeConverter typeConverter;
  // Convert unknown types to itself
  typeConverter.addConversion([](mlir::Type type) { return type; });

  auto context = &getContext();
  typeConverter.addConversion(
      [](mlir::Type type, llvm::SmallVectorImpl<mlir::Type> &retTypes)
          -> llvm::Optional<mlir::LogicalResult> {
        if (isOmittedType(type))
          return mlir::success();

        return llvm::None;
      });
  imex::populateTupleTypeConverter(*context, typeConverter);

  auto materializeCast = [](mlir::OpBuilder &builder, mlir::Type type,
                            mlir::ValueRange inputs,
                            mlir::Location loc) -> llvm::Optional<mlir::Value> {
    if (inputs.size() == 1)
      return builder
          .create<mlir::UnrealizedConversionCastOp>(loc, type, inputs.front())
          .getResult(0);

    return llvm::None;
  };
  typeConverter.addArgumentMaterialization(materializeCast);
  typeConverter.addSourceMaterialization(materializeCast);
  typeConverter.addTargetMaterialization(materializeCast);

  mlir::RewritePatternSet patterns(context);
  mlir::ConversionTarget target(*context);

  auto isNum = [&](mlir::Type t) -> bool {
    if (!t)
      return false;

    auto res = typeConverter.convertType(t);
    return res &&
           res.isa<mlir::IntegerType, mlir::FloatType, mlir::IndexType>();
  };

  auto isTuple = [&](mlir::Type t) -> bool {
    if (!t)
      return false;

    auto res = typeConverter.convertType(t);
    return res && res.isa<mlir::TupleType>();
  };

  target.addDynamicallyLegalOp<plier::BinOp>([&](plier::BinOp op) {
    auto lhsType = op.getLhs().getType();
    auto rhsType = op.getRhs().getType();
    if (op.getOp() == "+" && isTuple(lhsType) && isTuple(rhsType))
      return false;

    return !isNum(lhsType) || !isNum(rhsType) || !isNum(op.getType());
  });
  target.addDynamicallyLegalOp<plier::UnaryOp>([&](plier::UnaryOp op) {
    return !isNum(op.getValue().getType()) && !isNum(op.getType());
  });
  target.addDynamicallyLegalOp<plier::CastOp>([&](plier::CastOp op) {
    auto inputType = op.getValue().getType();
    if (isOmittedType(inputType))
      return false;

    auto srcType = typeConverter.convertType(inputType);
    auto dstType = typeConverter.convertType(op.getType());
    if (srcType == dstType && inputType != op.getType())
      return false;

    return srcType == dstType || !isNum(srcType) || !isNum(dstType);
  });
  target.addDynamicallyLegalOp<plier::ConstOp, plier::GlobalOp>(
      [&](mlir::Operation *op) {
        auto type = typeConverter.convertType(op->getResult(0).getType());
        if (!type)
          return true;

        if (type.isa<mlir::NoneType, imex::util::TypeVarType>())
          return false;

        return !type.isIntOrFloat();
      });
  target.addDynamicallyLegalOp<imex::util::UndefOp>(
      [&](imex::util::UndefOp op) {
        auto srcType = op.getType();
        auto dstType = typeConverter.convertType(srcType);
        return srcType == dstType;
      });

  target.addDynamicallyLegalOp<plier::GetItemOp>(
      [&](plier::GetItemOp op) -> llvm::Optional<bool> {
        auto type = typeConverter.convertType(op.getValue().getType());
        if (type.isa_and_nonnull<mlir::TupleType>())
          return false;

        return llvm::None;
      });
  target.addIllegalOp<plier::BuildTupleOp>();
  target.addLegalOp<imex::util::BuildTupleOp, imex::util::TupleExtractOp>();

  patterns.insert<
      // clang-format off
      BinOpLowering,
      BinOpTupleLowering,
      UnaryOpLowering,
      LowerCasts,
      ConstOpLowering,
      LiteralLowering<plier::CastOp>,
      LiteralLowering<plier::GlobalOp>,
      OmittedLowering,
      LowerGlobals,
      UndefOpLowering,
      BuildTupleConversionPattern,
      GetItemTupleConversionPattern
      // clang-format on
      >(typeConverter, context);

  imex::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                           patterns, target);
  imex::populateTupleTypeConversionRewritesAndTarget(typeConverter, patterns,
                                                     target);

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns))))
    signalPassFailure();
}

static void populatePlierToStdPipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<PlierToStdPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<BuiltinCallsLoweringPass>());
  pm.addPass(imex::createForceInlinePass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}
} // namespace

void registerPlierToStdPipeline(imex::PipelineRegistry &registry) {
  registry.registerPipeline([](auto sink) {
    auto stage = getHighLoweringStage();
    sink(plierToStdPipelineName(), {plierToScfPipelineName()}, {stage.end},
         {plierToScfPipelineName()}, &populatePlierToStdPipeline);
  });
}

llvm::StringRef plierToStdPipelineName() { return "plier_to_std"; }
