// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/CastUtils.hpp"
#include "imex/Dialect/imex_util/Dialect.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>

namespace {
mlir::Type makeSignless(mlir::Type type) {
  if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    if (!intType.isSignless()) {
      return mlir::IntegerType::get(intType.getContext(), intType.getWidth());
    }
  }
  return type;
}
} // namespace

mlir::Value imex::indexCast(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value val, mlir::Type dstType) {
  auto srcType = val.getType();
  assert(srcType.isa<mlir::IndexType>() || dstType.isa<mlir::IndexType>());
  if (srcType == dstType)
    return val;

  auto newSrcType = makeSignless(srcType);
  if (newSrcType != srcType)
    val = builder.createOrFold<imex::util::SignCastOp>(loc, newSrcType, val);

  auto newDstType = makeSignless(dstType);
  val = builder.createOrFold<mlir::arith::IndexCastOp>(loc, newDstType, val);
  if (newDstType != dstType)
    val = builder.createOrFold<imex::util::SignCastOp>(loc, dstType, val);

  return val;
}

mlir::Value imex::indexCast(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value src) {
  return indexCast(builder, loc, src,
                   mlir::IndexType::get(builder.getContext()));
}

mlir::Type imex::makeSignlessType(mlir::Type type) {
  if (auto intType = type.dyn_cast<mlir::IntegerType>())
    return makeSignlessType(intType);

  return type;
}

mlir::IntegerType imex::makeSignlessType(mlir::IntegerType type) {
  if (!type.isSignless())
    return mlir::IntegerType::get(type.getContext(), type.getWidth());

  return type;
}

static bool isInt(mlir::Type type) {
  assert(type);
  return type.isa<mlir::IntegerType>();
}

static bool isFloat(mlir::Type type) {
  assert(type);
  return type.isa<mlir::FloatType>();
}

static bool isIndex(mlir::Type type) {
  assert(type);
  return type.isa<mlir::IndexType>();
}

static mlir::Value intCast(mlir::OpBuilder &rewriter, mlir::Location loc,
                           mlir::Value val, mlir::Type dstType) {
  auto srcIntType = val.getType().cast<mlir::IntegerType>();
  auto dstIntType = dstType.cast<mlir::IntegerType>();
  auto srcSignless = imex::makeSignlessType(srcIntType);
  auto dstSignless = imex::makeSignlessType(dstIntType);
  auto srcBits = srcIntType.getWidth();
  auto dstBits = dstIntType.getWidth();

  if (srcIntType != srcSignless)
    val = rewriter.createOrFold<imex::util::SignCastOp>(loc, srcSignless, val);

  if (dstBits > srcBits) {
    if (srcIntType.isSigned()) {
      val = rewriter.createOrFold<mlir::arith::ExtSIOp>(loc, dstSignless, val);
    } else {
      val = rewriter.createOrFold<mlir::arith::ExtUIOp>(loc, dstSignless, val);
    }
  } else if (dstBits < srcBits) {
    if (dstBits == 1) {
      // Special handling for bool
      auto zero = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, srcBits);
      auto cmp = rewriter.createOrFold<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, val, zero);
      auto trueVal = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      auto falseVal = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 1);
      val = rewriter.createOrFold<mlir::arith::SelectOp>(loc, cmp, falseVal,
                                                         trueVal);
    } else {
      val = rewriter.createOrFold<mlir::arith::TruncIOp>(loc, dstSignless, val);
    }
  }

  if (dstIntType != dstSignless)
    val = rewriter.createOrFold<imex::util::SignCastOp>(loc, dstIntType, val);

  return val;
}

static mlir::Value intFloatCast(mlir::OpBuilder &rewriter, mlir::Location loc,
                                mlir::Value val, mlir::Type dstType) {
  auto srcIntType = val.getType().cast<mlir::IntegerType>();
  auto signlessType = imex::makeSignlessType(srcIntType);
  if (val.getType() != signlessType)
    val = rewriter.createOrFold<imex::util::SignCastOp>(loc, signlessType, val);

  if (srcIntType.isSigned()) {
    return rewriter.createOrFold<mlir::arith::SIToFPOp>(loc, dstType, val);
  } else {
    return rewriter.createOrFold<mlir::arith::UIToFPOp>(loc, dstType, val);
  }
}

static mlir::Value floatIntCast(mlir::OpBuilder &rewriter, mlir::Location loc,
                                mlir::Value val, mlir::Type dstType) {
  auto dstIntType = dstType.cast<mlir::IntegerType>();
  mlir::Value res;
  auto dstSignlessType = imex::makeSignlessType(dstIntType);
  if (dstIntType.getWidth() == 1) {
    // Special handling for bool
    auto floatType = val.getType().cast<mlir::FloatType>();
    auto getZeroFloat = [&]() -> llvm::APFloat {
      if (floatType.isF64())
        return llvm::APFloat(0.0);
      if (floatType.isF32())
        return llvm::APFloat(0.0f);
      llvm_unreachable("Umhandled float type");
    };
    auto zero = rewriter.create<mlir::arith::ConstantFloatOp>(
        loc, getZeroFloat(), floatType);
    auto cmp = rewriter.createOrFold<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OEQ, val, zero);
    auto trueVal = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    auto falseVal = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 1);
    res = rewriter.createOrFold<mlir::arith::SelectOp>(loc, cmp, falseVal,
                                                       trueVal);
  } else if (dstIntType.isSigned()) {
    res = rewriter.create<mlir::arith::FPToSIOp>(loc, dstSignlessType, val);
  } else {
    res = rewriter.create<mlir::arith::FPToUIOp>(loc, dstSignlessType, val);
  }
  if (dstSignlessType != dstIntType) {
    return rewriter.createOrFold<imex::util::SignCastOp>(loc, dstIntType, res);
  }
  return res;
}

static mlir::Value indexCastImpl(mlir::OpBuilder &rewriter, mlir::Location loc,
                                 mlir::Value val, mlir::Type dstType) {
  if (val.getType().isa<mlir::FloatType>()) {
    auto intType = rewriter.getI64Type();
    val = rewriter.createOrFold<mlir::arith::FPToSIOp>(loc, intType, val);
  }
  if (dstType.isa<mlir::FloatType>()) {
    auto intType = rewriter.getI64Type();
    val = imex::indexCast(rewriter, loc, val, intType);
    return rewriter.createOrFold<mlir::arith::SIToFPOp>(loc, dstType, val);
  }
  return imex::indexCast(rewriter, loc, val, dstType);
}

static mlir::Value floatCastImpl(mlir::OpBuilder &rewriter, mlir::Location loc,
                                 mlir::Value val, mlir::Type dstType) {
  auto srcFloatType = val.getType().cast<mlir::FloatType>();
  auto dstFloatType = dstType.cast<mlir::FloatType>();
  assert(srcFloatType != dstFloatType);
  if (dstFloatType.getWidth() > srcFloatType.getWidth()) {
    return rewriter.createOrFold<mlir::arith::ExtFOp>(loc, dstFloatType, val);
  } else {
    return rewriter.createOrFold<mlir::arith::TruncFOp>(loc, dstFloatType, val);
  }
}

struct CastHandler {
  using selector_t = bool (*)(mlir::Type);
  using cast_op_t = mlir::Value (*)(mlir::OpBuilder &, mlir::Location,
                                    mlir::Value, mlir::Type);
  selector_t src;
  selector_t dst;
  cast_op_t cast_op;
};

static const CastHandler castHandlers[] = {
    {&isInt, &isInt, &intCast},           {&isInt, &isFloat, &intFloatCast},
    {&isFloat, &isInt, &floatIntCast},    {&isIndex, &isInt, &indexCastImpl},
    {&isInt, &isIndex, &indexCastImpl},   {&isFloat, &isFloat, &floatCastImpl},
    {&isIndex, &isFloat, &indexCastImpl}, {&isFloat, &isIndex, &indexCastImpl},
};

bool imex::canConvert(mlir::Type srcType, mlir::Type dstType) {
  if (srcType == dstType)
    return true;

  for (auto &h : castHandlers)
    if (h.src(srcType) && h.dst(dstType))
      return true;

  return false;
}

mlir::Value imex::doConvert(mlir::OpBuilder &rewriter, mlir::Location loc,
                            mlir::Value val, mlir::Type dstType) {
  assert(dstType);
  auto srcType = val.getType();
  if (srcType == dstType)
    return val;

  for (auto &h : castHandlers)
    if (h.src(srcType) && h.dst(dstType))
      return h.cast_op(rewriter, loc, val, dstType);

  return nullptr;
}
