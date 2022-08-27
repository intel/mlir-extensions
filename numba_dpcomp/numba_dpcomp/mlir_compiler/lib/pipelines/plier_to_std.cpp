// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define _USE_MATH_DEFINES
#include <cmath>

#include "pipelines/plier_to_scf.hpp"
#include "pipelines/plier_to_std.hpp"
#include "pipelines/pre_low_simplifications.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
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

#include "mlir-extensions/Dialect/plier/dialect.hpp"
#include "mlir-extensions/Dialect/plier_util/dialect.hpp"

#include "mlir-extensions/Transforms/call_lowering.hpp"
#include "mlir-extensions/Transforms/cast_utils.hpp"
#include "mlir-extensions/Transforms/const_utils.hpp"
#include "mlir-extensions/Transforms/inline_utils.hpp"
#include "mlir-extensions/Transforms/pipeline_utils.hpp"
#include "mlir-extensions/Transforms/rewrite_wrapper.hpp"
#include "mlir-extensions/Transforms/type_conversion.hpp"

#include "base_pipeline.hpp"
#include "loop_utils.hpp"
#include "mangle.hpp"
#include "mlir-extensions/compiler/pipeline_registry.hpp"
#include "py_func_resolver.hpp"
#include "py_linalg_resolver.hpp"

namespace {
static mlir::Type mapIntType(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  unsigned numBits = 0;
  if (name.consume_front("int") && !name.consumeInteger<unsigned>(10, numBits))
    return mlir::IntegerType::get(&ctx, numBits, mlir::IntegerType::Signed);

  if (name.consume_front("uint") && !name.consumeInteger<unsigned>(10, numBits))
    return mlir::IntegerType::get(&ctx, numBits, mlir::IntegerType::Unsigned);

  return nullptr;
}

static mlir::Type mapIntLiteralType(mlir::MLIRContext &ctx,
                                    llvm::StringRef &name) {
  int64_t value = 0;
  if (name.consume_front("Literal[int](") &&
      !name.consumeInteger<int64_t>(10, value) && name.consume_front(")")) {
    auto type = mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Signed);
    auto attr = mlir::IntegerAttr::get(type, value);
    return plier::LiteralType::get(attr);
  }
  return nullptr;
}

static mlir::Type mapBoolLiteralType(mlir::MLIRContext &ctx,
                                     llvm::StringRef &name) {
  if (name.consume_front("Literal[bool](")) {
    auto type = mlir::IntegerType::get(&ctx, 1);
    mlir::IntegerAttr attr;
    if (name.consume_front("True") && name.consume_front(")")) {
      attr = mlir::IntegerAttr::get(type, 1);
    } else if (name.consume_front("False") && name.consume_front(")")) {
      attr = mlir::IntegerAttr::get(type, 0);
    } else {
      return nullptr;
    }
    return plier::LiteralType::get(attr);
  }
  return nullptr;
}

static mlir::Type mapBoolType(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  if (name.consume_front("bool"))
    return mlir::IntegerType::get(&ctx, 1);

  return nullptr;
}

static mlir::Type mapFloatType(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  unsigned numBits = 0;
  if (name.consume_front("float") &&
      !name.consumeInteger<unsigned>(10, numBits)) {
    switch (numBits) {
    case 64:
      return mlir::Float64Type::get(&ctx);
    case 32:
      return mlir::Float32Type::get(&ctx);
    case 16:
      return mlir::Float16Type::get(&ctx);
    }
  }
  return nullptr;
}

static bool consumeUntil(llvm::StringRef &name, llvm::StringRef end) {
  while (!name.empty()) {
    if (name.consume_front(end))
      return true;

    const std::pair<const char *, const char *> pairs[] = {
        // clang-format off
            {"(",")"},
            {"[","]"},
            {"<",">"},
            {"{","}"},
        // clang-format on
    };

    bool consumed = false;
    for (auto it : pairs) {
      if (name.consume_front(it.first)) {
        consumed = true;
        if (!consumeUntil(name, it.second))
          return false;
      }
    }

    if (!consumed)
      name = name.drop_front();
  }
  return false;
}

static mlir::Type mapPlierTypeName(mlir::MLIRContext &ctx,
                                   llvm::StringRef &name);
static bool mapTypeHelper(mlir::MLIRContext &ctx, llvm::StringRef &name,
                          mlir::Type &ret, llvm::StringRef end) {
  auto nameCopy = name;
  auto type = mapPlierTypeName(ctx, nameCopy);
  if (type && nameCopy.consume_front(end)) {
    ret = type;
    name = nameCopy;
    return true;
  }
  nameCopy = name;
  if (consumeUntil(nameCopy, end)) {
    auto len = name.size() - nameCopy.size() - end.size();
    ret = plier::PyType::get(&ctx, name.take_front(len));
    name = nameCopy;
    return true;
  }
  return false;
}

static mlir::Type mapPairType(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  mlir::Type first;
  mlir::Type second;
  if (name.consume_front("pair<") && mapTypeHelper(ctx, name, first, ", ") &&
      mapTypeHelper(ctx, name, second, ">")) {
    return mlir::TupleType::get(&ctx, {first, second});
  }
  return nullptr;
}

static mlir::Type mapUnitupleType(mlir::MLIRContext &ctx,
                                  llvm::StringRef &name) {
  mlir::Type type;
  unsigned count = 0;
  if (name.consume_front("UniTuple(") &&
      mapTypeHelper(ctx, name, type, " x ") &&
      !name.consumeInteger<unsigned>(10, count) && name.consume_front(")")) {
    llvm::SmallVector<mlir::Type> types(count, type);
    return mlir::TupleType::get(&ctx, types);
  }
  return nullptr;
}

static mlir::Type mapTupleType(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  if (!name.consume_front("Tuple("))
    return nullptr;

  if (name.consume_front(")"))
    return mlir::TupleType::get(&ctx, llvm::None);

  llvm::SmallVector<mlir::Type> types;
  auto temp = name;
  if (!consumeUntil(temp, ")"))
    return nullptr;
  auto len = name.size() - temp.size();
  temp = name.take_front(len);
  while (true) {
    mlir::Type type;
    if (mapTypeHelper(ctx, temp, type, ", ")) {
      types.push_back(type);
      continue;
    }
    if (mapTypeHelper(ctx, temp, type, ")")) {
      types.push_back(type);
      break;
    }
  }
  name = name.drop_front(len);
  return mlir::TupleType::get(&ctx, types);
}

static mlir::Type mapFuncType(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  if (name.consume_front("Function(") &&
      name.consume_front("<class 'bool'>") && // TODO unhardcode;
      name.consume_front(")"))
    return mlir::FunctionType::get(&ctx, {}, {});

  return nullptr;
}

static mlir::Type mapDtypeType(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  if (name.consume_front("dtype(") && name.consume_back(")")) {
    auto innerType = mapPlierTypeName(ctx, name);
    if (innerType)
      return plier::TypeVar::get(innerType);
  } else if (name.consume_front("class(") && name.consume_back(")")) {
    auto innerType = mapPlierTypeName(ctx, name);
    if (innerType)
      return plier::TypeVar::get(innerType);
  }
  return nullptr;
}

static mlir::Type mapNoneType(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  if (name.consume_front("none"))
    return mlir::NoneType::get(&ctx);

  return nullptr;
}

static mlir::Type mapDispatcherType(mlir::MLIRContext &ctx,
                                    llvm::StringRef &name) {
  if (name.consume_front("type(CPUDispatcher(") && consumeUntil(name, "))"))
    return imex::util::OpaqueType::get(&ctx);

  return nullptr;
}

static mlir::Type mapSliceType(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  if (name.consume_front("slice<") && consumeUntil(name, ">"))
    return plier::SliceType::get(&ctx);

  return nullptr;
}

static mlir::Type mapPlierTypeName(mlir::MLIRContext &ctx,
                                   llvm::StringRef &name) {
  using func_t =
      mlir::Type (*)(mlir::MLIRContext & ctx, llvm::StringRef & name);
  const func_t handlers[] = {
      // clang-format off
      &mapIntType,
      &mapIntLiteralType,
      &mapBoolLiteralType,
      &mapBoolType,
      &mapFloatType,
      &mapPairType,
      &mapUnitupleType,
      &mapTupleType,
      &mapFuncType,
      &mapDtypeType,
      &mapNoneType,
      &mapDispatcherType,
      &mapSliceType,
      // clang-format on
  };
  for (auto h : handlers) {
    auto temp_name = name;
    auto t = h(ctx, temp_name);
    if (static_cast<bool>(t)) {
      name = temp_name;
      return t;
    }
  }
  return nullptr;
}

static mlir::Type mapPlierType(mlir::Type type) {
  assert(type);
  if (!type.isa<plier::PyType>())
    return type;

  auto name = type.cast<plier::PyType>().getName();
  return mapPlierTypeName(*type.getContext(), name);
}

static mlir::Type dropLiteralType(mlir::Type t) {
  assert(t);
  if (auto literal = t.dyn_cast<plier::LiteralType>())
    return dropLiteralType(
        literal.getValue().cast<mlir::TypedAttr>().getType());

  return t;
}

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

static bool isIndex(mlir::Type type) {
  assert(type);
  return type.isa<mlir::IndexType>();
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

    auto value = adaptor.val();
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
  if (auto pytype = type.dyn_cast<plier::PyType>()) {
    auto name = pytype.getName();
    if (name.consume_front("omitted(") && name.consume_back(")")) {
      return true;
    }
  }
  return false;
}

static mlir::Attribute makeSignlessAttr(mlir::Attribute val) {
  auto type = val.cast<mlir::TypedAttr>().getType();
  if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    if (!intType.isSignless()) {
      auto newType = plier::makeSignlessType(intType);
      return mlir::IntegerAttr::get(
          newType, plier::getIntAttrValue(val.cast<mlir::IntegerAttr>()));
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

    if (auto literal = convertedType.template dyn_cast<plier::LiteralType>()) {
      auto loc = op.getLoc();
      auto attrVal = literal.getValue();
      auto dstType = attrVal.template cast<mlir::TypedAttr>().getType();
      auto val = makeSignlessAttr(attrVal).template cast<mlir::TypedAttr>();
      auto newVal =
          rewriter.create<mlir::arith::ConstantOp>(loc, val).getResult();
      if (dstType != val.getType())
        newVal = rewriter.create<imex::util::SignCastOp>(loc, dstType, newVal);

      rewriter.replaceOp(op, newVal);
      return mlir::success();
    }

    if (auto typevar = convertedType.template dyn_cast<plier::TypeVar>()) {
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
      if (!isOmittedType(type))
        return {};

      auto name = type.cast<plier::PyType>().getName();
      if (!name.consume_front("omitted(default=") || !name.consume_back(")"))
        return {};

      int64_t intVal;
      if (dstType.isa<mlir::IntegerType>() && !name.getAsInteger(10, intVal)) {
        return rewriter.getIntegerAttr(dstType, intVal);
      }

      double dblVal;
      if (dstType.isF32() && !name.getAsDouble(dblVal))
        return rewriter.getF32FloatAttr(static_cast<float>(dblVal));

      if (dstType.isF64() && !name.getAsDouble(dblVal))
        return rewriter.getF64FloatAttr(dblVal);

      return {};
    };

    if (auto omittedAttr =
            getOmittedValue(adaptor.value().getType(), convertedType)) {
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
    auto name = op.name();
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
  matchAndRewrite(imex::util::UndefOp op, imex::util::UndefOp::Adaptor /*adapator*/,
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

mlir::Value intCast(mlir::PatternRewriter &rewriter, mlir::Location loc,
                    mlir::Value val, mlir::Type dstType) {
  auto srcIntType = val.getType().cast<mlir::IntegerType>();
  auto dstIntType = dstType.cast<mlir::IntegerType>();
  auto srcSignless = plier::makeSignlessType(srcIntType);
  auto dstSignless = plier::makeSignlessType(dstIntType);
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

mlir::Value intFloatCast(mlir::PatternRewriter &rewriter, mlir::Location loc,
                         mlir::Value val, mlir::Type dstType) {
  auto srcIntType = val.getType().cast<mlir::IntegerType>();
  auto signlessType = plier::makeSignlessType(srcIntType);
  if (val.getType() != signlessType)
    val = rewriter.createOrFold<imex::util::SignCastOp>(loc, signlessType, val);

  if (srcIntType.isSigned()) {
    return rewriter.createOrFold<mlir::arith::SIToFPOp>(loc, dstType, val);
  } else {
    return rewriter.createOrFold<mlir::arith::UIToFPOp>(loc, dstType, val);
  }
}

mlir::Value floatIntCast(mlir::PatternRewriter &rewriter, mlir::Location loc,
                         mlir::Value val, mlir::Type dstType) {
  auto dstIntType = dstType.cast<mlir::IntegerType>();
  mlir::Value res;
  auto dstSignlessType = plier::makeSignlessType(dstIntType);
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

mlir::Value indexCastImpl(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Value val, mlir::Type dstType) {
  if (val.getType().isa<mlir::FloatType>()) {
    auto intType = rewriter.getI64Type();
    val = rewriter.createOrFold<mlir::arith::FPToSIOp>(loc, intType, val);
  }
  if (dstType.isa<mlir::FloatType>()) {
    auto intType = rewriter.getI64Type();
    val = plier::indexCast(rewriter, loc, val, intType);
    return rewriter.createOrFold<mlir::arith::SIToFPOp>(loc, dstType, val);
  }
  return plier::indexCast(rewriter, loc, val, dstType);
}

mlir::Value floatCastImpl(mlir::PatternRewriter &rewriter, mlir::Location loc,
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

mlir::Value doCast(mlir::PatternRewriter &rewriter, mlir::Location loc,
                   mlir::Value val, mlir::Type dstType) {
  assert(dstType);
  auto srcType = val.getType();
  if (auto literal = srcType.dyn_cast<plier::LiteralType>()) {
    auto attr = literal.getValue().cast<mlir::TypedAttr>();
    auto signlessAttr = makeSignlessAttr(attr).cast<mlir::TypedAttr>();
    val = rewriter.create<mlir::arith::ConstantOp>(loc, signlessAttr);
    if (signlessAttr.getType() != attr.getType())
      val = rewriter.create<imex::util::SignCastOp>(loc, attr.getType(), val);

    srcType = val.getType();
  }

  if (srcType == dstType)
    return val;

  struct Handler {
    using selector_t = bool (*)(mlir::Type);
    using cast_op_t = mlir::Value (*)(mlir::PatternRewriter &, mlir::Location,
                                      mlir::Value, mlir::Type);
    selector_t src;
    selector_t dst;
    cast_op_t cast_op;
  };

  const Handler handlers[] = {
      {&isInt, &isInt, &intCast},
      {&isInt, &isFloat, &intFloatCast},
      {&isFloat, &isInt, &floatIntCast},
      {&isIndex, &isInt, &indexCastImpl},
      {&isInt, &isIndex, &indexCastImpl},
      {&isFloat, &isFloat, &floatCastImpl},
      {&isIndex, &isFloat, &indexCastImpl},
      {&isFloat, &isIndex, &indexCastImpl},
  };

  for (auto &h : handlers)
    if (h.src(srcType) && h.dst(dstType))
      return h.cast_op(rewriter, loc, val, dstType);

  return nullptr;
}

mlir::Value invalidReplaceOp(mlir::PatternRewriter & /*rewriter*/,
                             mlir::Location /*loc*/,
                             mlir::ValueRange /*operands*/,
                             mlir::Type /*newType*/) {
  llvm_unreachable("invalidReplaceOp");
}

template <typename T>
mlir::Value replaceOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                      mlir::ValueRange operands, mlir::Type newType) {
  auto signlessType = plier::makeSignlessType(newType);
  llvm::SmallVector<mlir::Value> newOperands(operands.size());
  for (auto it : llvm::enumerate(operands))
    newOperands[it.index()] = doCast(rewriter, loc, it.value(), signlessType);

  auto res = rewriter.createOrFold<T>(loc, newOperands);
  return doCast(rewriter, loc, res, newType);
}

mlir::Value replaceIpowOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::ValueRange operands, mlir::Type newType) {
  auto f64Type = rewriter.getF64Type();
  auto a = doCast(rewriter, loc, operands[0], f64Type);
  auto b = doCast(rewriter, loc, operands[1], f64Type);
  auto fres = rewriter.create<mlir::math::PowFOp>(loc, a, b).getResult();
  return doCast(rewriter, loc, fres, newType);
}

mlir::Value replaceItruedivOp(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::ValueRange operands,
                              mlir::Type newType) {
  assert(newType.isa<mlir::FloatType>());
  auto lhs = doCast(rewriter, loc, operands[0], newType);
  auto rhs = doCast(rewriter, loc, operands[1], newType);
  return rewriter.createOrFold<mlir::arith::DivFOp>(loc, lhs, rhs);
}

mlir::Value replaceIfloordivOp(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, mlir::ValueRange operands,
                               mlir::Type newType) {
  auto newIntType = newType.cast<mlir::IntegerType>();
  auto signlessType = plier::makeSignlessType(newIntType);
  auto lhs = doCast(rewriter, loc, operands[0], signlessType);
  auto rhs = doCast(rewriter, loc, operands[1], signlessType);
  mlir::Value res;
  if (newIntType.isSigned()) {
    res = rewriter.createOrFold<mlir::arith::FloorDivSIOp>(loc, lhs, rhs);
  } else {
    res = rewriter.createOrFold<mlir::arith::DivUIOp>(loc, lhs, rhs);
  }
  return doCast(rewriter, loc, res, newType);
}

mlir::Value replaceFfloordivOp(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, mlir::ValueRange operands,
                               mlir::Type newType) {
  assert(newType.isa<mlir::FloatType>());
  auto lhs = doCast(rewriter, loc, operands[0], newType);
  auto rhs = doCast(rewriter, loc, operands[1], newType);
  auto res = rewriter.createOrFold<mlir::arith::DivFOp>(loc, lhs, rhs);
  return rewriter.createOrFold<mlir::math::FloorOp>(loc, res);
}

mlir::Value replaceImodOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::ValueRange operands, mlir::Type newType) {
  auto signlessType = plier::makeSignlessType(operands[0].getType());
  auto a = doCast(rewriter, loc, operands[0], signlessType);
  auto b = doCast(rewriter, loc, operands[1], signlessType);
  auto v1 = rewriter.create<mlir::arith::RemSIOp>(loc, a, b).getResult();
  auto v2 = rewriter.create<mlir::arith::AddIOp>(loc, v1, b).getResult();
  auto res = rewriter.create<mlir::arith::RemSIOp>(loc, v2, b).getResult();
  return doCast(rewriter, loc, res, newType);
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
  auto signlessType = plier::makeSignlessType(type);
  auto a = doCast(rewriter, loc, operands[0], signlessType);
  auto b = doCast(rewriter, loc, operands[1], signlessType);
  if (SignedPred == UnsignedPred || type.isSigned()) {
    return rewriter.createOrFold<mlir::arith::CmpIOp>(loc, SignedPred, a, b);
  } else {
    return rewriter.createOrFold<mlir::arith::CmpIOp>(loc, UnsignedPred, a, b);
  }
}

template <mlir::arith::CmpFPredicate Pred>
mlir::Value replaceCmpfOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::ValueRange operands, mlir::Type /*newType*/) {
  auto signlessType = plier::makeSignlessType(operands[0].getType());
  auto a = doCast(rewriter, loc, operands[0], signlessType);
  auto b = doCast(rewriter, loc, operands[1], signlessType);
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
    auto type0 = dropLiteralType(operands[0].getType());
    auto type1 = dropLiteralType(operands[1].getType());
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
          doCast(rewriter, loc, convertedOperands[0], finalType),
          doCast(rewriter, loc, convertedOperands[1], finalType)};
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
        if (h.type == op.op()) {
          auto res = (h.*mem)(rewriter, loc, convertedOperands, resType);
          if (res.getType() != resType)
            res = rewriter.createOrFold<imex::util::SignCastOp>(loc, resType, res);
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
    auto lhs = adaptor.lhs();
    auto rhs = adaptor.rhs();
    auto lhsType = lhs.getType().dyn_cast<mlir::TupleType>();
    if (!lhsType)
      return mlir::failure();

    auto loc = op->getLoc();
    if (adaptor.op() == "+") {
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
          auto elem =
              rewriter.create<plier::GetItemOp>(loc, elemType, arg, ind);
          newArgs.emplace_back(elem);
          newTypes.emplace_back(elemType);
        }
      }

      auto newTupleType = mlir::TupleType::get(getContext(), newTypes);
      rewriter.replaceOpWithNewOp<plier::BuildTupleOp>(op, newTupleType,
                                                       newArgs);
      return mlir::success();
    }

    return mlir::failure();
  }
};

static mlir::Value negate(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Value val, mlir::Type resType) {
  val = doCast(rewriter, loc, val, resType);
  if (auto itype = resType.dyn_cast<mlir::IntegerType>()) {
    auto signless = plier::makeSignlessType(resType);
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
  return doCast(rewriter, loc, arg, resType);
}

static mlir::Value unaryMinus(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::Value arg,
                              mlir::Type resType) {
  return negate(rewriter, loc, arg, resType);
}

static mlir::Value unaryNot(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::Value arg, mlir::Type resType) {
  auto i1 = rewriter.getIntegerType(1);
  auto casted = doCast(rewriter, loc, arg, i1);
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
    signlessType = plier::makeSignlessType(intType);
    if (intType != signlessType)
      arg = rewriter.create<imex::util::SignCastOp>(loc, signlessType, arg);
  }

  auto all = rewriter.create<mlir::arith::ConstantIntOp>(loc, -1, signlessType);

  arg = rewriter.create<mlir::arith::XOrIOp>(loc, all, arg);

  if (intType != signlessType)
    arg = rewriter.create<imex::util::SignCastOp>(loc, intType, arg);

  if (resType != arg.getType())
    arg = doCast(rewriter, loc, arg, resType);

  return arg;
}

struct UnaryOpLowering : public mlir::OpConversionPattern<plier::UnaryOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::UnaryOp op, plier::UnaryOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter = *getTypeConverter();
    auto arg = adaptor.value();
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

    auto opname = op.op();
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
    auto src = adaptor.value();
    auto dstType = converter.convertType(op.getType());
    if (!dstType)
      return mlir::failure();

    auto srcType = src.getType();
    if (srcType == dstType) {
      rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(op, dstType,
                                                                    src);
      return mlir::success();
    }

    auto res = doCast(rewriter, op.getLoc(), src, dstType);
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
  plier::addPipelineJumpMarker(mod, marker);
}

mlir::LogicalResult
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

mlir::LogicalResult
lowerRangeImpl(plier::PyCallOp op, mlir::ValueRange operands,
               llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs,
               mlir::PatternRewriter &rewriter) {
  auto parent = op->getParentOp();
  auto res = lowerRange(op, operands, kwargs, rewriter);
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

struct BuiltinCallsLowering final : public plier::CallOpLowering {
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

    auto results = std::move(res).getValue();
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

struct ExternalCallsLowering final : public plier::CallOpLowering {
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
    : public plier::RewriteWrapperPass<
          BuiltinCallsLoweringPass, void, void, BuiltinCallsLowering,
          plier::ExpandCallVarargs, ExternalCallsLowering> {};

struct PlierToStdPass
    : public mlir::PassWrapper<PlierToStdPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlierToStdPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<plier::PlierDialect>();
    registry.insert<imex::util::ImexUtilDialect>();
  }

  void runOnOperation() override;
};

void PlierToStdPass::runOnOperation() {
  mlir::TypeConverter typeConverter;
  // Convert unknown types to itself
  typeConverter.addConversion([](mlir::Type type) { return type; });

  auto context = &getContext();
  populateStdTypeConverter(*context, typeConverter);
  plier::populateTupleTypeConverter(*context, typeConverter);

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
    return res && res.isa<mlir::IntegerType, mlir::FloatType, mlir::IndexType,
                          plier::LiteralType>();
  };

  auto isTuple = [&](mlir::Type t) -> bool {
    if (!t)
      return false;

    auto res = typeConverter.convertType(t);
    return res && res.isa<mlir::TupleType>();
  };

  target.addDynamicallyLegalOp<plier::BinOp>([&](plier::BinOp op) {
    auto lhsType = op.lhs().getType();
    auto rhsType = op.rhs().getType();
    if (op.op() == "+" && isTuple(lhsType) && isTuple(rhsType))
      return false;

    return !isNum(lhsType) || !isNum(rhsType) || !isNum(op.getType());
  });
  target.addDynamicallyLegalOp<plier::UnaryOp>([&](plier::UnaryOp op) {
    return !isNum(op.value().getType()) && !isNum(op.getType());
  });
  target.addDynamicallyLegalOp<plier::CastOp>([&](plier::CastOp op) {
    auto inputType = op.value().getType();
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

        if (type.isa<mlir::NoneType, plier::TypeVar>())
          return false;

        if (auto literal = type.dyn_cast<plier::LiteralType>())
          type = literal.getValue().cast<mlir::TypedAttr>().getType();

        return !type.isIntOrFloat();
      });
  target.addDynamicallyLegalOp<imex::util::UndefOp>([&](imex::util::UndefOp op) {
    auto srcType = op.getType();
    auto dstType = typeConverter.convertType(srcType);
    return srcType == dstType;
  });

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
      UndefOpLowering
      // clang-format on
      >(typeConverter, context);

  plier::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                            patterns, target);
  plier::populateTupleTypeConversionRewritesAndTarget(typeConverter, patterns,
                                                      target);

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns))))
    signalPassFailure();
}

struct ConvertLiteralTypesPass
    : public mlir::PassWrapper<PlierToStdPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    mlir::TypeConverter typeConverter;
    // Convert unknown types to itself
    typeConverter.addConversion([](mlir::Type type) { return type; });

    auto context = &getContext();
    typeConverter.addConversion([](plier::LiteralType type) {
      return type.getValue().cast<mlir::TypedAttr>().getType();
    });

    auto materializeCast =
        [](mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs,
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

    plier::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                              patterns, target);
    plier::populateTupleTypeConversionRewritesAndTarget(typeConverter, patterns,
                                                        target);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      signalPassFailure();
  }
};

static void populatePlierToStdPipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<PlierToStdPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<BuiltinCallsLoweringPass>());
  pm.addPass(plier::createForceInlinePass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(std::make_unique<ConvertLiteralTypesPass>());
  pm.addPass(mlir::createCanonicalizerPass());
}
} // namespace

void populateStdTypeConverter(mlir::MLIRContext & /*context*/,
                              mlir::TypeConverter &converter) {
  converter.addConversion(
      [](mlir::Type type, llvm::SmallVectorImpl<mlir::Type> &retTypes)
          -> llvm::Optional<mlir::LogicalResult> {
        if (isOmittedType(type))
          return mlir::success();

        auto ret = mapPlierType(type);
        if (!ret)
          return llvm::None;

        retTypes.push_back(ret);
        return mlir::success();
      });
}

void registerPlierToStdPipeline(plier::PipelineRegistry &registry) {
  registry.registerPipeline([](auto sink) {
    auto stage = getHighLoweringStage();
    sink(plierToStdPipelineName(), {plierToScfPipelineName()}, {stage.end},
         {plierToScfPipelineName()}, &populatePlierToStdPipeline);
  });
}

llvm::StringRef plierToStdPipelineName() { return "plier_to_std"; }
