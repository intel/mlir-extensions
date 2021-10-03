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

#include "pipelines/plier_to_std.hpp"
#include "pipelines/plier_to_scf.hpp"
#include "pipelines/pre_low_simplifications.hpp"

#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/TypeSwitch.h>

#include "plier/dialect.hpp"

#include "plier/pass/rewrite_wrapper.hpp"
#include "plier/rewrites/arg_lowering.hpp"
#include "plier/rewrites/call_lowering.hpp"
#include "plier/rewrites/cast_lowering.hpp"
#include "plier/rewrites/force_inline.hpp"
#include "plier/rewrites/type_conversion.hpp"
#include "plier/transforms/cast_utils.hpp"
#include "plier/transforms/const_utils.hpp"
#include "plier/transforms/func_utils.hpp"
#include "plier/transforms/pipeline_utils.hpp"

#include "base_pipeline.hpp"
#include "loop_utils.hpp"
#include "mangle.hpp"
#include "plier/compiler/pipeline_registry.hpp"
#include "py_func_resolver.hpp"
#include "py_linalg_resolver.hpp"

namespace {
mlir::Type map_int_type(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  unsigned num_bits = 0;
  if (name.consume_front("int") && !name.consumeInteger<unsigned>(10, num_bits))
    return mlir::IntegerType::get(&ctx, num_bits, mlir::IntegerType::Signed);

  if (name.consume_front("uint") &&
      !name.consumeInteger<unsigned>(10, num_bits))
    return mlir::IntegerType::get(&ctx, num_bits, mlir::IntegerType::Unsigned);

  return nullptr;
}

mlir::Type map_int_literal_type(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  int64_t value = 0;
  if (name.consume_front("Literal[int](") &&
      !name.consumeInteger<int64_t>(10, value) && name.consume_front(")")) {
    auto type = mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Signed);
    auto attr = mlir::IntegerAttr::get(type, value);
    return plier::LiteralType::get(attr);
  }
  return nullptr;
}

mlir::Type map_bool_literal_type(mlir::MLIRContext &ctx,
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

mlir::Type map_bool_type(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  if (name.consume_front("bool")) {
    return mlir::IntegerType::get(&ctx, 1);
  }
  return nullptr;
}

mlir::Type map_float_type(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  unsigned num_bits = 0;
  if (name.consume_front("float") &&
      !name.consumeInteger<unsigned>(10, num_bits)) {
    switch (num_bits) {
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

bool consume_until(llvm::StringRef &name, llvm::StringRef end) {
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
        if (!consume_until(name, it.second))
          return false;
      }
    }

    if (!consumed)
      name = name.drop_front();
  }
  return false;
}

mlir::Type map_plier_type_name(mlir::MLIRContext &ctx, llvm::StringRef &name);
bool map_type_helper(mlir::MLIRContext &ctx, llvm::StringRef &name,
                     mlir::Type &ret, llvm::StringRef end) {
  auto nameCopy = name;
  auto type = map_plier_type_name(ctx, nameCopy);
  if (type && nameCopy.consume_front(end)) {
    ret = type;
    name = nameCopy;
    return true;
  }
  nameCopy = name;
  if (consume_until(nameCopy, end)) {
    auto len = name.size() - nameCopy.size() - end.size();
    ret = plier::PyType::get(&ctx, name.take_front(len));
    name = nameCopy;
    return true;
  }
  return false;
}

mlir::Type map_pair_type(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  mlir::Type first;
  mlir::Type second;
  if (name.consume_front("pair<") && map_type_helper(ctx, name, first, ", ") &&
      map_type_helper(ctx, name, second, ">")) {
    return mlir::TupleType::get(&ctx, {first, second});
  }
  return nullptr;
}

mlir::Type map_unituple_type(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  mlir::Type type;
  unsigned count = 0;
  if (name.consume_front("UniTuple(") &&
      map_type_helper(ctx, name, type, " x ") &&
      !name.consumeInteger<unsigned>(10, count) && name.consume_front(")")) {
    llvm::SmallVector<mlir::Type> types(count, type);
    return mlir::TupleType::get(&ctx, types);
  }
  return nullptr;
}

mlir::Type map_tuple_type(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  if (!name.consume_front("Tuple("))
    return nullptr;

  if (name.consume_front(")"))
    return mlir::TupleType::get(&ctx, llvm::None);

  llvm::SmallVector<mlir::Type> types;
  auto temp = name;
  if (!consume_until(temp, ")"))
    return nullptr;
  auto len = name.size() - temp.size();
  temp = name.take_front(len);
  while (true) {
    mlir::Type type;
    if (map_type_helper(ctx, temp, type, ", ")) {
      types.push_back(type);
      continue;
    }
    if (map_type_helper(ctx, temp, type, ")")) {
      types.push_back(type);
      break;
    }
  }
  name = name.drop_front(len);
  return mlir::TupleType::get(&ctx, types);
}

mlir::Type map_func_type(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  if (name.consume_front("Function(") &&
      name.consume_front("<class 'bool'>") && // TODO unhardcode;
      name.consume_front(")")) {
    return mlir::FunctionType::get(&ctx, {}, {});
  }
  return nullptr;
}

mlir::Type map_dtype_type(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  if (name.consume_front("dtype(") && name.consume_back(")")) {
    auto innerType = map_plier_type_name(ctx, name);
    if (innerType) {
      return plier::TypeVar::get(innerType);
    }
  }
  return nullptr;
}

mlir::Type map_none_type(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  if (name.consume_front("none"))
    return mlir::NoneType::get(&ctx);

  return nullptr;
}

mlir::Type map_plier_type_name(mlir::MLIRContext &ctx, llvm::StringRef &name) {
  using func_t =
      mlir::Type (*)(mlir::MLIRContext & ctx, llvm::StringRef & name);
  const func_t handlers[] = {
      // clang-format off
      &map_int_type,
      &map_int_literal_type,
      &map_bool_literal_type,
      &map_bool_type,
      &map_float_type,
      &map_pair_type,
      &map_unituple_type,
      &map_tuple_type,
      &map_func_type,
      &map_dtype_type,
      &map_none_type,
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

mlir::Type map_plier_type(mlir::Type type) {
  assert(type);
  if (!type.isa<plier::PyType>())
    return type;

  auto name = type.cast<plier::PyType>().getName();
  return map_plier_type_name(*type.getContext(), name);
}

static mlir::Type dropLiteralType(mlir::Type t) {
  assert(t);
  if (auto literal = t.dyn_cast<plier::LiteralType>())
    return dropLiteralType(literal.getValue().getType());

  return t;
}

static bool is_supported_type(mlir::Type type) {
  assert(type);
  return type.isIntOrFloat();
}

bool is_int(mlir::Type type) {
  assert(type);
  return type.isa<mlir::IntegerType>();
}

bool is_float(mlir::Type type) {
  assert(type);
  return type.isa<mlir::FloatType>();
}

bool is_index(mlir::Type type) {
  assert(type);
  return type.isa<mlir::IndexType>();
}

struct ConstOpLowering : public mlir::OpConversionPattern<plier::ConstOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::ConstOp op, plier::ConstOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto value = adaptor.val();
    if (is_supported_type(value.getType())) {
      if (auto intAttr = value.dyn_cast<mlir::IntegerAttr>()) {
        auto type = intAttr.getType().cast<mlir::IntegerType>();
        if (!type.isSignless()) {
          auto intVal = intAttr.getValue().getSExtValue();
          auto constVal = rewriter.create<mlir::ConstantIntOp>(
              op.getLoc(), intVal, type.getWidth());
          rewriter.replaceOpWithNewOp<plier::SignCastOp>(op, type, constVal);
          return mlir::success();
        }
      }
      rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, value);
      return mlir::success();
    }

    auto &converter = *getTypeConverter();
    if (auto type = converter.convertType(op.getType())) {
      if (type.isa<mlir::NoneType>()) {
        rewriter.replaceOpWithNewOp<plier::UndefOp>(op, type);
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

bool isOmittedType(mlir::Type type) {
  if (auto pytype = type.dyn_cast<plier::PyType>()) {
    auto name = pytype.getName();
    if (name.consume_front("omitted(") && name.consume_back(")")) {
      return true;
    }
  }
  return false;
}

mlir::Attribute makeSignlessAttr(mlir::Attribute val) {
  auto type = val.getType();
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
  matchAndRewrite(Op op, typename Op::Adaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = op.getType();
    auto &converter = *(this->getTypeConverter());
    auto convertedType = converter.convertType(type);
    if (!convertedType)
      return mlir::failure();

    if (convertedType.template isa<mlir::NoneType>()) {
      rewriter.replaceOpWithNewOp<plier::UndefOp>(op, convertedType);
      return mlir::success();
    }

    auto literal = convertedType.template dyn_cast<plier::LiteralType>();
    if (!literal)
      return mlir::failure();

    auto loc = op.getLoc();
    auto attrVal = literal.getValue();
    auto dstType = attrVal.getType();
    auto val = makeSignlessAttr(attrVal);
    auto newVal = rewriter.create<mlir::ConstantOp>(loc, val).getResult();
    if (dstType != val.getType())
      newVal = rewriter.create<plier::SignCastOp>(loc, dstType, newVal);

    rewriter.replaceOp(op, newVal);
    return mlir::success();
  }
};

struct UndefOpLowering : public mlir::OpConversionPattern<plier::UndefOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::UndefOp op, plier::UndefOp::Adaptor /*adapator*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto oldType = op.getType();
    auto &converter = *getTypeConverter();
    auto type = converter.convertType(oldType);
    if (!type || oldType == type)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<plier::UndefOp>(op, type);
    return mlir::success();
  }
};

struct ReturnOpLowering : public mlir::OpRewritePattern<mlir::ReturnOp> {
  ReturnOpLowering(mlir::TypeConverter & /*typeConverter*/,
                   mlir::MLIRContext *context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::ReturnOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto operands = op.getOperands();
    auto func = mlir::cast<mlir::FuncOp>(op->getParentOp());
    auto res_types = func.getType().getResults();
    assert(res_types.size() == operands.size() || res_types.empty());
    bool converted = (res_types.size() != operands.size());
    llvm::SmallVector<mlir::Value, 4> new_vals;
    for (auto it : llvm::zip(operands, res_types)) {
      auto src = std::get<0>(it);
      auto dst = std::get<1>(it);
      if (src.getType() != dst) {
        auto new_op = rewriter.create<plier::CastOp>(op.getLoc(), dst, src);
        new_vals.push_back(new_op);
        converted = true;
      } else {
        new_vals.push_back(src);
      }
    }
    if (converted) {
      rewriter.create<mlir::ReturnOp>(op.getLoc(), new_vals);
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct SelectOpLowering : public mlir::OpRewritePattern<mlir::SelectOp> {
  SelectOpLowering(mlir::TypeConverter &typeConverter,
                   mlir::MLIRContext *context)
      : OpRewritePattern(context), converter(typeConverter) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::SelectOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto oldType = op.getType();
    auto newType = converter.convertType(oldType);
    if (!newType || oldType == newType)
      return mlir::failure();

    auto loc = op.getLoc();
    auto getSrcArg = [&](mlir::Value arg) -> mlir::Value {
      if (arg.getType() != newType)
        return rewriter.create<plier::CastOp>(loc, newType, arg);

      return arg;
    };

    auto trueArg = getSrcArg(op.getTrueValue());
    auto falseArg = getSrcArg(op.getFalseValue());
    rewriter.replaceOpWithNewOp<mlir::SelectOp>(op, op.condition(), trueArg,
                                                falseArg);
    return mlir::success();
  }

private:
  mlir::TypeConverter &converter;
};

static mlir::Type coerce(mlir::Type type0, mlir::Type type1) {
  // TODO: proper rules
  assert(type0 != type1);
  auto get_bits_count = [](mlir::Type type) -> unsigned {
    assert(type);
    if (type.isa<mlir::IntegerType>()) {
      return type.cast<mlir::IntegerType>().getWidth();
    }
    if (type.isa<mlir::Float16Type>()) {
      return 11;
    }
    if (type.isa<mlir::Float32Type>()) {
      return 24;
    }
    if (type.isa<mlir::Float64Type>()) {
      return 53;
    }
    llvm_unreachable("Unhandled type");
  };
  auto f0 = is_float(type0);
  auto f1 = is_float(type1);
  if (f0 && !f1)
    return type0;

  if (!f0 && f1)
    return type1;

  return get_bits_count(type0) < get_bits_count(type1) ? type1 : type0;
}

mlir::Value int_cast(mlir::PatternRewriter &rewriter, mlir::Location loc,
                     mlir::Value val, mlir::Type dstType) {
  auto srcIntType = val.getType().cast<mlir::IntegerType>();
  auto dstIntType = dstType.cast<mlir::IntegerType>();
  auto srcSignless = plier::makeSignlessType(srcIntType);
  auto dstSignless = plier::makeSignlessType(dstIntType);
  auto srcBits = srcIntType.getWidth();
  auto dstBits = dstIntType.getWidth();

  if (srcIntType != srcSignless)
    val = rewriter.createOrFold<plier::SignCastOp>(loc, srcSignless, val);

  if (dstBits > srcBits) {
    if (srcIntType.isSigned()) {
      val = rewriter.createOrFold<mlir::SignExtendIOp>(loc, val, dstSignless);
    } else {
      val = rewriter.createOrFold<mlir::ZeroExtendIOp>(loc, val, dstSignless);
    }
  } else if (dstBits < srcBits) {
    if (dstBits == 1) {
      // Special handling for bool
      auto zero = rewriter.create<mlir::ConstantIntOp>(loc, 0, srcBits);
      auto cmp = rewriter.createOrFold<mlir::CmpIOp>(
          loc, mlir::CmpIPredicate::eq, val, zero);
      auto trueVal = rewriter.create<mlir::ConstantIntOp>(loc, 1, 1);
      auto falseVal = rewriter.create<mlir::ConstantIntOp>(loc, 0, 1);
      val = rewriter.createOrFold<mlir::SelectOp>(loc, cmp, falseVal, trueVal);
    } else {
      val = rewriter.createOrFold<mlir::TruncateIOp>(loc, val, dstSignless);
    }
  }

  if (dstIntType != dstSignless)
    val = rewriter.createOrFold<plier::SignCastOp>(loc, dstIntType, val);

  return val;
}

mlir::Value int_float_cast(mlir::PatternRewriter &rewriter, mlir::Location loc,
                           mlir::Value val, mlir::Type dstType) {
  auto srcIntType = val.getType().cast<mlir::IntegerType>();
  auto signlessType = plier::makeSignlessType(srcIntType);
  if (val.getType() != signlessType) {
    val = rewriter.createOrFold<plier::SignCastOp>(loc, signlessType, val);
  }

  if (srcIntType.isSigned()) {
    return rewriter.createOrFold<mlir::SIToFPOp>(loc, val, dstType);
  } else {
    return rewriter.createOrFold<mlir::UIToFPOp>(loc, val, dstType);
  }
}

mlir::Value float_int_cast(mlir::PatternRewriter &rewriter, mlir::Location loc,
                           mlir::Value val, mlir::Type dstType) {
  auto dstIntType = dstType.cast<mlir::IntegerType>();
  mlir::Value res;
  auto dstSignlessType = plier::makeSignlessType(dstIntType);
  if (dstIntType.getWidth() == 1) {
    // Special handling for bool
    auto zero = rewriter.create<mlir::ConstantFloatOp>(
        loc, llvm::APFloat(0.0), val.getType().cast<mlir::FloatType>());
    auto cmp = rewriter.createOrFold<mlir::CmpFOp>(
        loc, mlir::CmpFPredicate::OEQ, val, zero);
    auto trueVal = rewriter.create<mlir::ConstantIntOp>(loc, 1, 1);
    auto falseVal = rewriter.create<mlir::ConstantIntOp>(loc, 0, 1);
    res = rewriter.createOrFold<mlir::SelectOp>(loc, cmp, falseVal, trueVal);
  } else if (dstIntType.isSigned()) {
    res = rewriter.create<mlir::FPToSIOp>(loc, val, dstSignlessType);
  } else {
    res = rewriter.create<mlir::FPToUIOp>(loc, val, dstSignlessType);
  }
  if (dstSignlessType != dstIntType) {
    return rewriter.createOrFold<plier::SignCastOp>(loc, dstIntType, res);
  }
  return res;
}

mlir::Value index_cast_impl(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::Value val, mlir::Type dstType) {
  return plier::index_cast(rewriter, loc, val, dstType);
}

mlir::Value float_cast_impl(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::Value val, mlir::Type dstType) {
  auto srcFloatType = val.getType().cast<mlir::FloatType>();
  auto dstFloatType = dstType.cast<mlir::FloatType>();
  assert(srcFloatType != dstFloatType);
  if (dstFloatType.getWidth() > srcFloatType.getWidth()) {
    return rewriter.createOrFold<mlir::FPExtOp>(loc, val, dstFloatType);
  } else {
    return rewriter.createOrFold<mlir::FPTruncOp>(loc, val, dstFloatType);
  }
}

mlir::Value doCast(mlir::PatternRewriter &rewriter, mlir::Location loc,
                   mlir::Value val, mlir::Type dstType) {
  assert(dstType);
  auto srcType = val.getType();
  if (auto literal = srcType.dyn_cast<plier::LiteralType>()) {
    auto attr = literal.getValue();
    auto signlessAttr = makeSignlessAttr(attr);
    val = rewriter.create<mlir::ConstantOp>(loc, signlessAttr);
    if (signlessAttr.getType() != attr.getType())
      val = rewriter.create<plier::SignCastOp>(loc, attr.getType(), val);

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
      {&is_int, &is_int, &int_cast},
      {&is_int, &is_float, &int_float_cast},
      {&is_float, &is_int, &float_int_cast},
      {&is_index, &is_int, &index_cast_impl},
      {&is_int, &is_index, &index_cast_impl},
      {&is_float, &is_float, &float_cast_impl},
  };

  for (auto &h : handlers) {
    if (h.src(srcType) && h.dst(dstType))
      return h.cast_op(rewriter, loc, val, dstType);
  }

  return nullptr;
}

template <typename T>
mlir::Value replace_op(mlir::PatternRewriter &rewriter, mlir::Location loc,
                       mlir::ValueRange operands, mlir::Type newType) {
  auto signlessType = plier::makeSignlessType(newType);
  llvm::SmallVector<mlir::Value> newOperands(operands.size());
  for (auto it : llvm::enumerate(operands))
    newOperands[it.index()] = doCast(rewriter, loc, it.value(), signlessType);

  auto res = rewriter.createOrFold<T>(loc, newOperands);
  return doCast(rewriter, loc, res, newType);
}

mlir::Value replace_ipow_op(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::ValueRange operands, mlir::Type newType) {
  auto f64Type = rewriter.getF64Type();
  auto a = doCast(rewriter, loc, operands[0], f64Type);
  auto b = doCast(rewriter, loc, operands[1], f64Type);
  auto fres = rewriter.create<mlir::math::PowFOp>(loc, a, b).getResult();
  return doCast(rewriter, loc, fres, newType);
}

mlir::Value replace_itruediv_op(mlir::PatternRewriter &rewriter,
                                mlir::Location loc, mlir::ValueRange operands,
                                mlir::Type newType) {
  assert(newType.isa<mlir::FloatType>());
  auto lhs = doCast(rewriter, loc, operands[0], newType);
  auto rhs = doCast(rewriter, loc, operands[1], newType);
  return rewriter.createOrFold<mlir::DivFOp>(loc, lhs, rhs);
}

mlir::Value replace_ifloordiv_op(mlir::PatternRewriter &rewriter,
                                 mlir::Location loc, mlir::ValueRange operands,
                                 mlir::Type newType) {
  auto newIntType = newType.cast<mlir::IntegerType>();
  auto signlessType = plier::makeSignlessType(newIntType);
  auto lhs = doCast(rewriter, loc, operands[0], signlessType);
  auto rhs = doCast(rewriter, loc, operands[1], signlessType);
  mlir::Value res;
  if (newIntType.isSigned()) {
    res = rewriter.createOrFold<mlir::SignedFloorDivIOp>(loc, lhs, rhs);
  } else {
    res = rewriter.createOrFold<mlir::UnsignedDivIOp>(loc, lhs, rhs);
  }
  return doCast(rewriter, loc, res, newType);
}

mlir::Value replace_ffloordiv_op(mlir::PatternRewriter &rewriter,
                                 mlir::Location loc, mlir::ValueRange operands,
                                 mlir::Type newType) {
  assert(newType.isa<mlir::FloatType>());
  auto lhs = doCast(rewriter, loc, operands[0], newType);
  auto rhs = doCast(rewriter, loc, operands[1], newType);
  auto res = rewriter.createOrFold<mlir::DivFOp>(loc, lhs, rhs);
  return rewriter.createOrFold<mlir::FloorFOp>(loc, res);
}

mlir::Value replace_imod_op(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::ValueRange operands, mlir::Type newType) {
  auto signlessType = plier::makeSignlessType(operands[0].getType());
  auto a = doCast(rewriter, loc, operands[0], signlessType);
  auto b = doCast(rewriter, loc, operands[1], signlessType);
  auto v1 = rewriter.create<mlir::SignedRemIOp>(loc, a, b).getResult();
  auto v2 = rewriter.create<mlir::AddIOp>(loc, v1, b).getResult();
  auto res = rewriter.create<mlir::SignedRemIOp>(loc, v2, b).getResult();
  return doCast(rewriter, loc, res, newType);
}

mlir::Value replace_fmod_op(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::ValueRange operands, mlir::Type /*newType*/) {
  auto a = operands[0];
  auto b = operands[1];
  auto v1 = rewriter.create<mlir::RemFOp>(loc, a, b).getResult();
  auto v2 = rewriter.create<mlir::AddFOp>(loc, v1, b).getResult();
  return rewriter.create<mlir::RemFOp>(loc, v2, b).getResult();
}

template <mlir::CmpIPredicate SignedPred,
          mlir::CmpIPredicate UnsignedPred = SignedPred>
mlir::Value replace_cmpi_op(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::ValueRange operands, mlir::Type /*newType*/) {
  assert(operands.size() == 2);
  assert(operands[0].getType() == operands[1].getType());
  auto type = operands[0].getType().cast<mlir::IntegerType>();
  auto signlessType = plier::makeSignlessType(type);
  auto a = doCast(rewriter, loc, operands[0], signlessType);
  auto b = doCast(rewriter, loc, operands[1], signlessType);
  if (SignedPred == UnsignedPred || type.isSigned()) {
    return rewriter.createOrFold<mlir::CmpIOp>(loc, SignedPred, a, b);
  } else {
    return rewriter.createOrFold<mlir::CmpIOp>(loc, UnsignedPred, a, b);
  }
}

template <mlir::CmpFPredicate Pred>
mlir::Value replace_cmpf_op(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::ValueRange operands, mlir::Type /*newType*/) {
  auto signlessType = plier::makeSignlessType(operands[0].getType());
  auto a = doCast(rewriter, loc, operands[0], signlessType);
  auto b = doCast(rewriter, loc, operands[1], signlessType);
  return rewriter.createOrFold<mlir::CmpFOp>(loc, Pred, a, b);
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
    if (!is_supported_type(type0) || !is_supported_type(type1))
      return mlir::failure();

    auto resType = converter.convertType(op.getType());
    if (!resType || !is_supported_type(resType))
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
        {"+", &replace_op<mlir::AddIOp>, &replace_op<mlir::AddFOp>},
        {"-", &replace_op<mlir::SubIOp>, &replace_op<mlir::SubFOp>},
        {"*", &replace_op<mlir::MulIOp>, &replace_op<mlir::MulFOp>},
        {"**", &replace_ipow_op, &replace_op<mlir::math::PowFOp>},
        {"/", &replace_itruediv_op, &replace_op<mlir::DivFOp>},
        {"//", &replace_ifloordiv_op, &replace_ffloordiv_op},
        {"%", &replace_imod_op, &replace_fmod_op},

        {">",
         &replace_cmpi_op<mlir::CmpIPredicate::sgt, mlir::CmpIPredicate::ugt>,
         &replace_cmpf_op<mlir::CmpFPredicate::OGT>},
        {">=",
         &replace_cmpi_op<mlir::CmpIPredicate::sge, mlir::CmpIPredicate::uge>,
         &replace_cmpf_op<mlir::CmpFPredicate::OGE>},
        {"<",
         &replace_cmpi_op<mlir::CmpIPredicate::slt, mlir::CmpIPredicate::ult>,
         &replace_cmpf_op<mlir::CmpFPredicate::OLT>},
        {"<=",
         &replace_cmpi_op<mlir::CmpIPredicate::sle, mlir::CmpIPredicate::ule>,
         &replace_cmpf_op<mlir::CmpFPredicate::OLE>},
        {"!=", &replace_cmpi_op<mlir::CmpIPredicate::ne>,
         &replace_cmpf_op<mlir::CmpFPredicate::ONE>},
        {"==", &replace_cmpi_op<mlir::CmpIPredicate::eq>,
         &replace_cmpf_op<mlir::CmpFPredicate::OEQ>},
    };

    using membptr_t = func_t OpDesc::*;
    auto call_handler = [&](membptr_t mem) {
      for (auto &h : handlers) {
        if (h.type == op.op()) {
          auto res = (h.*mem)(rewriter, loc, convertedOperands, resType);
          if (res.getType() != resType)
            res = rewriter.createOrFold<plier::SignCastOp>(loc, resType, res);
          rewriter.replaceOp(op, res);
          return mlir::success();
        }
      }
      return mlir::failure();
    };

    if (is_int(finalType)) {
      return call_handler(&OpDesc::iop);
    } else if (is_float(finalType)) {
      return call_handler(&OpDesc::fop);
    }
    return mlir::failure();
  }
};

mlir::Value negate(mlir::PatternRewriter &rewriter, mlir::Location loc,
                   mlir::Value val, mlir::Type resType) {
  val = doCast(rewriter, loc, val, resType);
  if (auto itype = resType.dyn_cast<mlir::IntegerType>()) {
    auto signless = plier::makeSignlessType(resType);
    if (signless != itype) {
      val = rewriter.create<plier::SignCastOp>(loc, signless, val);
    }
    // TODO: no int negation?
    auto zero = rewriter.create<mlir::ConstantOp>(
        loc, mlir::IntegerAttr::get(signless, 0));
    auto res = rewriter.create<mlir::SubIOp>(loc, zero, val).getResult();
    if (signless != itype) {
      res = rewriter.create<plier::SignCastOp>(loc, itype, res);
    }
    return res;
  }
  if (resType.isa<mlir::FloatType>()) {
    return rewriter.create<mlir::NegFOp>(loc, val);
  }
  llvm_unreachable("negate: unsupported type");
}

struct UnaryOpLowering : public mlir::OpConversionPattern<plier::UnaryOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::UnaryOp op, plier::UnaryOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter = *getTypeConverter();
    auto arg = adaptor.value();
    auto type = arg.getType();
    if (!is_supported_type(type))
      return mlir::failure();

    auto resType = converter.convertType(op.getType());
    if (!resType)
      return mlir::failure();

    auto loc = op.getLoc();
    if (op.op() == "+") {
      arg = doCast(rewriter, loc, arg, resType);
      rewriter.replaceOp(op, arg);
      return mlir::success();
    }
    assert(op.op() == "-");
    auto new_val = negate(rewriter, loc, arg, resType);
    rewriter.replaceOp(op, new_val);
    return mlir::success();
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
    if (srcType == dstType)
      return mlir::failure();

    auto res = doCast(rewriter, op.getLoc(), src, dstType);
    if (!res)
      return mlir::failure();

    rewriter.replaceOp(op, res);

    return mlir::success();
  }
};

struct FixupWhileYieldTypes
    : public mlir::OpRewritePattern<mlir::scf::YieldOp> {
  FixupWhileYieldTypes(mlir::MLIRContext *context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op->getParentOp());
    if (whileOp == nullptr) {
      return mlir::failure();
    }

    auto loc = rewriter.getUnknownLoc();
    auto results = op.results();
    bool changed = false;
    assert(whileOp->getNumOperands() == op->getNumOperands());
    llvm::SmallVector<mlir::Value> newResults(whileOp->getNumOperands());

    rewriter.setInsertionPoint(op);
    for (auto it :
         llvm::enumerate(llvm::zip(whileOp->getOperandTypes(), results))) {
      auto index = it.index();
      auto value = it.value();
      auto operand_type = std::get<0>(value);
      auto result = std::get<1>(value);
      auto result_type = result.getType();

      if (result_type != operand_type) {
        auto cast = rewriter.create<plier::CastOp>(loc, operand_type, result);
        changed = true;
        newResults[index] = cast.getResult();
      } else {
        newResults[index] = result;
      }
    }

    if (changed) {
      rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, newResults);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct FixupWhileTypes : public mlir::OpRewritePattern<mlir::scf::WhileOp> {
  FixupWhileTypes(mlir::TypeConverter & /*typeConverter*/,
                  mlir::MLIRContext *context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::WhileOp op,
                  mlir::PatternRewriter &rewriter) const override {
    bool changed = false;
    mlir::OpBuilder::InsertionGuard g(rewriter);
    auto before_block = &op.before().front();
    rewriter.startRootUpdate(op);
    rewriter.setInsertionPointToStart(before_block);
    assert(before_block->getNumArguments() == op.getNumOperands());
    auto loc = rewriter.getUnknownLoc();
    for (auto it :
         llvm::zip(op.getOperandTypes(), before_block->getArguments())) {
      auto new_type = std::get<0>(it);
      auto arg = std::get<1>(it);
      auto old_type = arg.getType();
      if (old_type != new_type) {
        auto cast = rewriter.create<plier::CastOp>(loc, old_type, arg);
        arg.setType(new_type);
        arg.replaceAllUsesExcept(cast, cast);
        changed = true;
      }
    }

    auto term =
        mlir::cast<mlir::scf::ConditionOp>(before_block->getTerminator());
    auto after_types = term.args().getTypes();

    auto after_block = &op.after().front();
    rewriter.setInsertionPointToStart(after_block);
    assert(after_block->getNumArguments() == term.args().size());
    for (auto it : llvm::zip(after_types, after_block->getArguments())) {
      auto new_type = std::get<0>(it);
      auto arg = std::get<1>(it);
      auto old_type = arg.getType();
      if (old_type != new_type) {
        auto cast = rewriter.create<plier::CastOp>(loc, old_type, arg);
        arg.setType(new_type);
        arg.replaceAllUsesExcept(cast, cast);
        changed = true;
      }
    }

    rewriter.setInsertionPointAfter(op);
    assert(op.getNumResults() == term.args().size());
    for (auto it : llvm::zip(after_types, op.getResults())) {
      auto new_type = std::get<0>(it);
      auto arg = std::get<1>(it);
      auto old_type = arg.getType();
      if (old_type != new_type) {
        auto cast = rewriter.create<plier::CastOp>(loc, old_type, arg);
        arg.setType(new_type);
        arg.replaceAllUsesExcept(cast, cast);
        changed = true;
      }
    }

    if (changed) {
      rewriter.finalizeRootUpdate(op);
    } else {
      rewriter.cancelRootUpdate(op);
    }
    return mlir::success(changed);
  }
};

struct PropagateBuildTupleTypes
    : public mlir::OpRewritePattern<plier::BuildTupleOp> {
  PropagateBuildTupleTypes(mlir::TypeConverter & /*typeConverter*/,
                           mlir::MLIRContext *context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(plier::BuildTupleOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto tupleType = op.getType().dyn_cast<mlir::TupleType>();
    if (tupleType && op.getOperandTypes() == tupleType.getTypes()) {
      return mlir::failure();
    }

    auto newType = mlir::TupleType::get(op.getContext(), op.getOperandTypes());
    rewriter.replaceOpWithNewOp<plier::BuildTupleOp>(op, newType,
                                                     op.getOperands());
    return mlir::success();
  }
};

struct FoldTupleGetitem : public mlir::OpRewritePattern<plier::GetItemOp> {
  FoldTupleGetitem(mlir::TypeConverter & /*typeConverter*/,
                   mlir::MLIRContext *context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(plier::GetItemOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto buildTuple = op.value().getDefiningOp<plier::BuildTupleOp>();
    if (!buildTuple) {
      return mlir::failure();
    }

    if (auto val = plier::getConstVal<mlir::IntegerAttr>(op.index())) {
      auto index = plier::getIntAttrValue(val);
      if (index >= 0 && index < buildTuple.getNumOperands()) {
        auto val = buildTuple.getOperand(static_cast<unsigned>(index));
        rewriter.replaceOp(op, val);
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

struct FoldSliceGetitem : public mlir::OpRewritePattern<plier::GetItemOp> {
  FoldSliceGetitem(mlir::TypeConverter & /*typeConverter*/,
                   mlir::MLIRContext *context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(plier::GetItemOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto buildSice = op.value().getDefiningOp<plier::BuildSliceOp>();
    if (!buildSice) {
      return mlir::failure();
    }

    auto loc = op.getLoc();
    auto indexType = rewriter.getIndexType();
    if (auto val = plier::getConstVal<mlir::IntegerAttr>(op.index())) {
      auto index = plier::getIntAttrValue(val);
      if (index >= 0 && index < 3 &&
          !buildSice.getOperand(static_cast<unsigned>(index))
               .getType()
               .isa<mlir::NoneType>()) {
        auto val = buildSice.getOperand(static_cast<unsigned>(index));
        rewriter.replaceOp(op, doCast(rewriter, loc, val, indexType));
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

struct FixupFuncOpaqueTypes : public mlir::OpRewritePattern<mlir::FuncOp> {
  FixupFuncOpaqueTypes(mlir::TypeConverter & /*typeConverter*/,
                       mlir::MLIRContext *context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::FuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // TODO: This is a hack. Remove after moving to dialect conversion
    auto funcType = op.getType();
    auto numArgs = funcType.getNumInputs();
    bool hasBody = !op.getBody().empty();

    auto opaqueType = plier::OpaqueType::get(op.getContext());
    bool changed = false;
    llvm::SmallVector<mlir::Type> newArgs(numArgs);
    for (auto i : llvm::seq(0u, numArgs)) {
      auto origType = funcType.getInput(i);
      auto type = origType.dyn_cast<plier::PyType>();
      auto canChange = (hasBody ? op.getArgument(i).getUsers().empty() : true);
      if (canChange && type &&
          type.getName().startswith("type(CPUDispatcher")) {
        newArgs[i] = opaqueType;
        if (hasBody)
          op.getArgument(i).setType(opaqueType);
        changed = true;
      } else {
        newArgs[i] = origType;
      }
    }

    if (changed) {
      auto newFuncType = mlir::FunctionType::get(op.getContext(), newArgs,
                                                 funcType.getResults());
      rewriter.updateRootInPlace(op, [&]() { op.setType(newFuncType); });
    }
    return mlir::success(changed);
  }
};

struct ExpandCallVarargs : public mlir::OpRewritePattern<plier::PyCallOp> {
  ExpandCallVarargs(mlir::TypeConverter & /*typeConverter*/,
                    mlir::MLIRContext *context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(plier::PyCallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto vararg = op.varargs();
    if (!vararg)
      return mlir::failure();

    auto varargType = vararg.getType().dyn_cast<mlir::TupleType>();
    if (!varargType)
      return mlir::failure();

    auto argsCount = op.args().size();
    auto varargsCount = varargType.size();
    llvm::SmallVector<mlir::Value> args(argsCount + varargsCount);
    llvm::copy(op.args(), args.begin());

    auto loc = op.getLoc();
    for (auto i : llvm::seq<size_t>(0, varargsCount)) {
      auto type = varargType.getType(i);
      auto index =
          rewriter.create<mlir::ConstantIndexOp>(loc, static_cast<int64_t>(i));
      args[argsCount + i] =
          rewriter.create<plier::GetItemOp>(loc, type, vararg, index);
    }

    auto resType = op.getType();
    rewriter.replaceOpWithNewOp<plier::PyCallOp>(op, resType, op.func(), args,
                                                 mlir::Value(), op.kwargs(),
                                                 op.func_name(), op.kw_names());
    return mlir::success();
  }
};

mlir::LogicalResult
lowerLen(plier::PyCallOp op, llvm::ArrayRef<mlir::Value> operands,
         llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs,
         mlir::PatternRewriter &rewriter, mlir::Type /*dstType*/) {
  if (!kwargs.empty()) {
    return mlir::failure();
  }
  if (operands.size() != 1) {
    return mlir::failure();
  }

  auto build_tuple = operands[0].getDefiningOp<plier::BuildTupleOp>();
  if (!build_tuple) {
    return mlir::failure();
  }

  auto size = rewriter.create<mlir::ConstantIndexOp>(
      op.getLoc(), build_tuple.getNumOperands());
  auto cast = rewriter.create<plier::CastOp>(op.getLoc(), op.getType(), size);
  rewriter.replaceOp(op, cast.getResult());
  return mlir::success();
}

mlir::LogicalResult
lowerSlice(plier::PyCallOp op, llvm::ArrayRef<mlir::Value> operands,
           llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs,
           mlir::PatternRewriter &rewriter, mlir::Type /*dstType*/) {
  if (!kwargs.empty()) {
    return mlir::failure();
  }
  if (operands.size() != 2 && operands.size() != 3) {
    return mlir::failure();
  }

  if (llvm::any_of(operands, [](mlir::Value op) {
        return !op.getType()
                    .isa<mlir::IntegerType, mlir::IndexType, mlir::NoneType>();
      })) {
    return mlir::failure();
  }

  auto begin = operands[0];
  auto end = operands[1];
  auto stride = [&]() -> mlir::Value {
    if (operands.size() == 3) {
      return operands[2];
    }
    return rewriter.create<mlir::ConstantIndexOp>(op.getLoc(), 1);
  }();

  rewriter.replaceOpWithNewOp<plier::BuildSliceOp>(op, begin, end, stride);
  return mlir::success();
}

mlir::LogicalResult
lowerCastFunc(plier::PyCallOp op, llvm::ArrayRef<mlir::Value> operands,
              llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs,
              mlir::PatternRewriter &rewriter, mlir::Type dstType) {
  if (!kwargs.empty()) {
    return mlir::failure();
  }
  if (operands.size() != 1) {
    return mlir::failure();
  }
  auto val = operands[0];
  bool success = false;
  auto replace_op = [&](mlir::Value val) {
    assert(!success);
    if (val) {
      rewriter.replaceOp(op, val);
      success = true;
    }
  };
  auto loc = op.getLoc();
  replace_op(doCast(rewriter, loc, val, dstType));
  return mlir::success(success);
}

mlir::FuncOp get_lib_symbol(mlir::ModuleOp mod, llvm::StringRef name,
                            mlir::FunctionType type,
                            mlir::PatternRewriter &rewriter) {
  assert(!name.empty());
  if (auto op = mod.lookupSymbol<mlir::FuncOp>(name)) {
    assert(op.getType() == type);
    return op;
  }

  return plier::add_function(rewriter, mod, name, type);
}

mlir::LogicalResult
lower_math_func(plier::PyCallOp op, llvm::StringRef name,
                llvm::ArrayRef<mlir::Value> args,
                llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs,
                mlir::PatternRewriter &rewriter) {
  if (!kwargs.empty()) {
    return mlir::failure();
  }
  auto ret_type = map_plier_type(op.getType());
  auto valid_type = [&](mlir::Type type) {
    return type.isa<mlir::Float32Type, mlir::Float64Type, mlir::IntegerType>();
  };
  if (ret_type && name.consume_front("math.") && args.size() == 1 &&
      valid_type(args[0].getType())) {
    auto loc = op.getLoc();
    mlir::Value arg = rewriter.create<plier::CastOp>(loc, ret_type, args[0]);
    auto is_float = ret_type.isa<mlir::Float32Type>();
    auto func_type =
        mlir::FunctionType::get(op.getContext(), ret_type, ret_type);
    auto module = op->getParentOfType<mlir::ModuleOp>();
    mlir::FuncOp func;
    if (is_float) {
      func = get_lib_symbol(module, name.str() + "f", func_type, rewriter);
    } else // double
    {
      func = get_lib_symbol(module, name, func_type, rewriter);
    }
    auto call = rewriter.create<mlir::CallOp>(loc, func, arg);
    rewriter.replaceOp(op, call.getResults());
    return mlir::success();
  }

  return mlir::failure();
}

mlir::LogicalResult
lowerRangeImpl(plier::PyCallOp op, llvm::ArrayRef<mlir::Value> operands,
               llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs,
               mlir::PatternRewriter &rewriter, mlir::Type /*dstType*/) {
  return lowerRange(op, operands, kwargs, rewriter);
}

struct CallLowerer {
  CallLowerer(mlir::TypeConverter &typeConverter) : converter(typeConverter) {}

  mlir::LogicalResult
  operator()(plier::PyCallOp op, llvm::StringRef name,
             llvm::ArrayRef<mlir::Value> args,
             llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs,
             mlir::PatternRewriter &rewriter) {
    if (mlir::succeeded(lower_math_func(op, name, args, kwargs, rewriter))) {
      return mlir::success();
    }

    auto resType = converter.convertType(op->getResult(0).getType());
    if (!resType) {
      return mlir::failure();
    }

    using func_t = mlir::LogicalResult (*)(
        plier::PyCallOp, llvm::ArrayRef<mlir::Value>,
        llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>>,
        mlir::PatternRewriter &, mlir::Type);
    const std::pair<llvm::StringRef, func_t> handlers[] = {
        {"bool", lowerCastFunc},  {"int", lowerCastFunc},
        {"float", lowerCastFunc}, {"range", lowerRangeImpl},
        {"len", lowerLen},        {"slice", lowerSlice},
    };
    for (auto &handler : handlers) {
      if (handler.first == name) {
        return handler.second(op, args, kwargs, rewriter, resType);
      }
    }

    mlir::ValueRange r(args);
    auto mangled_name = mangle(name, r.getTypes());
    if (!mangled_name.empty()) {
      auto mod = op->getParentOfType<mlir::ModuleOp>();
      assert(mod);
      auto func = mod.lookupSymbol<mlir::FuncOp>(mangled_name);
      if (!func) {
        func = pyResolver.getFunc(name, r.getTypes());
        if (func) {
          func.setPrivate();
          func.setName(mangled_name);
        }
      }
      if (func) {
        assert(func.getType().getNumResults() == op->getNumResults());
        auto new_func_call =
            rewriter.create<mlir::CallOp>(op.getLoc(), func, args);
        rewriter.replaceOp(op, new_func_call.getResults());
        return mlir::success();
      }
    }

    return mlir::failure();
  }

private:
  mlir::TypeConverter &converter;
  PyFuncResolver pyResolver;
};

static mlir::Value skipCast(mlir::Value val) {
  if (auto cast = val.getDefiningOp<plier::CastOp>())
    return cast.value();

  return val;
};

static void rerun_scf_pipeline(mlir::Operation *op) {
  assert(nullptr != op);
  auto marker =
      mlir::StringAttr::get(op->getContext(), plierToScfPipelineName());
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  assert(nullptr != mod);
  plier::add_pipeline_jump_marker(mod, marker);
}

struct BuiltinCallsLowering : public mlir::OpRewritePattern<plier::PyCallOp> {
  BuiltinCallsLowering(mlir::MLIRContext *context)
      : OpRewritePattern(context),
        resolver("numba_dpcomp.mlir.builtin.funcs", "registry") {}

  mlir::LogicalResult
  matchAndRewrite(plier::PyCallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto funcName = op.func_name();

    llvm::SmallVector<mlir::Value> args;
    args.reserve(op.args().size() + 1);
    auto func = op.func();
    auto getattr =
        mlir::dyn_cast_or_null<plier::GetattrOp>(func.getDefiningOp());
    if (getattr)
      args.emplace_back(skipCast(getattr.getOperand()));

    for (auto arg : op.args())
      args.emplace_back(skipCast(arg));

    llvm::SmallVector<std::pair<llvm::StringRef, mlir::Value>> kwargs;
    for (auto it : llvm::zip(op.kwargs(), op.kw_names())) {
      auto arg = skipCast(std::get<0>(it));
      auto name = std::get<1>(it).cast<mlir::StringAttr>();
      kwargs.emplace_back(name.getValue(), arg);
    }

    //    for (auto &handler : builtinFuncsHandlers)
    //      if (handler.first == funcName)
    //        return handler.second(op, args, kwargs, rewriter);

    auto res =
        resolver.rewriteFunc(funcName, op.getLoc(), rewriter, args, kwargs);
    if (!res)
      return mlir::failure();

    rerun_scf_pipeline(op);
    rewriter.replaceOp(op, *res);
    return mlir::success();
  }

private:
  PyLinalgResolver resolver;
};

struct BuiltinCallsLoweringPass
    : public plier::RewriteWrapperPass<BuiltinCallsLoweringPass, void, void,
                                       BuiltinCallsLowering> {};

struct ForceInlinePass
    : public plier::RewriteWrapperPass<ForceInlinePass, void, void,
                                       plier::ForceInline> {};

struct PlierToStdPass
    : public mlir::PassWrapper<PlierToStdPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<plier::PlierDialect>();
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::math::MathDialect>();
  }

  void runOnOperation() override;
};

void PlierToStdPass::runOnOperation() {
  mlir::TypeConverter typeConverter;
  // Convert unknown types to itself
  typeConverter.addConversion([](mlir::Type type) { return type; });

  auto context = &getContext();
  populateStdTypeConverter(*context, typeConverter);

  auto materializeCast = [](mlir::OpBuilder &builder, mlir::Type type,
                            mlir::ValueRange inputs,
                            mlir::Location loc) -> llvm::Optional<mlir::Value> {
    // TODO: UnrealizedConversionCast
    if (inputs.size() == 1)
      return builder.create<plier::CastOp>(loc, type, inputs.front())
          .getResult();

    return llvm::None;
  };
  typeConverter.addArgumentMaterialization(materializeCast);
  typeConverter.addSourceMaterialization(materializeCast);
  typeConverter.addTargetMaterialization(materializeCast);

  mlir::RewritePatternSet patterns(context);
  mlir::ConversionTarget target(*context);

  auto isNum = [&](mlir::Type t) -> bool {
    auto res = typeConverter.convertType(t);
    return res && res.isIntOrFloat();
  };

  target.addDynamicallyLegalOp<plier::BinOp>([&](plier::BinOp op) {
    return !isNum(op.rhs().getType()) && !isNum(op.lhs().getType()) &&
           !isNum(op.getType());
  });
  target.addDynamicallyLegalOp<plier::UnaryOp>([&](plier::UnaryOp op) {
    return !isNum(op.value().getType()) && !isNum(op.getType());
  });
  target.addDynamicallyLegalOp<plier::CastOp>([&](plier::CastOp op) {
    auto srcType = typeConverter.convertType(op.value().getType());
    auto dstType = typeConverter.convertType(op.getType());
    return srcType == dstType;
  });
  target.addDynamicallyLegalOp<plier::ConstOp, plier::GlobalOp>(
      [&](mlir::Operation *op) {
        auto type = typeConverter.convertType(op->getResult(0).getType());
        if (!type)
          return true;

        if (auto literal = type.dyn_cast<plier::LiteralType>())
          type = literal.getValue().getType();

        return !type.isIntOrFloat();
      });
  target.addDynamicallyLegalOp<plier::UndefOp>([&](plier::UndefOp op) {
    auto srcType = op.getType();
    auto dstType = typeConverter.convertType(srcType);
    return srcType == dstType;
  });

  patterns.insert<
      // clang-format off
      BinOpLowering,
      UnaryOpLowering,
      LowerCasts,
      ConstOpLowering,
      LiteralLowering<plier::CastOp>,
      LiteralLowering<plier::GlobalOp>,
      UndefOpLowering
      // clang-format on
      >(typeConverter, context);

  plier::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                            patterns, target);

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns))))
    signalPassFailure();

  //  patterns.insert<
  //      // clang-format off
  //      plier::FuncOpSignatureConversion,
  //      plier::FixupIfTypes,
  //      plier::FixCallOmittedArgs,
  //      plier::FixupIfYieldTypes,
  //      LiteralLowering<plier::CastOp>,
  //      LiteralLowering<plier::GlobalOp>,
  //      UndefOpLowering,
  //      ReturnOpLowering,
  //      ConstOpLowering,
  //      SelectOpLowering,
  //      BinOpLowering,
  //      UnaryOpLowering,
  //      FixupWhileTypes,
  //      PropagateBuildTupleTypes,
  //      FoldTupleGetitem,
  //      FoldSliceGetitem,
  //      FixupFuncOpaqueTypes,
  //      ExpandCallVarargs
  //      // clang-format on
  //      >(typeConverter, context);

  //  patterns.insert<FixupWhileYieldTypes, plier::ArgOpLowering>(context);

  //  patterns.insert<plier::CastOpLowering>(typeConverter, context, &doCast);

  //  CallLowerer callLowerer(typeConverter);

  //  patterns.insert<plier::CallOpLowering>(typeConverter, context,
  //                                         std::ref(callLowerer));

  //  mlir::populateStdExpandOpsPatterns(patterns);

  //  for (auto *op : context->getRegisteredOperations())
  //    op->getCanonicalizationPatterns(patterns, context);

  //  (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
  //  std::move(patterns));
}

void populate_plier_to_std_pipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<PlierToStdPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<BuiltinCallsLoweringPass>());
  pm.addPass(std::make_unique<ForceInlinePass>());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createStdExpandOpsPass());
  pm.addPass(mlir::createCanonicalizerPass());
}
} // namespace

void populateStdTypeConverter(mlir::MLIRContext & /*context*/,
                              mlir::TypeConverter &converter) {
  converter.addConversion(
      [](mlir::Type type, llvm::SmallVectorImpl<mlir::Type> &ret_types)
          -> llvm::Optional<mlir::LogicalResult> {
        if (isOmittedType(type)) {
          return mlir::success();
        }
        auto ret = map_plier_type(type);
        if (!ret) {
          return llvm::None;
        }
        ret_types.push_back(ret);
        return mlir::success();
      });
}

void populateTupleTypeConverter(mlir::MLIRContext & /*context*/,
                                mlir::TypeConverter &converter) {
  converter.addConversion(
      [&converter](mlir::TupleType type) -> llvm::Optional<mlir::Type> {
        llvm::SmallVector<mlir::Type> newTypes(type.size());
        for (auto it : llvm::enumerate(type)) {
          auto converted = converter.convertType(it.value());
          if (!converted)
            return mlir::Type{};
          newTypes[it.index()] = converted;
        }
        return mlir::TupleType::get(type.getContext(), newTypes);
      });
}

void registerPlierToStdPipeline(plier::PipelineRegistry &registry) {
  registry.register_pipeline([](auto sink) {
    auto stage = getHighLoweringStage();
    sink(plierToStdPipelineName(), {plierToScfPipelineName()}, {stage.end},
         {plierToScfPipelineName()}, &populate_plier_to_std_pipeline);
  });
}

llvm::StringRef plierToStdPipelineName() { return "plier_to_std"; }
