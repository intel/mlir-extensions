// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PlierToStdTypeConversion.hpp"

#include "PyTypeConverter.hpp"

#include "imex/Dialect/imex_util/Dialect.hpp"
#include "imex/Dialect/plier/Dialect.hpp"

#include <pybind11/pybind11.h>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>

namespace py = pybind11;

static mlir::Type getBoolType(mlir::MLIRContext &ctx) {
  return mlir::IntegerType::get(&ctx, 1, mlir::IntegerType::Signless);
}

template <unsigned Width, bool Signed>
static mlir::Type getIntType(mlir::MLIRContext &ctx) {
  auto sign =
      (Signed ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned);
  return mlir::IntegerType::get(&ctx, Width, sign);
}

static mlir::Type getFloat16Type(mlir::MLIRContext &ctx) {
  return mlir::FloatType::getF16(&ctx);
}

static mlir::Type getFloat32Type(mlir::MLIRContext &ctx) {
  return mlir::FloatType::getF32(&ctx);
}

static mlir::Type getFloat64Type(mlir::MLIRContext &ctx) {
  return mlir::FloatType::getF64(&ctx);
}

static mlir::Type getComplex64Type(mlir::MLIRContext &ctx) {
  return mlir::ComplexType::get(getFloat32Type(ctx));
}

static mlir::Type getComplex128Type(mlir::MLIRContext &ctx) {
  return mlir::ComplexType::get(getFloat64Type(ctx));
}

static mlir::Type getNoneType(mlir::MLIRContext &ctx) {
  return mlir::NoneType::get(&ctx);
}

static mlir::Type getSliceType(mlir::MLIRContext &ctx) {
  return plier::SliceType::get(&ctx);
}

using TypeFunc = mlir::Type (*)(mlir::MLIRContext &);
static const constexpr std::pair<llvm::StringLiteral, TypeFunc>
    PrimitiveTypes[] = {
        // clang-format off
        {"boolean", &getBoolType},

        {"int8",  &getIntType<8, true>},
        {"uint8", &getIntType<8, false>},
        {"int16",  &getIntType<16, true>},
        {"uint16", &getIntType<16, false>},
        {"int32",  &getIntType<32, true>},
        {"uint32", &getIntType<32, false>},
        {"int64",  &getIntType<64, true>},
        {"uint64", &getIntType<64, false>},

        {"float16", &getFloat16Type},
        {"float32", &getFloat32Type},
        {"float64", &getFloat64Type},

        {"complex64", &getComplex64Type},
        {"complex128", &getComplex128Type},

        {"none", &getNoneType},

        {"slice2_type", &getSliceType},
        {"slice3_type", &getSliceType},
        // clang-format on
};

namespace {
struct Conversion {
  Conversion(PyTypeConverter &conv) : converter(conv) {
    py::object mod = py::module::import("numba.core.types");
    for (auto [i, it] : llvm::enumerate(PrimitiveTypes)) {
      auto [name, func] = it;
      auto obj = mod.attr(name.data());
      primitiveTypes[i] = {obj, func};
    }

    tupleType = mod.attr("Tuple");
    uniTupleType = mod.attr("UniTuple");
    pairType = mod.attr("Pair");

    literalType = mod.attr("Literal");
    dispatcherType = mod.attr("Dispatcher");
    functionType = mod.attr("Function");
    numberClassType = mod.attr("NumberClass");
    omittedType = mod.attr("Omitted");
  }

  llvm::Optional<mlir::Type> operator()(mlir::MLIRContext &context,
                                        py::handle obj) {
    for (auto &[cls, func] : primitiveTypes) {
      if (obj.is(cls))
        return func(context);
    }

    if (py::isinstance(obj, tupleType)) {
      llvm::SmallVector<mlir::Type> types;
      for (auto elem : obj.attr("types").cast<py::tuple>()) {
        auto type = converter.convertType(context, elem);
        if (!type)
          return std::nullopt;

        types.emplace_back(type);
      }
      return mlir::TupleType::get(&context, types);
    }

    if (py::isinstance(obj, uniTupleType)) {
      auto type = converter.convertType(context, obj.attr("dtype"));
      if (!type)
        return std::nullopt;

      auto count = obj.attr("count").cast<size_t>();
      llvm::SmallVector<mlir::Type> types(count, type);
      return mlir::TupleType::get(&context, types);
    }

    if (py::isinstance(obj, pairType)) {
      auto first = converter.convertType(context, obj.attr("first_type"));
      if (!first)
        return std::nullopt;

      auto second = converter.convertType(context, obj.attr("second_type"));
      if (!second)
        return std::nullopt;

      mlir::Type types[] = {first, second};
      return mlir::TupleType::get(&context, types);
    }

    if (py::isinstance(obj, literalType)) {
      auto value = obj.attr("literal_value");
      if (py::isinstance<py::float_>(value))
        return getFloat64Type(context);

      if (py::isinstance<py::int_>(value))
        return getIntType<64, true>(context);

      if (py::isinstance<py::bool_>(value))
        return getBoolType(context);

      return std::nullopt;
    }

    if (py::isinstance(obj, dispatcherType))
      return imex::util::OpaqueType::get(&context);

    if (py::isinstance(obj, functionType))
      return mlir::FunctionType::get(&context, {}, {});

    if (py::isinstance(obj, numberClassType)) {
      auto type = converter.convertType(context, obj.attr("instance_type"));
      if (!type)
        return std::nullopt;

      return imex::util::TypeVarType::get(type);
    }

    if (py::isinstance(obj, omittedType)) {
      auto value = obj.attr("value");

      auto [type, attr] = [&]() -> std::pair<mlir::Type, mlir::Attribute> {
        if (py::isinstance<py::float_>(value)) {
          auto type = mlir::Float64Type::get(&context);
          auto val = mlir::FloatAttr::get(type, value.cast<double>());
          return {type, val};
        }

        if (py::isinstance<py::int_>(value)) {
          auto type = mlir::IntegerType::get(&context, 64);
          auto val = mlir::IntegerAttr::get(type, value.cast<int64_t>());
          return {type, val};
        }

        if (py::isinstance<py::bool_>(value)) {
          auto type = mlir::IntegerType::get(&context, 1);
          auto val = mlir::IntegerAttr::get(
              type, static_cast<int64_t>(value.cast<bool>()));
          return {type, val};
        }

        if (value.is_none()) {
          auto type = mlir::NoneType::get(&context);
          return {type, nullptr};
        }

        return {nullptr, nullptr};
      }();

      if (!type)
        return std::nullopt;

      return plier::OmittedType::get(&context, type, attr);
    }

    return std::nullopt;
  }

private:
  PyTypeConverter &converter;

  using TypePair = std::pair<py::object, TypeFunc>;
  std::array<TypePair, std::size(PrimitiveTypes)> primitiveTypes;

  py::object tupleType;
  py::object uniTupleType;
  py::object pairType;

  py::object literalType;
  py::object dispatcherType;
  py::object functionType;
  py::object numberClassType;
  py::object omittedType;
};
} // namespace

void populateStdTypeConverter(PyTypeConverter &converter) {
  converter.addConversion(Conversion(converter));
}
