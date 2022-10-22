// Copyright 2022 Intel Corporation
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
          return llvm::None;

        types.emplace_back(type);
      }
      return mlir::TupleType::get(&context, types);
    }

    if (py::isinstance(obj, uniTupleType)) {
      auto type = converter.convertType(context, obj.attr("dtype"));
      if (!type)
        return llvm::None;

      auto count = obj.attr("count").cast<size_t>();
      llvm::SmallVector<mlir::Type> types(count, type);
      return mlir::TupleType::get(&context, types);
    }

    if (py::isinstance(obj, pairType)) {
      auto first = converter.convertType(context, obj.attr("first_type"));
      if (!first)
        return llvm::None;

      auto second = converter.convertType(context, obj.attr("second_type"));
      if (!second)
        return llvm::None;

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

      return llvm::None;
    }

    if (py::isinstance(obj, dispatcherType))
      return imex::util::OpaqueType::get(&context);

    if (py::isinstance(obj, functionType))
      return mlir::FunctionType::get(&context, {}, {});

    return llvm::None;
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
};
} // namespace

void populateStdTypeConverter(PyTypeConverter &converter) {
  converter.addConversion(Conversion(converter));
}
