// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PyMapTypes.hpp"

#include <pybind11/pybind11.h>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeRange.h>

namespace py = pybind11;

namespace {
template <unsigned Width, mlir::IntegerType::SignednessSemantics Signed>
static bool isInt(mlir::Type type) {
  if (auto t = type.dyn_cast<mlir::IntegerType>())
    if (t.getWidth() == Width && t.getSignedness() == Signed)
      return true;

  return false;
}

template <unsigned Width> bool isFloat(mlir::Type type) {
  if (auto f = type.dyn_cast<mlir::FloatType>())
    if (f.getWidth() == Width)
      return true;

  return false;
}

template <unsigned Width> static bool isFloatComplex(mlir::Type type) {
  if (auto c = type.dyn_cast<mlir::ComplexType>()) {
    auto f = c.getElementType().dyn_cast<mlir::FloatType>();
    if (!f)
      return false;

    if (f.getWidth() == Width)
      return true;
  }

  return false;
}

static bool isNone(mlir::Type type) { return type.isa<mlir::NoneType>(); }

static py::object mapType(const py::handle &typesMod, mlir::Type type) {
  using fptr_t = bool (*)(mlir::Type);
  const std::pair<fptr_t, llvm::StringRef> primitiveTypes[] = {
      {&isInt<1, mlir::IntegerType::Signed>, "boolean"},
      {&isInt<1, mlir::IntegerType::Signless>, "boolean"},
      {&isInt<1, mlir::IntegerType::Unsigned>, "boolean"},

      {&isInt<8, mlir::IntegerType::Signed>, "int8"},
      {&isInt<8, mlir::IntegerType::Signless>, "int8"},
      {&isInt<8, mlir::IntegerType::Unsigned>, "uint8"},

      {&isInt<16, mlir::IntegerType::Signed>, "int16"},
      {&isInt<16, mlir::IntegerType::Signless>, "int16"},
      {&isInt<16, mlir::IntegerType::Unsigned>, "uint16"},

      {&isInt<32, mlir::IntegerType::Signed>, "int32"},
      {&isInt<32, mlir::IntegerType::Signless>, "int32"},
      {&isInt<32, mlir::IntegerType::Unsigned>, "uint32"},

      {&isInt<64, mlir::IntegerType::Signed>, "int64"},
      {&isInt<64, mlir::IntegerType::Signless>, "int64"},
      {&isInt<64, mlir::IntegerType::Unsigned>, "uint64"},

      {&isFloat<32>, "float32"},
      {&isFloat<64>, "float64"},

      {&isFloatComplex<32>, "complex64"},
      {&isFloatComplex<64>, "complex128"},

      {&isNone, "none"},
  };

  for (auto h : primitiveTypes) {
    if (h.first(type)) {
      auto name = h.second;
      return typesMod.attr(py::str(name.data(), name.size()));
    }
  }

  if (auto m = type.dyn_cast<mlir::ShapedType>()) {
    auto elemType = mapType(typesMod, m.getElementType());
    if (!elemType)
      return {};

    auto ndims = py::int_(m.getRank());
    auto arrayType = typesMod.attr("Array");
    return arrayType(elemType, ndims, py::str("C"));
  }

  if (auto t = type.dyn_cast<mlir::TupleType>()) {
    py::tuple ret(t.size());
    for (auto it : llvm::enumerate(t.getTypes())) {
      auto inner = mapType(typesMod, it.value());
      if (!inner)
        return {};
      ret[it.index()] = std::move(inner);
    }
    return std::move(ret);
  }
  return {};
}
} // namespace

py::object mapTypeToNumba(py::handle types_mod, mlir::Type type) {
  auto elem = mapType(types_mod, type);
  if (!elem)
    return py::none();

  return elem;
}

py::object mapTypesToNumba(py::handle types_mod, mlir::TypeRange types) {
  py::list ret(types.size());
  for (auto it : llvm::enumerate(types)) {
    auto type = mapTypeToNumba(types_mod, it.value());
    if (type.is_none())
      return py::none();

    ret[it.index()] = std::move(type);
  }
  return std::move(ret);
}
