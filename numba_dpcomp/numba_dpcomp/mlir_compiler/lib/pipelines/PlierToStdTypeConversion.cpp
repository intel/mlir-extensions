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

#include <pybind11/pybind11.h>

#include <mlir/IR/BuiltinTypes.h>
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
        // clang-format on
};

namespace {
struct Conversion {
  Conversion() {
    py::object mod = py::module::import("numba.core.types");
    for (auto [i, it] : llvm::enumerate(PrimitiveTypes)) {
      auto [name, func] = it;
      auto obj = mod.attr(name.data());
      primitiveTypes[i] = {obj, func};
    }
  }

  llvm::Optional<mlir::Type> operator()(mlir::MLIRContext &context,
                                        py::handle obj) {
    for (auto &[cls, func] : primitiveTypes) {
      if (obj.is(cls))
        return func(context);
    }
    return llvm::None;
  }

private:
  using TypePair = std::pair<py::object, TypeFunc>;
  std::array<TypePair, std::size(PrimitiveTypes)> primitiveTypes;
};
} // namespace

void populateStdTypeConverter(PyTypeConverter &converter) {
  converter.addConversion(Conversion());
}
