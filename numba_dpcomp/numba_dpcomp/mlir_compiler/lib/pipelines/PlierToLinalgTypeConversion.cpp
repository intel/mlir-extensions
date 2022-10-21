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

#include "PlierToLinalgTypeConversion.hpp"

#include "PyTypeConverter.hpp"

#include "imex/Dialect/imex_util/Dialect.hpp"

#include <pybind11/pybind11.h>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>

namespace py = pybind11;

namespace {
struct Conversion {
  Conversion(PyTypeConverter &conv) : converter(conv) {
    py::object mod = py::module::import("numba.core.types");
    dtype = mod.attr("DType");
    array = mod.attr("Array");
  }

  llvm::Optional<mlir::Type> operator()(mlir::MLIRContext &context,
                                        py::handle obj) {
    if (py::isinstance(obj, dtype)) {
      auto type = converter.convertType(context, obj.attr("dtype"));
      if (!type)
        return llvm::None;

      return imex::util::TypeVar::get(type);
    }

    return llvm::None;
  }

private:
  PyTypeConverter &converter;

  py::object dtype;
  py::object array;
};
} // namespace

void populateArrayTypeConverter(PyTypeConverter &converter) {
  converter.addConversion(Conversion(converter));
}
