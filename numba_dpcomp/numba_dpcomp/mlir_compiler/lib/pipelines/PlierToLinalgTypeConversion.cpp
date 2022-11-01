// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PlierToLinalgTypeConversion.hpp"

#include "PyTypeConverter.hpp"

#include "imex/Dialect/imex_util/Dialect.hpp"
#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"

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

      return imex::util::TypeVarType::get(type);
    }

    if (py::isinstance(obj, array)) {
      auto elemType = converter.convertType(context, obj.attr("dtype"));
      if (!elemType)
        return llvm::None;

      auto layout = obj.attr("layout").cast<std::string>();

      auto ndim = obj.attr("ndim").cast<size_t>();
      llvm::SmallVector<int64_t> shape(ndim, mlir::ShapedType::kDynamicSize);

      return imex::ntensor::NTensorType::get(shape, elemType, /*env*/ {},
                                             llvm::StringRef(layout));
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
