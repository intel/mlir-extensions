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

#include "LowerToGpuTypeConversion.hpp"

#include "PyTypeConverter.hpp"

#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"

#include <pybind11/pybind11.h>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>

namespace py = pybind11;

namespace {
struct Conversion {
  Conversion(PyTypeConverter &conv) : converter(conv) {
    py::object mod = py::module::import("numba_dpcomp.mlir.dpctl_interop");
    usmArrayType = mod.attr("USMNdArrayType");
  }

  llvm::Optional<mlir::Type> operator()(mlir::MLIRContext &context,
                                        py::handle obj) {
    if (usmArrayType.is_none() || !py::isinstance(obj, usmArrayType))
      return llvm::None;

    auto elemType = converter.convertType(context, obj.attr("dtype"));
    if (!elemType)
      return llvm::None;

    auto layout = obj.attr("layout").cast<std::string>();

    auto ndim = obj.attr("ndim").cast<size_t>();
    llvm::SmallVector<int64_t> shape(ndim, mlir::ShapedType::kDynamicSize);

    // TODO: environment
    return imex::ntensor::NTensorType::get(shape, elemType, /*env*/ {},
                                           llvm::StringRef(layout));
  }

private:
  PyTypeConverter &converter;

  py::object usmArrayType;
};
} // namespace

void populateGpuTypeConverter(PyTypeConverter &converter) {
  converter.addConversion(Conversion(converter));
}
