// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "LowerToGpuTypeConversion.hpp"

#include "PyTypeConverter.hpp"

#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOps.hpp"
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
      return std::nullopt;

    auto elemType = converter.convertType(context, obj.attr("dtype"));
    if (!elemType)
      return std::nullopt;

    auto layout = obj.attr("layout").cast<std::string>();

    auto ndim = obj.attr("ndim").cast<size_t>();

    auto fixedDims = obj.attr("fixed_dims").cast<py::tuple>();
    if (fixedDims.size() != ndim)
      return std::nullopt;

    llvm::SmallVector<int64_t> shape(ndim);
    for (auto [i, dim] : llvm::enumerate(fixedDims)) {
      if (dim.is_none()) {
        shape[i] = mlir::ShapedType::kDynamic;
      } else {
        shape[i] = dim.cast<int64_t>();
      }
    }

    auto devAttr = mlir::StringAttr::get(
        &context, obj.attr("filter_string").cast<std::string>());
    auto env = gpu_runtime::GPURegionDescAttr::get(&context, devAttr);

    return imex::ntensor::NTensorType::get(shape, elemType, env,
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
