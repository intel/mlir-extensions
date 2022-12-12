// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CheckGpuCaps.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

using ResolveFptr = bool (*)(OffloadDeviceCapabilities *, const char *);

static ResolveFptr getResolver() {
  static ResolveFptr resolver = []() {
    py::object mod = py::module::import("numba_dpcomp.mlir.gpu_runtime");
    py::object attr = mod.attr("get_device_caps_addr");
    return reinterpret_cast<ResolveFptr>(attr.cast<uintptr_t>());
  }();
  return resolver;
}

llvm::Optional<OffloadDeviceCapabilities>
getOffloadDeviceCapabilities(const std::string &name) {
  auto resolver = getResolver();
  if (!resolver)
    return std::nullopt;

  OffloadDeviceCapabilities ret;
  if (!resolver(&ret, name.c_str()))
    return std::nullopt;

  if (ret.spirvMajorVersion == 0 && ret.spirvMinorVersion == 0)
    return std::nullopt;

  return ret;
}

llvm::Optional<std::string> getDefaultDevice() {
  py::object mod = py::module::import("numba_dpcomp.mlir.dpctl_interop");
  py::object res = mod.attr("get_default_device_name")();
  if (res.is_none())
    return std::nullopt;

  return res.cast<std::string>();
}
