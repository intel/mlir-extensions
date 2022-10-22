// Copyright 202 Intel Corporation
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

#include "CheckGpuCaps.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

using ResolveFptr = bool (*)(OffloadDeviceCapabilities *);

static ResolveFptr getResolver() {
  static ResolveFptr resolver = []() {
    py::object mod = py::module::import("numba_dpcomp.mlir.gpu_runtime");
    py::object attr = mod.attr("get_device_caps_addr");
    return reinterpret_cast<ResolveFptr>(attr.cast<uintptr_t>());
  }();
  return resolver;
}

llvm::Optional<OffloadDeviceCapabilities> getOffloadDeviceCapabilities() {
  auto resolver = getResolver();
  if (!resolver)
    return llvm::None;

  OffloadDeviceCapabilities ret;
  if (!resolver(&ret))
    return llvm::None;

  if (ret.spirvMajorVersion == 0 && ret.spirvMinorVersion == 0)
    return llvm::None;

  return ret;
}

llvm::Optional<std::string> getDefaultDevice() {
  py::object mod = py::module::import("numba_dpcomp.mlir.dpctl_interop");
  py::object res = mod.attr("get_default_device_name")();
  if (res.is_none())
    return llvm::None;

  return res.cast<std::string>();
}
