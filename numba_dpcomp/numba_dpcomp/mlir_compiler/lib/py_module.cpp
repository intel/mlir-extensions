// Copyright 2021 Intel Corporation
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

#include <pybind11/pybind11.h>

#include "py_module.hpp"

#include "lowering.hpp"

namespace {
bool is_dpnp_supported() {
#ifdef IMEX_USE_DPNP
  return true;
#else
  return false;
#endif
}
} // namespace

PYBIND11_MODULE(mlir_compiler, m) {
  m.def("init_compiler", &initCompiler, "No docs");
  m.def("create_module", &createModule, "No docs");
  m.def("lower_function", &lowerFunction, "No docs");
  m.def("compile_module", &compileModule, "No docs");
  m.def("module_str", &moduleStr, "No docs");
  m.def("is_dpnp_supported", &is_dpnp_supported, "No docs");
}
