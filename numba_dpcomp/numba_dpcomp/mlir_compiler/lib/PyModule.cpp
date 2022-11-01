// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <pybind11/pybind11.h>

#include "PyModule.hpp"

#include "Lowering.hpp"

static bool is_dpnp_supported() {
#ifdef IMEX_USE_DPNP
  return true;
#else
  return false;
#endif
}

PYBIND11_MODULE(mlir_compiler, m) {
  m.def("init_compiler", &initCompiler, "No docs");
  m.def("create_module", &createModule, "No docs");
  m.def("lower_function", &lowerFunction, "No docs");
  m.def("compile_module", &compileModule, "No docs");
  m.def("register_symbol", &registerSymbol, "No docs");
  m.def("get_function_pointer", &getFunctionPointer, "No docs");
  m.def("release_module", &releaseModule, "No docs");
  m.def("module_str", &moduleStr, "No docs");
  m.def("is_dpnp_supported", &is_dpnp_supported, "No docs");
}
