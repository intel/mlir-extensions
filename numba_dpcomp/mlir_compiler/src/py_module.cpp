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

PYBIND11_MODULE(mlir_compiler, m)
{
    m.def("init_compiler", &init_compiler, "No docs");
    m.def("create_module", &create_module, "No docs");
    m.def("lower_function", &lower_function, "No docs");
    m.def("compile_module", &compile_module, "No docs");
    m.def("module_str", &module_str, "No docs");
}
