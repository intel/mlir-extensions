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

#pragma once

namespace pybind11
{
class bytes;
class capsule;
class object;
class str;
}

pybind11::capsule create_module();

pybind11::capsule lower_function(const pybind11::object& compilation_context,
                                 const pybind11::capsule& py_mod,
                                 const pybind11::object& func_ir);

pybind11::bytes compile_module(const pybind11::object& compilation_context,
                               const pybind11::capsule& py_mod);

pybind11::str module_str(const pybind11::capsule& py_mod);
