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

namespace pybind11 {
class bytes;
class capsule;
class dict;
class int_;
class object;
class str;
} // namespace pybind11

pybind11::capsule initCompiler(pybind11::dict settings);

pybind11::capsule createModule(pybind11::dict settings);

pybind11::capsule lowerFunction(const pybind11::object &compilationContext,
                                const pybind11::capsule &pyMod,
                                const pybind11::object &funcIr);

pybind11::capsule compileModule(const pybind11::capsule &compiler,
                                const pybind11::object &compilationContext,
                                const pybind11::capsule &pyMod);

void registerSymbol(const pybind11::capsule &compiler,
                    const pybind11::str &name, const pybind11::int_ &ptr);

pybind11::int_ getFunctionPointer(const pybind11::capsule &compiler,
                                  const pybind11::capsule &module,
                                  pybind11::str funcName);

void releaseModule(const pybind11::capsule &compiler,
                   const pybind11::capsule &module);

pybind11::str moduleStr(const pybind11::capsule &pyMod);
