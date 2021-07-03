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
class list;
class object;
class handle;
} // namespace pybind11

namespace mlir {
class Type;
class TypeRange;
} // namespace mlir

pybind11::object map_type_to_numba(pybind11::handle types_mod, mlir::Type type);
pybind11::object map_types_to_numba(pybind11::handle types_mod,
                                    mlir::TypeRange types);
