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

#include "py_map_types.hpp"

#include <pybind11/pybind11.h>

#include <mlir/IR/TypeRange.h>
#include <mlir/IR/BuiltinTypes.h>

#include "plier/dialect.hpp"

namespace py = pybind11;

namespace
{
template<unsigned Width, mlir::IntegerType::SignednessSemantics Signed>
bool is_int(mlir::Type type)
{
    if (auto t = type.dyn_cast<mlir::IntegerType>())
    {
        if (t.getWidth() == Width && t.getSignedness() == Signed)
        {
            return true;
        }
    }
    return false;
}

template<unsigned Width>
bool is_float(mlir::Type type)
{
    if (auto f = type.dyn_cast<mlir::FloatType>())
    {
        if (f.getWidth() == Width)
        {
            return true;
        }
    }
    return false;
}

bool is_none(mlir::Type type)
{
    return type == plier::PyType::getNone(type.getContext());
}

py::object map_type(const py::handle& types_mod, mlir::Type type)
{
    using fptr_t = bool(*)(mlir::Type);
    const std::pair<fptr_t, llvm::StringRef> primitive_types[] = {
        {&is_int<1, mlir::IntegerType::Signed>,   "boolean"},
        {&is_int<1, mlir::IntegerType::Signless>, "boolean"},
        {&is_int<1, mlir::IntegerType::Unsigned>, "boolean"},

        {&is_int<8, mlir::IntegerType::Signed>,    "int8"},
        {&is_int<8, mlir::IntegerType::Signless>,  "int8"},
        {&is_int<8, mlir::IntegerType::Unsigned>, "uint8"},

        {&is_int<16, mlir::IntegerType::Signed>,    "int16"},
        {&is_int<16, mlir::IntegerType::Signless>,  "int16"},
        {&is_int<16, mlir::IntegerType::Unsigned>, "uint16"},

        {&is_int<32, mlir::IntegerType::Signed>,    "int32"},
        {&is_int<32, mlir::IntegerType::Signless>,  "int32"},
        {&is_int<32, mlir::IntegerType::Unsigned>, "uint32"},

        {&is_int<64, mlir::IntegerType::Signed>,    "int64"},
        {&is_int<64, mlir::IntegerType::Signless>,  "int64"},
        {&is_int<64, mlir::IntegerType::Unsigned>, "uint64"},

        {&is_float<32>, "float32"},
        {&is_float<64>, "float64"},

        {&is_none, "none"},
    };

    for (auto h : primitive_types)
    {
        if (h.first(type))
        {
            auto name = h.second;
            return types_mod.attr(py::str(name.data(), name.size()));
        }
    }

    if (auto m = type.dyn_cast<mlir::ShapedType>())
    {
        auto elem_type = map_type(types_mod, m.getElementType());
        if (!elem_type)
        {
            return {};
        }
        auto ndims = py::int_(m.getRank());
        auto array_type = types_mod.attr("Array");
        return array_type(elem_type, ndims, py::str("C"));
    }
    return {};
}
}

py::object map_type_to_numba(pybind11::handle types_mod, mlir::Type type)
{
    auto elem = map_type(types_mod, type);
    if (!elem)
    {
        return py::none();
    }
    return elem;
}

py::object map_types_to_numba(pybind11::handle types_mod, mlir::TypeRange types)
{
    py::list ret(types.size());
    for (auto it : llvm::enumerate(types))
    {
        auto type = map_type_to_numba(types_mod, it.value());
        if (type.is_none())
        {
            return py::none();
        }
        ret[it.index()] = std::move(type);
    }
    return std::move(ret);
}

