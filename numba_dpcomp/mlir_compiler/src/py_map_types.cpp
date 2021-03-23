#include "py_map_types.hpp"

#include <pybind11/pybind11.h>

#include <mlir/IR/TypeRange.h>
#include <mlir/IR/BuiltinTypes.h>

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
    };

    for (auto h : primitive_types)
    {
        if (h.first(type))
        {
            auto name = h.second;
            return types_mod.attr(py::str(name.data(), name.size()));
        }
    }

    if (auto m = type.dyn_cast<mlir::MemRefType>())
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
pybind11::object map_type_to_numba(pybind11::handle types_mod, mlir::Type type)
{
    auto elem = map_type(types_mod, type);
    if (!elem)
    {
        return py::none();
    }
    return elem;
}

pybind11::list map_types_to_numba(pybind11::handle types_mod, mlir::TypeRange types)
{
    py::list ret(types.size());
    for (auto it : llvm::enumerate(types))
    {
        ret[it.index()] = map_type_to_numba(types_mod, it.value());
    }
    return ret;
}

