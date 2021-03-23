#pragma once

namespace pybind11
{
class list;
class object;
class handle;
}

namespace mlir
{
class Type;
class TypeRange;
}

pybind11::object map_type_to_numba(pybind11::handle types_mod, mlir::Type type);
pybind11::list map_types_to_numba(pybind11::handle types_mod, mlir::TypeRange types);
