#include <pybind11/pybind11.h>

#include "py_module.hpp"

#include "lowering.hpp"

PYBIND11_MODULE(mlir_compiler, m)
{
    m.def("create_module", &create_module, "todo");
    m.def("lower_function", &lower_function, "todo");
    m.def("compile_module", &compile_module, "todo");
    m.def("module_str", &module_str, "todo");
}
