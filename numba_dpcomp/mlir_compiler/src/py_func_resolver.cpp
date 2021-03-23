#include "py_func_resolver.hpp"

#include <pybind11/pybind11.h>

#include <mlir/IR/BuiltinOps.h>

#include "py_map_types.hpp"

namespace py = pybind11;

struct PyFuncResolver::Context
{
    py::handle resolver;
    py::handle compiler;
    py::handle types;
};

PyFuncResolver::PyFuncResolver():
    context(std::make_unique<Context>())
{
    auto registry_mod = py::module::import("numba.mlir.func_registry");
    auto compiler_mod = py::module::import("numba.mlir.inner_compiler");
    context->resolver = registry_mod.attr("find_active_func");
    context->compiler = compiler_mod.attr("compile_func");
    context->types = py::module::import("numba.core.types");
}

PyFuncResolver::~PyFuncResolver()
{

}

mlir::FuncOp PyFuncResolver::get_func(llvm::StringRef name, mlir::TypeRange types)
{
    assert(!name.empty());
    auto py_name = py::str(name.data(), name.size());
    auto py_func = context->resolver(py_name);
    if (py_func.is_none())
    {
        return {};
    }
    auto py_types = map_types_to_numba(context->types, types);
    if (py_types.is_none())
    {
        return {};
    }
    auto res = static_cast<mlir::Operation*>(context->compiler(py_func, py_types).cast<py::capsule>());
    auto func = (res ? mlir::cast<mlir::FuncOp>(res) : nullptr);
    return func;
}
