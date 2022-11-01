// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PyFuncResolver.hpp"

#include "PyMapTypes.hpp"

#include <pybind11/pybind11.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>

namespace py = pybind11;

struct PyFuncResolver::Context {
  py::handle resolver;
  py::handle compiler;
  py::handle types;
};

PyFuncResolver::PyFuncResolver() : context(std::make_unique<Context>()) {
  auto registryMod = py::module::import("numba_dpcomp.mlir.func_registry");
  auto compilerMod = py::module::import("numba_dpcomp.mlir.inner_compiler");
  context->resolver = registryMod.attr("find_active_func");
  context->compiler = compilerMod.attr("compile_func");
  context->types = py::module::import("numba.core.types");
}

PyFuncResolver::~PyFuncResolver() {}

mlir::func::FuncOp PyFuncResolver::getFunc(llvm::StringRef name,
                                           mlir::TypeRange types) const {
  assert(!name.empty());
  auto funcDesc = context->resolver(py::str(name.data(), name.size()));
  if (funcDesc.is_none())
    return {};

  auto funcDescTuple = funcDesc.cast<py::tuple>();

  auto pyFunc = funcDescTuple[0];
  auto flags = funcDescTuple[1];
  auto pyTypes = mapTypesToNumba(context->types, types);
  if (pyTypes.is_none())
    return {};

  auto res = static_cast<mlir::Operation *>(
      context->compiler(pyFunc, pyTypes, flags).cast<py::capsule>());
  return mlir::cast_or_null<mlir::func::FuncOp>(res);
}
