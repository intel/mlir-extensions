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

#include "py_func_resolver.hpp"

#include <pybind11/pybind11.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include "py_map_types.hpp"

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
