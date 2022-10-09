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

#include "NumpyResolver.hpp"

#include <pybind11/pybind11.h>

#include "llvm/ADT/StringRef.h"

namespace py = pybind11;

class NumpyResolver::Impl {
public:
  py::module mod;
  py::object map;

  py::object getFuncDesc(llvm::StringRef name) {
    return map(py::str(name.data(), name.size()));
  }
};

NumpyResolver::NumpyResolver(const char *modName, const char *mapName)
    : impl(std::make_unique<Impl>()) {
  impl->mod = py::module::import(modName);
  impl->map = impl->mod.attr(mapName);
}

NumpyResolver::~NumpyResolver() {}

bool NumpyResolver::hasFunc(llvm::StringRef name) const {
  return !impl->getFuncDesc(name).is_none();
}
