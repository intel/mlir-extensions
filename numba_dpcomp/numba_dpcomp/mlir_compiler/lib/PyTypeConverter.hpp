// Copyright 2022 Intel Corporation
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

#include <functional>

#include <llvm/ADT/Optional.h>
#include <llvm/ADT/SmallVector.h>

namespace mlir {
class MLIRContext;
class Type;
} // namespace mlir

namespace pybind11 {
class handle;
}

class PyTypeConverter {
public:
  PyTypeConverter() = default;
  PyTypeConverter(const PyTypeConverter &) = delete;

  using Conversion = std::function<llvm::Optional<mlir::Type>(
      mlir::MLIRContext &, pybind11::handle)>;

  void addConversion(Conversion conv);

  mlir::Type convertType(mlir::MLIRContext &context,
                         pybind11::handle pyObj) const;

private:
  llvm::SmallVector<Conversion, 0> conversions;
};
