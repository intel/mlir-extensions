// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
