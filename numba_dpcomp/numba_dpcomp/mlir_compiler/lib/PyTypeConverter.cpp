// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PyTypeConverter.hpp"

#include <pybind11/pybind11.h>

#include <mlir/IR/Types.h>

void PyTypeConverter::addConversion(Conversion conv) {
  assert(conv && "Invalid conversion func");
  conversions.emplace_back(std::move(conv));
}

mlir::Type PyTypeConverter::convertType(mlir::MLIRContext &context,
                                        pybind11::handle pyObj) const {
  // Iterate in reverse order to follow mlir type conversion behavior.
  for (auto &conv : llvm::reverse(conversions)) {
    assert(conv && "Invalid conversion func");
    llvm::Optional<mlir::Type> result = conv(context, pyObj);
    if (!result)
      continue;

    return *result;
  }
  return nullptr;
}
