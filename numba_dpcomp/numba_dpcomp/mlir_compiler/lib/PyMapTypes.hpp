// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace pybind11 {
class list;
class object;
class handle;
} // namespace pybind11

namespace mlir {
class Type;
class TypeRange;
} // namespace mlir

pybind11::object mapTypeToNumba(pybind11::handle typesMod, mlir::Type type);
pybind11::object mapTypesToNumba(pybind11::handle typesMod,
                                 mlir::TypeRange types);
