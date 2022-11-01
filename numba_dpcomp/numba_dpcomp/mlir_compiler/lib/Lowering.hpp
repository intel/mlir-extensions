// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace pybind11 {
class bytes;
class capsule;
class dict;
class int_;
class object;
class str;
} // namespace pybind11

pybind11::capsule initCompiler(pybind11::dict settings);

pybind11::capsule createModule(pybind11::dict settings);

pybind11::capsule lowerFunction(const pybind11::object &compilationContext,
                                const pybind11::capsule &pyMod,
                                const pybind11::object &funcIr);

pybind11::capsule compileModule(const pybind11::capsule &compiler,
                                const pybind11::object &compilationContext,
                                const pybind11::capsule &pyMod);

void registerSymbol(const pybind11::capsule &compiler,
                    const pybind11::str &name, const pybind11::int_ &ptr);

pybind11::int_ getFunctionPointer(const pybind11::capsule &compiler,
                                  const pybind11::capsule &module,
                                  pybind11::str funcName);

void releaseModule(const pybind11::capsule &compiler,
                   const pybind11::capsule &module);

pybind11::str moduleStr(const pybind11::capsule &pyMod);
