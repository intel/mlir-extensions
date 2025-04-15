//===- ImexExtension.cpp - Extension module -------------------------------===//
//
// Copyright 2025 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "imex-c/Dialects.h"
#include "imex-c/Passes.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

NB_MODULE(_imex_mlir, m) {
  m.doc() = "Intel Extension for MLIR (IMEX) Python binding";

  mlirRegisterAllIMEXPasses();

  //===--------------------------------------------------------------------===//
  // region dialect
  //===--------------------------------------------------------------------===//
  auto regionM = m.def_submodule("region");

  regionM.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__region__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);
}
