//===- Passes.cpp - IMEX CAPI Registration ----------------------*- C++ -*-===//
//
// Copyright 2025 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "imex/InitIMEXPasses.h"
#include "mlir-c/Pass.h"
#include "mlir/CAPI/Pass.h"

using namespace imex;

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void mlirRegisterAllIMEXPasses() { registerAllPasses(); }
#ifdef __cplusplus
}
#endif
