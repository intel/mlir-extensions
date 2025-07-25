//===- Passes.h - IMEX CAPI Registration ------------------------*- C++ -*-===//
//
// Copyright 2025 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the IMEX C API passes.
//
//===----------------------------------------------------------------------===//

#ifndef IMEX_MLIR_C_PASSES_H
#define IMEX_MLIR_C_PASSES_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void mlirRegisterAllIMEXPasses(void);

#ifdef __cplusplus
}
#endif
#endif // IMEX_MLIR_C_PASSES_H
