//===- Dialects.cpp - IMEX CAPI Registration --------------------*- C++ -*-===//
//
// Copyright 2025 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "imex-c/Dialects.h"
#include "imex/Dialect/Region/IR/RegionOps.h"
#include "mlir/CAPI/Registration.h"

// name, namespace, classname
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Region, region,
                                      imex::region::RegionDialect)
