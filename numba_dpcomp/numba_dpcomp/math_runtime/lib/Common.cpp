// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Common.hpp"
#include "dpcomp-math-runtime_export.h"

extern "C" {
DPCOMP_MATH_RUNTIME_EXPORT void dpcompMathRuntimeInit() {
  // Nothing
}

DPCOMP_MATH_RUNTIME_EXPORT void dpcompMathRuntimeFinalize() {
  // Nothing
}
}
