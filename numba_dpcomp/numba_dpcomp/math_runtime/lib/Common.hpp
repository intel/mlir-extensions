// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <array>
#include <cstdlib>

#include "dpcomp-math-runtime_export.h"

template <size_t NumDims, typename T> struct Memref {
  void *userData;
  T *data;
  size_t offset;
  std::array<size_t, NumDims> dims;
  std::array<size_t, NumDims> strides;
};
