// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <array>
#include <cstdlib>

#define fatal_failure(format, ...)                                             \
  do {                                                                         \
    fprintf(stderr, format, ##__VA_ARGS__);                                    \
    fflush(stderr);                                                            \
    abort();                                                                   \
  } while (0)

template <size_t NumDims, typename T> struct Memref {
  void *userData;
  T *data;
  size_t offset;
  std::array<size_t, NumDims> dims;
  std::array<size_t, NumDims> strides;
};

template <size_t NumDims, typename T>
static T *getMemrefData(const Memref<NumDims, T> *src) {
  return src->data + src->offset;
}
