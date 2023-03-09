//===- ImexRunnerUtils.cpp - IMEX Runtime Utilities -----------------------===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file includes runtime functions that can be called from mlir code
///
//===----------------------------------------------------------------------===//

#include "imex/ExecutionEngine/ImexRunnerUtils.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

// NOLINTBEGIN(*-identifier-naming)

extern "C" void _mlir_ciface_printMemrefBF16(UnrankedMemRefType<bf16> *M) {
  impl::printMemRef(*M);
}

extern "C" void _mlir_ciface_printMemrefF16(UnrankedMemRefType<f16> *M) {
  impl::printMemRef(*M);
}

extern "C" void printMemrefBF16(int64_t rank, void *ptr) {
  UnrankedMemRefType<bf16> descriptor = {rank, ptr};
  _mlir_ciface_printMemrefBF16(&descriptor);
}

extern "C" void printMemrefF16(int64_t rank, void *ptr) {
  UnrankedMemRefType<f16> descriptor = {rank, ptr};
  _mlir_ciface_printMemrefF16(&descriptor);
}

// Copied f16 and bf16 conversion code from
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/ExecutionEngine/Float16bits.cpp

// Union used to make the int/float aliasing explicit so we can access the raw
// bits.
union Float32Bits {
  uint32_t u;
  float f;
};

const uint32_t kF32MantiBits = 23;
const uint32_t kF32HalfMantiBitDiff = 13;
const uint32_t kF32HalfBitDiff = 16;
const Float32Bits kF32Magic = {113 << kF32MantiBits};
const uint32_t kF32HalfExpAdjust = (127 - 15) << kF32MantiBits;

// Converts the 16 bit representation of a half precision value to a float
// value. This implementation is adapted from Eigen.
static float half2float(uint16_t halfValue) {
  const uint32_t shiftedExp =
      0x7c00 << kF32HalfMantiBitDiff; // Exponent mask after shift.

  // Initialize the float representation with the exponent/mantissa bits.
  Float32Bits f = {
      static_cast<uint32_t>((halfValue & 0x7fff) << kF32HalfMantiBitDiff)};
  const uint32_t exp = shiftedExp & f.u;
  f.u += kF32HalfExpAdjust; // Adjust the exponent

  // Handle exponent special cases.
  if (exp == shiftedExp) {
    // Inf/NaN
    f.u += kF32HalfExpAdjust;
  } else if (exp == 0) {
    // Zero/Denormal?
    f.u += 1 << kF32MantiBits;
    f.f -= kF32Magic.f;
  }

  f.u |= (halfValue & 0x8000) << kF32HalfBitDiff; // Sign bit.
  return f.f;
}

const uint32_t kF32BfMantiBitDiff = 16;

// Converts the 16 bit representation of a bfloat value to a float value. This
// implementation is adapted from Eigen.
static float bfloat2float(uint16_t bfloatBits) {
  Float32Bits floatBits;
  floatBits.u = static_cast<uint32_t>(bfloatBits) << kF32BfMantiBitDiff;
  return floatBits.f;
}

// For information on how to Iterate over UnrankedMemRefType, start with
// https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/ExecutionEngine/CRunnerUtils.h
extern "C" bool _mlir_ciface_allcloseF16(UnrankedMemRefType<f16> *M,
                                         UnrankedMemRefType<float> *N) {
  // atol, rtol values copied from
  // https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
  // values may need to adjusted in the future
  const float atol = 1e-04;
  const float rtol = 1e-03;
  DynamicMemRefType<f16> DM = DynamicMemRefType<f16>(*M);
  DynamicMemRefType<float> DN = DynamicMemRefType<float>(*N);
  DynamicMemRefIterator<f16> i = DM.begin();
  DynamicMemRefIterator<float> j = DN.begin();
  for (; i != DM.end() && j != DN.end(); ++i, ++j) {
    f16 lhs = *i;
    float rhs = *j;
    if (fabs(half2float(lhs.bits) - rhs) > atol + rtol * fabs(rhs)) {
      return false;
    }
  }
  return true;
}

extern "C" bool _mlir_ciface_allcloseBF16(UnrankedMemRefType<bf16> *M,
                                          UnrankedMemRefType<float> *N) {
  // atol, rtol values copied from
  // https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
  // values may need to adjusted in the future
  const float atol = 1e-08;
  const float rtol = 1e-05;
  DynamicMemRefType<bf16> DM = DynamicMemRefType<bf16>(*M);
  DynamicMemRefType<float> DN = DynamicMemRefType<float>(*N);
  DynamicMemRefIterator<bf16> i = DM.begin();
  DynamicMemRefIterator<float> j = DN.begin();
  for (; i != DM.end() && j != DN.end(); ++i, ++j) {
    bf16 lhs = *i;
    float rhs = *j;
    if (fabs(bfloat2float(lhs.bits) - rhs) > atol + rtol * fabs(rhs)) {
      return false;
    }
  }
  return true;
}

extern "C" bool _mlir_ciface_allcloseF32(UnrankedMemRefType<float> *M,
                                         UnrankedMemRefType<float> *N) {
  // atol, rtol values copied from
  // https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
  // values may need to adjusted in the future
  const float atol = 1e-08;
  const float rtol = 1e-05;
  DynamicMemRefType<float> DM = DynamicMemRefType<float>(*M);
  DynamicMemRefType<float> DN = DynamicMemRefType<float>(*N);
  DynamicMemRefIterator<float> i = DM.begin();
  DynamicMemRefIterator<float> j = DN.begin();
  for (; i != DM.end() && j != DN.end(); ++i, ++j) {
    float lhs = *i;
    float rhs = *j;
    if (fabs(lhs - rhs) > atol + rtol * fabs(rhs)) {
      return false;
    }
  }
  return true;
}

extern "C" void _mlir_ciface_printAllcloseF16(UnrankedMemRefType<f16> *M,
                                              UnrankedMemRefType<float> *N) {
  if (_mlir_ciface_allcloseF16(M, N)) {
    std::cout << "[ALLCLOSE: TRUE]\n";
  } else {
    std::cout << "[ALLCLOSE: FALSE]\n";
  }
}

extern "C" void _mlir_ciface_printAllcloseBF16(UnrankedMemRefType<bf16> *M,
                                               UnrankedMemRefType<float> *N) {
  if (_mlir_ciface_allcloseBF16(M, N)) {
    std::cout << "[ALLCLOSE: TRUE]\n";
  } else {
    std::cout << "[ALLCLOSE: FALSE]\n";
  }
}

extern "C" void _mlir_ciface_printAllcloseF32(UnrankedMemRefType<float> *M,
                                              UnrankedMemRefType<float> *N) {
  if (_mlir_ciface_allcloseF32(M, N)) {
    std::cout << "[ALLCLOSE: TRUE]\n";
  } else {
    std::cout << "[ALLCLOSE: FALSE]\n";
  }
}

// NOLINTEND(*-identifier-naming)
