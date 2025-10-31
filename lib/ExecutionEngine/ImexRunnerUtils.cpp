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
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/Float16bits.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>

// NOLINTBEGIN(*-identifier-naming)

/// Fills the given 1D unranked memref with the given float value.
template <typename T>
void _mlir_ciface_fillResource1D(UnrankedMemRefType<T> *ptr, // NOLINT
                                 const float value) {
  static_assert(std::is_same_v<T, bf16> || std::is_same_v<T, f16> ||
                std::is_same_v<T, float>);
  DynamicMemRefType<T> Dptr = DynamicMemRefType<T>(*ptr);
  T fill_val(value);
  std::fill(Dptr.begin(), Dptr.end(), fill_val);
}

template <typename T>
void _mlir_ciface_fillResource1DRandom(UnrankedMemRefType<T> *ptr,
                                       const float lower, const float upper,
                                       const bool genInt) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(lower, upper);

  DynamicMemRefType<T> Dptr = DynamicMemRefType<T>(*ptr);
  for (DynamicMemRefIterator<T> i = Dptr.begin(); i != Dptr.end(); ++i) {
    *i = T(genInt ? static_cast<int>(dist(gen)) : dist(gen));
  }
}

template <typename T> void _mlir_ciface_printMemref(UnrankedMemRefType<T> *M) {
  impl::printMemRef(*M);
}

/// Fills the given 1D bf16 memref with the given float value.
extern "C" void
_mlir_ciface_fillResource1DBF16(UnrankedMemRefType<bf16> *ptr, // NOLINT
                                float value) {
  _mlir_ciface_fillResource1D(ptr, value);
}

/// Fills the given 1D f16 memref with the given float value.
extern "C" void
_mlir_ciface_fillResource1DF16(UnrankedMemRefType<f16> *ptr, // NOLINT
                               float value) {
  _mlir_ciface_fillResource1D(ptr, value);
}

/// Fills the given 1D float (f32) memref with the given float value.
extern "C" void
_mlir_ciface_fillResource1DF32(UnrankedMemRefType<float> *ptr, // NOLINT
                               float value) {
  _mlir_ciface_fillResource1D(ptr, value);
}

/// Fills 1D memref of bf16 type with random values uniformly
extern "C" void
_mlir_ciface_fillResource1DRandomBF16(UnrankedMemRefType<bf16> *ptr,
                                      const float lower, const float upper,
                                      const bool genInt) {
  _mlir_ciface_fillResource1DRandom(ptr, lower, upper, genInt);
}

/// Fills 1D memref of f16 type with random values uniformly
extern "C" void
_mlir_ciface_fillResource1DRandomF16(UnrankedMemRefType<f16> *ptr,
                                     const float lower, const float upper,
                                     const bool genInt) {
  _mlir_ciface_fillResource1DRandom(ptr, lower, upper, genInt);
}

/// Fills 1D memref of f32 type with random values uniformly
extern "C" void
_mlir_ciface_fillResource1DRandomF32(UnrankedMemRefType<float> *ptr,
                                     const float lower, const float upper,
                                     const bool genInt) {
  _mlir_ciface_fillResource1DRandom(ptr, lower, upper, genInt);
}

extern "C" void _mlir_ciface_printMemrefBF16(UnrankedMemRefType<bf16> *M) {
  _mlir_ciface_printMemref(M);
}

extern "C" void _mlir_ciface_printMemrefF16(UnrankedMemRefType<f16> *M) {
  _mlir_ciface_printMemref(M);
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

template <typename T> float getFloat(T val) {
  static_assert(std::is_same_v<T, bf16> || std::is_same_v<T, f16> ||
                std::is_same_v<T, float>);
  if constexpr (std::is_same_v<T, bf16>) {
    return bfloat2float(val.bits);
  } else if constexpr (std::is_same_v<T, f16>) {
    return half2float(val.bits);
  } else if constexpr (std::is_same_v<T, float>) {
    return val;
  }
}

// For information on how to Iterate over UnrankedMemRefType, start with
// https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/ExecutionEngine/CRunnerUtils.h
template <typename T>
bool _mlir_ciface_allclose(UnrankedMemRefType<T> *M,
                           UnrankedMemRefType<float> *N) {
  // atol, rtol values copied from
  // https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
  // values may need to adjusted in the future
  const float atol = 1e-04;
  const float rtol = 1e-03;
  DynamicMemRefType<T> DM = DynamicMemRefType<T>(*M);
  DynamicMemRefType<float> DN = DynamicMemRefType<float>(*N);
  DynamicMemRefIterator<T> i = DM.begin();
  DynamicMemRefIterator<float> j = DN.begin();
  for (; i != DM.end() && j != DN.end(); ++i, ++j) {
    float lhs = getFloat(*i);
    float rhs = *j;
    if (fabs(lhs - rhs) > atol + rtol * fabs(rhs)) {
      return false;
    }
  }
  return true;
}

template <typename T>
void _mlir_ciface_printAllclose(UnrankedMemRefType<T> *M,
                                UnrankedMemRefType<float> *N) {
  if (_mlir_ciface_allclose(M, N)) {
    std::cout << "[ALLCLOSE: TRUE]\n";
  } else {
    std::cout << "[ALLCLOSE: FALSE]\n";
  }
}

template <typename T>
void _mlir_ciface_printMaxError(UnrankedMemRefType<T> *M,
                                UnrankedMemRefType<T> *N) {
  DynamicMemRefType<T> DM = DynamicMemRefType<T>(*M);
  DynamicMemRefType<T> DN = DynamicMemRefType<T>(*N);
  DynamicMemRefIterator<T> i = DM.begin();
  DynamicMemRefIterator<T> j = DN.begin();
  std::pair<double, DynamicMemRefIterator<T>> max_rel_err_idx{0.0, DM.begin()};
  std::pair<double, DynamicMemRefIterator<T>> max_abs_err_idx{0.0, DM.begin()};
  uint64_t idx = 0;
  double max_rel_error_i = std::numeric_limits<double>::infinity(),
         max_rel_error_j = std::numeric_limits<double>::infinity();
  double max_abs_error_i = std::numeric_limits<double>::infinity(),
         max_abs_error_j = std::numeric_limits<double>::infinity();
  for (; i != DM.end() && j != DN.end(); ++i, ++j, ++idx) {
    const double i_val = getFloat(*i);
    const double j_val = getFloat(*j);
    const double delta = fabs(i_val - j_val);
    const double rel_error = delta / fmax(fabs(i_val), fabs(j_val));
    if (delta > max_abs_err_idx.first) {
      max_abs_err_idx = {delta, i};
      max_abs_error_i = i_val;
      max_abs_error_j = j_val;
    }
    if (rel_error > max_rel_err_idx.first) {
      max_rel_err_idx = {rel_error, i};
      max_rel_error_i = i_val;
      max_rel_error_j = j_val;
    }
  }
  std::cout << "Max absolute error " << max_abs_err_idx.first
            << " at idx=" << std::distance(DM.begin(), max_abs_err_idx.second)
            << " (i=" << max_abs_error_i << ", j=" << max_abs_error_j << ")"
            << '\n';
  std::cout << "Max relative error " << max_rel_err_idx.first
            << " at idx=" << std::distance(DM.begin(), max_rel_err_idx.second)
            << " (i=" << max_rel_error_i << ", j=" << max_rel_error_j << ")"
            << '\n';
}

template <typename ABTy, typename CTy>
void _mlir_ciface_gemm(UnrankedMemRefType<ABTy> *A, UnrankedMemRefType<ABTy> *B,
                       UnrankedMemRefType<float> *C) {
  DynamicMemRefType<ABTy> DA = DynamicMemRefType<ABTy>(*A);
  DynamicMemRefType<ABTy> DB = DynamicMemRefType<ABTy>(*B);
  DynamicMemRefType<float> DC = DynamicMemRefType<float>(*C);
  if (DA.rank != 2 || DB.rank != 2 || DC.rank != 2) {
    std::cout << "Expecting 2D memrefs, got A rank: " << DA.rank
              << ", B rank: " << DB.rank << ", C rank: " << DC.rank << '\n';
    return;
  }
  if (DA.sizes[1] != DB.sizes[0] || DA.sizes[0] != DC.sizes[0] ||
      DB.sizes[1] != DC.sizes[1]) {
    std::cout << "Incompatible matrix dimensions: A: [" << DA.sizes[0] << ", "
              << DA.sizes[1] << "], B: [" << DB.sizes[0] << ", " << DB.sizes[1]
              << "], C: [" << DC.sizes[0] << ", " << DC.sizes[1] << "]\n";
    return;
  }
  if ((DA.strides[0] != DA.sizes[1]) || (DA.strides[1] != 1) ||
      (DB.strides[0] != DB.sizes[1]) || (DB.strides[1] != 1) ||
      (DC.strides[0] != DC.sizes[1]) || (DC.strides[1] != 1)) {
    std::cout << "Expecting A strides to be [M, 1], B strides to be [K, 1], C "
                 "strides to be [M, 1]\n";
    return;
  }
  int M = DA.sizes[0];
  int N = DB.sizes[1];
  int K = DA.sizes[1];

  float *storageA = (float *)malloc(M * K * sizeof(float));
  float *storageB = (float *)malloc(K * N * sizeof(float));
  float *storageC = (float *)malloc(M * N * sizeof(float));

  DynamicMemRefIterator<ABTy> aIt(DA);
  for (int i = 0; i < M * K; ++i) {
    storageA[i] = getFloat(*aIt);
    ++aIt;
  }

  DynamicMemRefIterator<ABTy> bIt(DB);
  for (int i = 0; i < K * N; ++i) {
    storageB[i] = getFloat(*bIt);
    ++bIt;
  }

  DynamicMemRefIterator<float> cIt(DC);
  for (int i = 0; i < M * N; ++i) {
    storageC[i] = getFloat(*cIt);
    ++cIt;
  }

  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        int a_idx = i * DA.strides[0] + k;
        int b_idx = k * DB.strides[0] + j;
        int c_idx = i * DC.strides[0] + j;
        storageC[c_idx] += storageA[a_idx] * storageB[b_idx];
      }
    }
  }

  // Store the result back to C.
  DynamicMemRefIterator<float> resultIt(DC);
  for (int i = 0; i < M * N; ++i) {
    float r = storageC[i];
    if constexpr (std::is_same_v<CTy, f16>) {
      f16 casted(r);
      *resultIt = getFloat(casted);
    } else if constexpr (std::is_same_v<CTy, bf16>) {
      bf16 casted(r);
      *resultIt = getFloat(casted);
    } else if constexpr (std::is_same_v<CTy, float>) {
      *resultIt = storageC[i];
    }
    ++resultIt;
  }

  free(storageC);
  free(storageA);
  free(storageB);
}

extern "C" void _mlir_ciface_printMaxErrorF16(UnrankedMemRefType<f16> *M,
                                              UnrankedMemRefType<f16> *N) {
  _mlir_ciface_printMaxError(M, N);
}

extern "C" void _mlir_ciface_printMaxErrorBF16(UnrankedMemRefType<bf16> *M,
                                               UnrankedMemRefType<bf16> *N) {
  _mlir_ciface_printMaxError(M, N);
}

extern "C" void _mlir_ciface_printMaxErrorF32(UnrankedMemRefType<float> *M,
                                              UnrankedMemRefType<float> *N) {
  _mlir_ciface_printMaxError(M, N);
}

extern "C" bool _mlir_ciface_allcloseF16(UnrankedMemRefType<f16> *M,
                                         UnrankedMemRefType<float> *N) {
  return _mlir_ciface_allclose(M, N);
}

extern "C" bool _mlir_ciface_allcloseBF16(UnrankedMemRefType<bf16> *M,
                                          UnrankedMemRefType<float> *N) {
  return _mlir_ciface_allclose(M, N);
}

extern "C" bool _mlir_ciface_allcloseF32(UnrankedMemRefType<float> *M,
                                         UnrankedMemRefType<float> *N) {
  return _mlir_ciface_allclose(M, N);
}

extern "C" void _mlir_ciface_printAllcloseF16(UnrankedMemRefType<f16> *M,
                                              UnrankedMemRefType<float> *N) {
  _mlir_ciface_printAllclose(M, N);
}

extern "C" void _mlir_ciface_printAllcloseBF16(UnrankedMemRefType<bf16> *M,
                                               UnrankedMemRefType<float> *N) {
  _mlir_ciface_printAllclose(M, N);
}

extern "C" void _mlir_ciface_printAllcloseF32(UnrankedMemRefType<float> *M,
                                              UnrankedMemRefType<float> *N) {
  _mlir_ciface_printAllclose(M, N);
}

extern "C" void _mlir_ciface_gemmF16F16F32(UnrankedMemRefType<f16> *A,
                                           UnrankedMemRefType<f16> *B,
                                           UnrankedMemRefType<float> *C) {
  _mlir_ciface_gemm<f16, float>(A, B, C);
}

extern "C" void _mlir_ciface_gemmF16F16F16(UnrankedMemRefType<f16> *A,
                                           UnrankedMemRefType<f16> *B,
                                           UnrankedMemRefType<float> *C) {
  _mlir_ciface_gemm<f16, f16>(A, B, C);
}

extern "C" void _mlir_ciface_gemmBF16BF16F32(UnrankedMemRefType<bf16> *A,
                                             UnrankedMemRefType<bf16> *B,
                                             UnrankedMemRefType<float> *C) {
  _mlir_ciface_gemm<bf16, float>(A, B, C);
}

// NOLINTEND(*-identifier-naming)
