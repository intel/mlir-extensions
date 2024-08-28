//===- ImexRunnerUtils.h - IMEX Runtime Utilities -------------------------===//
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

#ifndef IMEX_EXECUTIONENGINE_IMEXRUNNERUTILS_H
#define IMEX_EXECUTIONENGINE_IMEXRUNNERUTILS_H

#ifdef _WIN32
#ifndef IMEX_RUNNERUTILS_EXPORT
#ifdef imex_runner_utils_EXPORTS
// We are building this library
#define IMEX_RUNNERUTILS_EXPORT __declspec(dllexport)
#else
// We are using this library
#define IMEX_RUNNERUTILS_EXPORT __declspec(dllimport)
#endif // imex_runner_utils_EXPORTS
#endif // IMEX_RUNNERUTILS_EXPORT
#else
// Non-windows: use visibility attributes.
#define IMEX_RUNNERUTILS_EXPORT __attribute__((visibility("default")))
#endif // _WIN32

#include "mlir/ExecutionEngine/Float16bits.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

template <typename T>
void _mlir_ciface_fillResource1D(UnrankedMemRefType<T> *ptr, // NOLINT
                                 const float value);
template <typename T>
void _mlir_ciface_fillResource1DRandom(UnrankedMemRefType<T> *ptr,
                                       const float lower, const float upper,
                                       const bool genInt);

template <typename T> void _mlir_ciface_printMemref(UnrankedMemRefType<T> *M);

template <typename T>
bool _mlir_ciface_allclose(UnrankedMemRefType<T> *M,
                           UnrankedMemRefType<float> *N);

template <typename T>
void _mlir_ciface_printAllclose(UnrankedMemRefType<T> *M,
                                UnrankedMemRefType<float> *N);

template <typename T>
void _mlir_ciface_printMaxError(UnrankedMemRefType<T> *M,
                                UnrankedMemRefType<T> *N);

extern "C" IMEX_RUNNERUTILS_EXPORT void
_mlir_ciface_fillResource1DRandomBF16(UnrankedMemRefType<bf16> *ptr,
                                      const float lower, const float upper,
                                      const bool genInt);
extern "C" IMEX_RUNNERUTILS_EXPORT void
_mlir_ciface_fillResource1DRandomF16(UnrankedMemRefType<f16> *ptr,
                                     const float lower, const float upper,
                                     const bool genInt);

extern "C" IMEX_RUNNERUTILS_EXPORT void
_mlir_ciface_fillResource1DRandomF32(UnrankedMemRefType<float> *ptr,
                                     const float lower, const float upper,
                                     const bool genInt);

extern "C" IMEX_RUNNERUTILS_EXPORT void
_mlir_ciface_printMemrefBF16(UnrankedMemRefType<bf16> *m);
extern "C" IMEX_RUNNERUTILS_EXPORT void
_mlir_ciface_printMemrefF16(UnrankedMemRefType<f16> *m);

extern "C" IMEX_RUNNERUTILS_EXPORT void printMemrefBF16(int64_t rank,
                                                        void *ptr);
extern "C" IMEX_RUNNERUTILS_EXPORT void printMemrefF16(int64_t rank, void *ptr);

extern "C" IMEX_RUNNERUTILS_EXPORT bool
_mlir_ciface_allcloseBF16(UnrankedMemRefType<bf16> *M,
                          UnrankedMemRefType<float> *N);
extern "C" IMEX_RUNNERUTILS_EXPORT bool
_mlir_ciface_allcloseF16(UnrankedMemRefType<f16> *M,
                         UnrankedMemRefType<float> *N);
extern "C" IMEX_RUNNERUTILS_EXPORT bool
_mlir_ciface_allcloseF32(UnrankedMemRefType<float> *M,
                         UnrankedMemRefType<float> *N);

extern "C" IMEX_RUNNERUTILS_EXPORT void
_mlir_ciface_printAllcloseBF16(UnrankedMemRefType<bf16> *M,
                               UnrankedMemRefType<float> *N);
extern "C" IMEX_RUNNERUTILS_EXPORT void
_mlir_ciface_printAllcloseF16(UnrankedMemRefType<f16> *M,
                              UnrankedMemRefType<float> *N);
extern "C" IMEX_RUNNERUTILS_EXPORT void
_mlir_ciface_printAllcloseF32(UnrankedMemRefType<float> *M,
                              UnrankedMemRefType<float> *N);

extern "C" IMEX_RUNNERUTILS_EXPORT void
_mlir_ciface_printMaxErrorF16(UnrankedMemRefType<f16> *M,
                              UnrankedMemRefType<f16> *N);

extern "C" IMEX_RUNNERUTILS_EXPORT void
_mlir_ciface_printMaxErrorBF16(UnrankedMemRefType<bf16> *M,
                               UnrankedMemRefType<bf16> *N);

extern "C" IMEX_RUNNERUTILS_EXPORT void
_mlir_ciface_printMaxErrorF32(UnrankedMemRefType<float> *M,
                              UnrankedMemRefType<float> *N);

#endif // IMEX_EXECUTIONENGINE_IMEXRUNNERUTILS_H
