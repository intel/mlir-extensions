// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <numeric>

#include "dpcomp-gpu-runtime_export.h"

// Stubs for kernel interface
[[noreturn]] static void stub(const char *funcName) {
  fprintf(stderr, "This function should never be called: %s\n", funcName);
  fflush(stderr);
  abort();
}

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

#define STUB() stub(__func__)

/// Fills the given 1D float memref with the given float value.
extern "C" DPCOMP_GPU_RUNTIME_EXPORT void
_mlir_ciface_fillResource1DFloat(MemRefDescriptor<float, 1> *ptr, // NOLINT
                                 float value) {
  std::fill_n(ptr->allocated, ptr->sizes[0], value);
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT int64_t
_mlir_ciface_get_global_id(int64_t) {
  STUB();
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT int64_t
_mlir_ciface_get_local_id(int64_t) {
  STUB();
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT int64_t
_mlir_ciface_get_global_size(int64_t) {
  STUB();
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT int64_t
_mlir_ciface_get_local_size(int64_t) {
  STUB();
}

#define ATOMIC_FUNC_DECL(op, suff, dt)                                         \
  extern "C" DPCOMP_GPU_RUNTIME_EXPORT dt _mlir_ciface_atomic_##op##_##suff(   \
      void *, dt) {                                                            \
    STUB();                                                                    \
  }

#define ATOMIC_FUNC_DECL2(op)                                                  \
  ATOMIC_FUNC_DECL(op, int32, int32_t)                                         \
  ATOMIC_FUNC_DECL(op, int64, int64_t)                                         \
  ATOMIC_FUNC_DECL(op, float32, float)                                         \
  ATOMIC_FUNC_DECL(op, float64, double)

ATOMIC_FUNC_DECL2(add)
ATOMIC_FUNC_DECL2(sub)

#undef ATOMIC_FUNC_DECL2
#undef ATOMIC_FUNC_DECL
