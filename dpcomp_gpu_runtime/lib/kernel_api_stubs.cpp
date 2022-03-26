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

#include "dpcomp-gpu-runtime_export.h"

// Stubs for kernel interface
[[noreturn]] static void stub(const char *funcName) {
  fprintf(stderr, "This function should never be called: %s\n", funcName);
  fflush(stderr);
  abort();
}

#define STUB() stub(__func__)

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

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void _mlir_ciface_kernel_barrier(int64_t) {
  STUB();
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void
_mlir_ciface_kernel_mem_fence(int64_t) {
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
