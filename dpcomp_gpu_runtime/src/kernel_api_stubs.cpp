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
_mlir_ciface_get_global_size(int64_t) {
  STUB();
}
