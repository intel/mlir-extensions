// Copyright 2022 Intel Corporation
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
#include <cstring>

#define mlir_c_runner_utils_EXPORTS 1
#include <mlir/ExecutionEngine/CRunnerUtils.h>

#include "sycl-runtime_export.h"

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

/// Fills the given 1D float memref with the given float value.
extern "C" SYCL_RUNTIME_EXPORT void
_mlir_ciface_fillResource1DFloat(MemRefDescriptor<float, 1> *ptr, // NOLINT
                                 float value) {
  std::fill_n(ptr->allocated, ptr->sizes[0], value);
}
