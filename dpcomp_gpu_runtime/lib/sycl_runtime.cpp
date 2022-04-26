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

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "dpcomp-gpu-runtime_export.h"

#include <CL/sycl.hpp>

namespace {

template <typename F> auto catchAll(F &&func) {
  try {
    return func();
  } catch (const std::exception &e) {
    fprintf(stdout, "An exception was thrown: %s\n", e.what());
    fflush(stdout);
    abort();
  } catch (...) {
    fprintf(stdout, "An unknown exception was thrown\n");
    fflush(stdout);
    abort();
  }
}

} // namespace

struct Sycl_Struct {
  Sycl_Struct(){};

  // Get the SYCL device
  sycl::device *get_device(const char *platform = "", size_t ordinal = 0) {
    sycl::device gpu_device;
    gpu_device = sycl::device(sycl::gpu_selector());
    return &gpu_device;
  }

private:
};

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *sycl_gpuGetDevice() {
  return catchAll([&]() {
    Sycl_Struct sycl_struct;
    return sycl_struct.get_device();
  });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *sycl_gpuCreateContext(void *device) {
  return catchAll([&]() { return nullptr; });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *sycl_gpuCreateStream(void *context) {
  return catchAll([&]() { return nullptr; });
}