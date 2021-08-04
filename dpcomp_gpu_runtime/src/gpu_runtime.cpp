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

#include "dpcomp-gpu-runtime_export.h"

#if 1 // Log functions
namespace {
struct FuncScope {
  FuncScope(const char *funcName) : name(funcName) {
    fprintf(stdout, "%s enter\n", name);
    fflush(stdout);
  }

  ~FuncScope() {
    fprintf(stdout, "%s exit\n", name);
    fflush(stdout);
  }

private:
  const char *name;
};
} // namespace
#define LOG_FUNC() FuncScope _scope(__func__)
#else
#define LOG_FUNC() (void)0
#endif

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *mgpuModuleLoad(void *data) {
  LOG_FUNC();
  return nullptr;
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void mgpuModuleUnload(void *module) {
  LOG_FUNC();
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *
mgpuModuleGetFunction(void *module, const char *name) {
  LOG_FUNC();
  return nullptr;
}

// The wrapper uses intptr_t instead of CUDA's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" DPCOMP_GPU_RUNTIME_EXPORT void
mgpuLaunchKernel(void *function, intptr_t gridX, intptr_t gridY, intptr_t gridZ,
                 intptr_t blockX, intptr_t blockY, intptr_t blockZ,
                 int32_t smem, void *stream, void **params, void **extra) {
  LOG_FUNC();
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *mgpuStreamCreate() {
  LOG_FUNC();
  return nullptr;
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void mgpuStreamDestroy(void *stream) {
  LOG_FUNC();
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void mgpuStreamSynchronize(void *stream) {
  LOG_FUNC();
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void mgpuStreamWaitEvent(void *stream,
                                                              void *event) {
  LOG_FUNC();
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *mgpuEventCreate() {
  LOG_FUNC();
  return nullptr;
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void mgpuEventDestroy(void *event) {
  LOG_FUNC();
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void mgpuEventSynchronize(void *event) {
  LOG_FUNC();
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void mgpuEventRecord(void *event,
                                                          void *stream) {
  LOG_FUNC();
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *mgpuMemAlloc(uint64_t sizeBytes,
                                                        void *stream) {
  LOG_FUNC();
  return nullptr;
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void mgpuMemFree(void *ptr, void *stream) {
  LOG_FUNC();
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void
mgpuMemcpy(void *dst, void *src, uint64_t sizeBytes, void *stream) {
  LOG_FUNC();
}
