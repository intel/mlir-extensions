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

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "dpcomp-runtime_export.h"

using init_func_t = void (*)(void *);
using release_func_t = void (*)(void *);

namespace {
struct Context {
  release_func_t releaseFunc;
  std::aligned_storage_t<8> data;
};
} // namespace

static void *toData(void *ptr) {
  return &(reinterpret_cast<Context *>(ptr)->data);
}

extern "C" DPCOMP_RUNTIME_EXPORT void *
dpcompTakeContext(void **ctxHandle, size_t contextSize, init_func_t init,
                  release_func_t release) {
  assert(ctxHandle);
  assert(contextSize > 0);
  if (!*ctxHandle) {
    auto inlineSize = sizeof(Context::data);
    auto bytesToAlloc = contextSize <= inlineSize
                            ? sizeof(Context)
                            : sizeof(Context) + contextSize - inlineSize;
    auto &context = *reinterpret_cast<Context *>(new char[bytesToAlloc]);
    if (init)
      init(&context.data);
    context.releaseFunc = release;
    *ctxHandle = &context;
  }
  return toData(*ctxHandle);
}

extern "C" DPCOMP_RUNTIME_EXPORT void dpcompReleaseContext(void * /*context*/) {
  // Nothing For now
}

extern "C" DPCOMP_RUNTIME_EXPORT void dpcompPurgeContext(void **ctxHandle) {
  assert(ctxHandle);
  if (*ctxHandle) {
    auto &context = *reinterpret_cast<Context *>(*ctxHandle);
    auto release = context.releaseFunc;
    if (release)
      release(&context.data);

    delete[] reinterpret_cast<char *>(&context);
    *ctxHandle = nullptr;
  }
}
