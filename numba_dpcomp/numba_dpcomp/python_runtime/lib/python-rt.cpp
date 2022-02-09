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

#include "python-rt.hpp"

#include <cstdlib>

// TODO: get rid of this definition
struct MemInfo {
  size_t refct;
  MemInfoDtorFunction dtor;
  void *dtor_info;
  void *data;
  size_t size;
  void *external_allocator;
};

using AllocFuncT = void *(*)(size_t);

// TODO: expose NRT_MemInfo_new from numba runtime
static AllocFuncT AllocFunc = nullptr;

extern "C" DPCOMP_PYTHON_RUNTIME_EXPORT void
dpcompSetMemInfoAllocFunc(void *func) {
  AllocFunc = reinterpret_cast<AllocFuncT>(func);
}

extern "C" DPCOMP_PYTHON_RUNTIME_EXPORT void *
dpcompAllocMemInfo(void *data, size_t size, MemInfoDtorFunction dtor,
                   void *dtorInfo) {
  if (!AllocFunc)
    return nullptr;

  auto meminfo = static_cast<MemInfo *>(AllocFunc(sizeof(MemInfo)));
  if (!meminfo)
    return nullptr;

  *meminfo = {};

  meminfo->refct = 1;
  meminfo->dtor = dtor;
  meminfo->dtor_info = dtorInfo;
  meminfo->data = data;
  meminfo->size = size;
  return meminfo;
}
