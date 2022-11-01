// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PythonRt.hpp"

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
