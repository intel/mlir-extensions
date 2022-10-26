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

#include "sycl-runtime_export.h"

#include <CL/sycl.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

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

#define L0_SAFE_CALL(call)                                                     \
  {                                                                            \
    ze_result_t status = (call);                                               \
    if (status != ZE_RESULT_SUCCESS) {                                         \
      std::cout << "L0 error " << status << std::endl;                         \
      exit(1);                                                                 \
    }                                                                          \
  }

} // namespace

struct ParamDesc {
  void *data;
  size_t size;

  bool operator==(const ParamDesc &rhs) const {
    return data == rhs.data && size == rhs.size;
  }

  bool operator!=(const ParamDesc &rhs) const { return !(*this == rhs); }
};

template <typename T> size_t countUntil(T *ptr, T &&elem) {
  assert(ptr);
  auto curr = ptr;
  while (*curr != elem) {
    ++curr;
  }
  return static_cast<size_t>(curr - ptr);
}

sycl::device get_default_device() {
  auto platform_list = sycl::platform::get_platforms();
  for (const auto &platform : platform_list) {
    auto platform_name = platform.get_info<sycl::info::platform::name>();
    bool is_level_zero = platform_name.find("Level-Zero") != std::string::npos;
    if (!is_level_zero)
      continue;

    return platform.get_devices()[0];
  }
}

struct GPU_SYCL_QUEUE {

  sycl::device sycl_device_;
  sycl::context sycl_context_;
  sycl::queue sycl_queue_;

  GPU_SYCL_QUEUE() {

    sycl_device_ = get_default_device();
    sycl_context_ = sycl::context(sycl_device_);
    sycl_queue_ = sycl::queue(sycl_context_, sycl_device_);
  }

  GPU_SYCL_QUEUE(sycl::device *device, sycl::context *context) {
    sycl_device_ = *device;
    sycl_context_ = *context;
    sycl_queue_ = sycl::queue(sycl_context_, sycl_device_);
  }
  GPU_SYCL_QUEUE(sycl::device *device) {

    sycl_device_ = *device;
    sycl_context_ = sycl::context(sycl_device_);
    sycl_queue_ = sycl::queue(sycl_context_, sycl_device_);
  }

  GPU_SYCL_QUEUE(sycl::context *context) {

    sycl_device_ = get_default_device();
    sycl_context_ = *context;
    sycl_queue_ = sycl::queue(sycl_context_, sycl_device_);
  }

}; // end of GPU_SYCL_QUEUE

void *alloc_device_memory(void *queue, size_t size, size_t alignment,
                          bool is_shared) {
  void *mem_ptr = nullptr;
  if (is_shared) {
    mem_ptr = sycl::aligned_alloc_shared(
        alignment, size, static_cast<GPU_SYCL_QUEUE *>(queue)->sycl_queue_);
  } else {
    mem_ptr = sycl::aligned_alloc_device(
        alignment, size, static_cast<GPU_SYCL_QUEUE *>(queue)->sycl_queue_);
  }
  return mem_ptr;
}

void dealloc_device_memory(void *queue, void *ptr) {
  sycl::free(ptr, static_cast<GPU_SYCL_QUEUE *>(queue)->sycl_queue_);
}

ze_module_handle_t loadModule(void *queue, const void *data, size_t dataSize) {
  assert(data);
  auto sycl_queue = static_cast<GPU_SYCL_QUEUE *>(queue)->sycl_queue_;
  ze_module_handle_t ze_module;
  ze_module_desc_t desc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                           nullptr,
                           ZE_MODULE_FORMAT_IL_SPIRV,
                           dataSize,
                           (const uint8_t *)data,
                           nullptr,
                           nullptr};
  auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      sycl_queue.get_device());
  auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      sycl_queue.get_context());
  L0_SAFE_CALL(zeModuleCreate(ze_ctx, ze_device, &desc, &ze_module, nullptr));
  return ze_module;
}

void *getKernel(void *queue, void *module, const char *name) {
  assert(module);
  assert(name);
  auto sycl_queue = static_cast<GPU_SYCL_QUEUE *>(queue)->sycl_queue_;
  ze_module_handle_t ze_module = static_cast<ze_module_handle_t>(module);
  ze_kernel_handle_t ze_kernel;
  sycl::kernel *sycl_kernel;
  ze_kernel_desc_t desc = {};
  desc.pKernelName = name;
  L0_SAFE_CALL(zeKernelCreate(ze_module, &desc, &ze_kernel));

  sycl::kernel_bundle<sycl::bundle_state::executable> kernel_bundle =
      sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                               sycl::bundle_state::executable>(
          {ze_module}, sycl_queue.get_context());

  auto kernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {kernel_bundle, ze_kernel}, sycl_queue.get_context());
  sycl_kernel = new sycl::kernel(kernel);
  return sycl_kernel;
}

void launchKernel(void *queue, sycl::kernel *kernel, size_t gridX, size_t gridY,
                  size_t gridZ, size_t blockX, size_t blockY, size_t blockZ,
                  size_t sharedMemBytes, ParamDesc *params, void *extra) {
  auto sycl_queue = static_cast<GPU_SYCL_QUEUE *>(queue)->sycl_queue_;
  auto sycl_global_range =
      ::sycl::range<3>(blockZ * gridZ, blockY * gridY, blockX * gridX);
  auto sycl_local_range = ::sycl::range<3>(blockZ, blockY, blockX);
  sycl::nd_range<3> sycl_nd_range(
      sycl::nd_range<3>(sycl_global_range, sycl_local_range));

  auto paramsCount = countUntil(params, ParamDesc{nullptr, 0});

  sycl_queue.submit([&](sycl::handler &cgh) {
    for (size_t i = 0; i < paramsCount; i++) {
      auto param = params[i];
      cgh.set_arg(static_cast<uint32_t>(i),
                  *(static_cast<void **>(param.data)));
    }
    cgh.parallel_for(sycl_nd_range, *kernel);
  });
}

// Wrappers

extern "C" SYCL_RUNTIME_EXPORT void *gpuCreateStream(void *device,
                                                     void *context) {
  return catchAll([&]() {
    if (!device && !context) {
      return new GPU_SYCL_QUEUE();
    } else if (device && context) {
      return new GPU_SYCL_QUEUE(static_cast<sycl::device *>(device),
                                static_cast<sycl::context *>(context));
    } else if (device && !context) {
      return new GPU_SYCL_QUEUE(static_cast<sycl::device *>(device));
    } else {
      return new GPU_SYCL_QUEUE(static_cast<sycl::context *>(context));
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT void gpuStreamDestroy(void *queue) {
  catchAll([&]() { delete static_cast<GPU_SYCL_QUEUE *>(queue); });
}

extern "C" SYCL_RUNTIME_EXPORT void *
gpuMemAlloc(void *queue, size_t size, size_t alignment, bool is_shared) {
  return catchAll(
      [&]() { return alloc_device_memory(queue, size, alignment, is_shared); });
}

extern "C" SYCL_RUNTIME_EXPORT void gpuMemFree(void *queue, void *ptr) {
  catchAll([&]() { dealloc_device_memory(queue, ptr); });
}

extern "C" SYCL_RUNTIME_EXPORT void *
gpuModuleLoad(void *queue, const void *data, size_t dataSize) {
  return catchAll([&]() { return loadModule(queue, data, dataSize); });
}

extern "C" SYCL_RUNTIME_EXPORT void *gpuKernelGet(void *queue, void *module,
                                                  const char *name) {
  return catchAll([&]() { return getKernel(queue, module, name); });
}

extern "C" SYCL_RUNTIME_EXPORT void
gpuLaunchKernel(void *queue, void *kernel, size_t gridX, size_t gridY,
                size_t gridZ, size_t blockX, size_t blockY, size_t blockZ,
                size_t sharedMemBytes, void *params, void *extra) {
  return catchAll([&]() {
    launchKernel(queue, static_cast<sycl::kernel *>(kernel), gridX, gridY,
                 gridZ, blockX, blockY, blockZ, sharedMemBytes,
                 static_cast<ParamDesc *>(params), extra);
  });
}

extern "C" SYCL_RUNTIME_EXPORT void gpuWait(void *queue) {

  catchAll([&]() { static_cast<GPU_SYCL_QUEUE *>(queue)->sycl_queue_.wait(); });
}
