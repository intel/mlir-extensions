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
    if (status != 0) {                                                         \
      std::cout << "L0 error " << status << std::endl;                         \
      exit(1);                                                                 \
    }                                                                          \
  }

} // namespace

struct ParamDesc {
  const void *data;
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

struct Queue {
  Queue() {
    sycl::context context;
    sycl::property_list propList{sycl::property::queue::in_order()};
    sycl_queue = sycl::queue(context, sycl::gpu_selector(), propList);
  }

  sycl::queue sycl_queue;

  ze_module_handle_t loadModule(const void *data, size_t dataSize) {
    assert(data);
    ze_module_handle_t ze_module;
    ze_module_desc_t desc = {};
    desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    desc.pInputModule = static_cast<const uint8_t *>(data);
    desc.inputSize = dataSize;
    auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        this->sycl_queue.get_device());
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        this->sycl_queue.get_context());
    L0_SAFE_CALL(zeModuleCreate(ze_ctx, ze_device, &desc, &ze_module, nullptr));
    return ze_module;
  }

  void getKernel(ze_module_handle_t module, const char *name,
                 sycl::kernel &kernel) {
    assert(module);
    assert(name);
    ze_kernel_handle_t ze_kernel;
    ze_kernel_desc_t desc = {};
    desc.pKernelName = name;
    L0_SAFE_CALL(zeKernelCreate(module, &desc, &ze_kernel));

    sycl::kernel_bundle<sycl::bundle_state::executable> kernel_bundle =
        sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                 sycl::bundle_state::executable>(
            {module}, this->sycl_queue.get_context());

    kernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
        {kernel_bundle, ze_kernel}, this->sycl_queue.get_context());
  }

  void launchKernel(sycl::kernel *kernel, size_t gridX, size_t gridY,
                    size_t gridZ, size_t blockX, size_t blockY, size_t blockZ,
                    size_t sharedMemBytes, ParamDesc *params, void *extra) {

    auto queue = this->sycl_queue;
    auto sycl_global_range =
        ::sycl::range<3>(blockZ * gridZ, blockY * gridY, blockX * gridX);
    auto sycl_local_range = ::sycl::range<3>(blockZ, blockY, blockX);
    sycl::nd_range<3> sycl_nd_range(
        sycl::nd_range<3>(sycl_global_range, sycl_local_range));

    auto paramsCount = countUntil(params, ParamDesc{nullptr, 0});

    queue.submit([&](::sycl::handler &cgh) {
      for (uint32_t i = 0; i < paramsCount; i++) {
        cgh.set_arg(i, params[i]);
      }
      cgh.parallel_for(sycl_nd_range, *kernel);
    });
  }

  static auto destroyModule(ze_module_handle_t module) {
    assert(module);
    L0_SAFE_CALL(zeModuleDestroy(module));
  }

private:
};

/*
extern "C" DPCOMP_GPU_RUNTIME_EXPORT sycl::device dpcompGpuGetDevice() {
  return catchAll([&]() {
    sycl::device device;
    auto platform_list = sycl::platform::get_platforms();
    for (const auto &platform : platform_list) {
      auto platform_name = platform.get_info<sycl::info::platform::name>();
      bool is_level_zero =
          platform_name.find("Level-Zero") != std::string::npos;
      if (!is_level_zero)
        continue;
      device = platform.get_devices()[0];
    }
    return device;
  });
}

 extern "C" DPCOMP_GPU_RUNTIME_EXPORT sycl::context
 iGpuCreateContext(sycl::device device) {
 return catchAll([&]() {
   auto sycl_context = sycl::context(device);
   return sycl_context;
 });
}
*/

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *iGpuCreateStream(void *queue) {
  return catchAll([&]() {
    if (!static_cast<Queue *>(queue))
      return new Queue();
    else
      return static_cast<Queue *>(queue);
  });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void iGpuStreamDestroy(void *queue) {
  catchAll([&]() { delete static_cast<Queue *>(queue); });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *
iGpuMemAlloc(size_t size, void *queue, int shared) {
  catchAll([&]() {
    void *mem_ptr;
    mem_ptr =
        sycl::malloc_device(size, static_cast<Queue *>(queue)->sycl_queue);
    return reinterpret_cast<void *>(mem_ptr);
  });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *
iGpuModuleLoad(void *queue, const void *data, size_t dataSize) {
  return catchAll([&]() {
    return static_cast<Queue *>(queue)->loadModule(data, dataSize);
  });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void iGpuModuleDestroy(void *module) {
  catchAll(
      [&]() { Queue::destroyModule(static_cast<ze_module_handle_t>(module)); });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *
iGpuKernelGet(void *queue, void *module, const char *name) {
  return catchAll([&]() {
    sycl::kernel *kernel;
    static_cast<Queue *>(queue)->getKernel(
        static_cast<ze_module_handle_t>(module), name, *kernel);
    return kernel;
  });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void iGpuKernelDestroy(void *kernel) {
  catchAll([&]() {});
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void
iGpuLaunchKernel(void *queue, void *kernel, size_t gridX, size_t gridY,
                 size_t gridZ, size_t blockX, size_t blockY, size_t blockZ,
                 size_t sharedMemBytes, void *params, void *extra) {
  return catchAll([&]() {
    static_cast<Queue *>(queue)->launchKernel(
        static_cast<sycl::kernel *>(kernel), gridX, gridY, gridZ, blockX,
        blockY, blockZ, sharedMemBytes, static_cast<ParamDesc *>(params),
        extra);
  });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void iGpuStreamSynchronize(void *queue) {

  catchAll([&]() { static_cast<Queue *>(queue)->sycl_queue.wait(); });
}

/*
extern "C" DPCOMP_GPU_RUNTIME_EXPORT sycl::event *iGpuEventCreate() {

  catchAll([&]() { return sycl::event event(); });
}


extern "C" DPCOMP_GPU_RUNTIME_EXPORT void iGpuEventRecord(void *event,
                                                          void *queue) {

  catchAll([&]() {
    sycl::event event =
        static_cast<Queue *>(queue)->sycl_queue.submit_barrier();
  });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void iGpuStreamWaitEvent(void *queue,
                                                              void *event) {

  catchAll([&]() {
    static_cast<Queue *>(queue)->sycl_queue.submit_barrier(
        {reinterpret_cast<sycl::event *>(event)});
  });
}
*/

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void iGpuDeAlloc(void *queue, void *ptr) {
  catchAll([&]() {});
}