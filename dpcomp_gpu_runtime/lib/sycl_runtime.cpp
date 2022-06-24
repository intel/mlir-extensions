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
#include <iostream>

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
    if (status != ZE_RESULT_SUCCESS) {                                                         \
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

struct Queue {
  Queue() { 
  std::cout<<"QUEUE CTOR "<<std::endl;  
  sycl::device device;
  auto platform_list = sycl::platform::get_platforms();
  for (const auto& platform : platform_list) {
    auto platform_name = platform.get_info<sycl::info::platform::name>();
    bool is_level_zero = platform_name.find("Level-Zero") != std::string::npos;
    if (!is_level_zero) continue;
    device = platform.get_devices()[0];
    std::cout<<"Device Found"<<std::endl;
  }
  sycl::context context(device);  
  sycl::property_list propList{sycl::property::queue::in_order()};
  sycl_queue = sycl::queue(context, device, propList);
  }

  sycl::queue sycl_queue;

  void *alloc_device_memory(size_t size, size_t alignment) {
    void *mem_ptr = nullptr;
    int shared = 1;
    if(shared){
      mem_ptr = sycl::aligned_alloc_shared(alignment, size, this->sycl_queue);
    }
    else{
      mem_ptr = sycl::malloc_device(size, this->sycl_queue);
    }
    std::cout<<"RESULT PTR IS "<<mem_ptr<<std::endl;
    return mem_ptr;
  }

  ze_module_handle_t loadModule(const void *data, size_t dataSize) {
    assert(data);
    ze_module_handle_t ze_module;
    ze_module_desc_t desc =  {ZE_STRUCTURE_TYPE_MODULE_DESC,
                                 nullptr,
                                 ZE_MODULE_FORMAT_IL_SPIRV,
                                 dataSize,
                                 (const uint8_t*)data,
                                 nullptr,
                                 nullptr};
    auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        this->sycl_queue.get_device());
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        this->sycl_queue.get_context());
    L0_SAFE_CALL(zeModuleCreate(ze_ctx, ze_device, &desc, &ze_module, nullptr));
    return ze_module;
  }

  void *getKernel(ze_module_handle_t module, const char *name) {
    std::cout<<"IN GET KERNEL "<<std::endl;	  
    assert(module);
    assert(name);
    ze_kernel_handle_t ze_kernel;
    sycl::kernel *sycl_kernel;
    ze_kernel_desc_t desc = {};
    desc.pKernelName = name;
    L0_SAFE_CALL(zeKernelCreate(module, &desc, &ze_kernel));

    sycl::kernel_bundle<sycl::bundle_state::executable> kernel_bundle =
        sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                 sycl::bundle_state::executable>(
            {module}, this->sycl_queue.get_context());

   auto kernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
        {kernel_bundle, ze_kernel}, this->sycl_queue.get_context());
    sycl_kernel = new sycl::kernel(kernel);
    return sycl_kernel;
  }

  void launchKernel(sycl::kernel *kernel, size_t gridX, size_t gridY,
                    size_t gridZ, size_t blockX, size_t blockY, size_t blockZ,
                    size_t sharedMemBytes, ParamDesc *params, void *extra) {

    std::cout<<"IN LAUNCH KERNEL "<<std::endl;	   
    auto queue = this->sycl_queue;
    auto sycl_global_range =
        ::sycl::range<3>(blockZ * gridZ, blockY * gridY, blockX * gridX);
    auto sycl_local_range = ::sycl::range<3>(blockZ, blockY, blockX);
    sycl::nd_range<3> sycl_nd_range(
        sycl::nd_range<3>(sycl_global_range, sycl_local_range));

    auto paramsCount = countUntil(params, ParamDesc{nullptr, 0});

    std::cout<<"PARAMS COUNT "<<paramsCount<<std::endl;
    queue.submit([&](sycl::handler &cgh) {
      for (size_t i = 0; i < paramsCount; i++) {
        auto param = params[i]; 
	std::cout<<"PARAM DATA ADDRESS IS "<<*(static_cast<void**>(param.data))<<std::endl;
	cgh.set_arg(static_cast<uint32_t>(i), *(static_cast<void**>(param.data)));
      }
      cgh.parallel_for(sycl_nd_range, *kernel);
    });
    std::cout<<"OUT LAUNCH KERNEL "<<std::endl;
  }

  static auto destroyModule(ze_module_handle_t module) {
    assert(module);
    L0_SAFE_CALL(zeModuleDestroy(module));
  }

private:
};

// Wrappers
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
iGpuMemAlloc(size_t size, size_t alignment, void *queue) {
    return static_cast<Queue *>(queue)->alloc_device_memory(size, alignment); 
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *
iGpuModuleLoad(void *queue, const void *data, size_t dataSize) {
    return static_cast<Queue *>(queue)->loadModule(data, dataSize);	
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void iGpuModuleDestroy(void *module) {
  catchAll(
      [&]() { Queue::destroyModule(static_cast<ze_module_handle_t>(module)); });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *
iGpuKernelGet(void *queue, void *module, const char *name) {
     return static_cast<Queue *>(queue)->getKernel(static_cast<ze_module_handle_t>(module), name);	
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void iGpuKernelDestroy(void *kernel) {
  catchAll([&]() {});
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void
iGpuLaunchKernel(void *queue, void *kernel, size_t gridX, size_t gridY,
                 size_t gridZ, size_t blockX, size_t blockY, size_t blockZ,
                 size_t sharedMemBytes, void *params, void *extra) {
    static_cast<Queue *>(queue)->launchKernel(
        static_cast<sycl::kernel *>(kernel), gridX, gridY, gridZ, blockX,
        blockY, blockZ, sharedMemBytes, static_cast<ParamDesc *>(params),
        extra);
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
