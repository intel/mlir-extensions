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

#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <level_zero/ze_api.h>
#include <map>
#include <mutex>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/queue.hpp> // for queue
#include <sycl/sycl.hpp>

#ifdef _WIN32
#define SYCL_RUNTIME_EXPORT __declspec(dllexport)
#else
#define SYCL_RUNTIME_EXPORT
#endif // _WIN32

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
      fprintf(stdout, "L0 error %d\n", status);                                \
      fflush(stdout);                                                          \
      abort();                                                                 \
    }                                                                          \
  }

} // namespace

struct SpirvModule {
  ze_module_handle_t module = nullptr;
  ~SpirvModule();
};

namespace {
// Create a Map for the spirv module lookup
std::map<void *, SpirvModule> moduleCache;
std::mutex mutexLock;
} // namespace

SpirvModule::~SpirvModule() {
  L0_SAFE_CALL(zeModuleDestroy(SpirvModule::module));
}

struct ParamDesc {
  void *data;
  size_t size;

  bool operator==(const ParamDesc &rhs) const {
    return data == rhs.data && size == rhs.size;
  }

  bool operator!=(const ParamDesc &rhs) const { return !(*this == rhs); }
};

struct EventDesc {
  void *event;

  bool operator==(const EventDesc &rhs) const { return event == rhs.event; }

  bool operator!=(const EventDesc &rhs) const { return !(*this == rhs); }
};

template <typename T> size_t countUntil(T *ptr, T &&elem) {
  assert(ptr);
  auto curr = ptr;
  while (*curr != elem) {
    ++curr;
  }
  return static_cast<size_t>(curr - ptr);
}

static sycl::device getDefaultDevice() {
  auto platformList = sycl::platform::get_platforms();
  for (const auto &platform : platformList) {
    auto platformName = platform.get_info<sycl::info::platform::name>();
    bool isLevelZero = platformName.find("Level-Zero") != std::string::npos;
    if (!isLevelZero)
      continue;

    return platform.get_devices()[0];
  }
}

struct GPUSYCLQUEUE {

  sycl::device syclDevice_;
  sycl::context syclContext_;
  sycl::queue syclQueue_;

  GPUSYCLQUEUE(sycl::property_list propList) {

    syclDevice_ = getDefaultDevice();
    syclContext_ = sycl::context(syclDevice_);
    syclQueue_ = sycl::queue(syclContext_, syclDevice_, propList);
  }

  GPUSYCLQUEUE(sycl::device *device, sycl::context *context,
               sycl::property_list propList) {
    syclDevice_ = *device;
    syclContext_ = *context;
    syclQueue_ = sycl::queue(syclContext_, syclDevice_, propList);
  }
  GPUSYCLQUEUE(sycl::device *device, sycl::property_list propList) {

    syclDevice_ = *device;
    syclContext_ = sycl::context(syclDevice_);
    syclQueue_ = sycl::queue(syclContext_, syclDevice_, propList);
  }

  GPUSYCLQUEUE(sycl::context *context, sycl::property_list propList) {

    syclDevice_ = getDefaultDevice();
    syclContext_ = *context;
    syclQueue_ = sycl::queue(syclContext_, syclDevice_, propList);
  }

}; // end of GPUSYCLQUEUE

#if 0
static std::string getDeviceID(GPUSYCLQUEUE *queue) {
  auto syclDevice = queue->syclDevice_;

  int deviceID =
      syclDevice.get_info<sycl::ext::intel::info::device::device_id>();
  std::string deviceName = "";
  switch (deviceID) {
  case 0x0BDA:          // 1100
  case 0x0BD5:          // 1550
  case 0x0BD0:          // Future
  case 0x0BD6:          // Future
  case 0x0BD7:          // Future
  case 0x0BDB:          // Future
  case 0x0BD9:          // Future
    deviceName = "pvc"; // Intel® Data Center GPU Max Series, XeHPC
    break;
  case 0x56C0:
  case 0x56C1:
    deviceName = "acm"; // Intel® Data Center GPU Flex Seriex, XeHPG
  default:
    deviceName = "pvc"; // TODO: Throw an error for unsupported platform
  }
  return deviceName;
}
#endif

static void *allocDeviceMemory(GPUSYCLQUEUE *queue, size_t size,
                               size_t alignment, bool isShared) {
  void *memPtr = nullptr;
  if (isShared) {
    memPtr = sycl::aligned_alloc_shared(alignment, size, queue->syclQueue_);
  } else {
    memPtr = sycl::aligned_alloc_device(alignment, size, queue->syclQueue_);
  }
  if (memPtr == nullptr) {
    throw std::runtime_error(
        "aligned_alloc_shared() failed to allocate memory!");
  }
  return memPtr;
}

static void deallocDeviceMemory(GPUSYCLQUEUE *queue, void *ptr) {
  sycl::free(ptr, queue->syclQueue_);
}

// @TODO: Add support for async copy
// Currnetly, we only support synchrnous copy?
static void memoryCopy(GPUSYCLQUEUE *queue, void *dstPtr, void *srcPtr,
                       size_t size) {
  queue->syclQueue_.memcpy(dstPtr, srcPtr, size).wait();
}

static ze_module_handle_t loadModule(GPUSYCLQUEUE *queue, const void *data,
                                     size_t dataSize) {
  assert(data);
  auto syclQueue = queue->syclQueue_;
  ze_module_handle_t zeModule;

  // TODO: Enable this for current Device
  // query and throw an error for unsupported platforms
  // getDeviceID(syclQueue);

  auto it = moduleCache.find((void *)data);
  // Check the map if the module is present/cached.
  if (it != moduleCache.end()) {
    return it->second.module;
  }

  ze_module_desc_t desc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                           nullptr,
                           ZE_MODULE_FORMAT_IL_SPIRV,
                           dataSize,
                           (const uint8_t *)data,
                           nullptr,
                           nullptr};

  std::string build_flags;
  // IGC auto-detection of scalar/vector backend does not work for native BF16
  // data type yet, hence we need to pass this flag explicitly for if native
  // bf16 data type is used and we need to use vector compute.
  if (getenv("IMEX_USE_IGC_VECTOR_BACK_END")) {
    build_flags += " -vc-codegen ";
  }
  // enable large register file if needed
  if (getenv("IMEX_ENABLE_LARGE_REG_FILE")) {
    build_flags += "-doubleGRF -Xfinalizer -noLocalSplit -Xfinalizer "
                   "-DPASTokenReduction -Xfinalizer -SWSBDepReduction "
                   "-Xfinalizer -printregusage -Xfinalizer -enableBCR";
  }
  desc.pBuildFlags = build_flags.c_str();
  auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      syclQueue.get_device());
  auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      syclQueue.get_context());
  L0_SAFE_CALL(zeModuleCreate(zeContext, zeDevice, &desc, &zeModule, nullptr));
  std::lock_guard<std::mutex> entryLock(mutexLock);
  moduleCache[(void *)data].module = zeModule;
  return zeModule;
}

static sycl::kernel *getKernel(GPUSYCLQUEUE *queue, ze_module_handle_t zeModule,
                               const char *name) {
  assert(zeModule);
  assert(name);
  auto syclQueue = queue->syclQueue_;
  ze_kernel_handle_t zeKernel;
  sycl::kernel *syclKernel;
  ze_kernel_desc_t desc = {};
  desc.pKernelName = name;

  L0_SAFE_CALL(zeKernelCreate(zeModule, &desc, &zeKernel));
  sycl::kernel_bundle<sycl::bundle_state::executable> kernelBundle =
      sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                               sycl::bundle_state::executable>(
          {zeModule}, syclQueue.get_context());

  auto kernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {kernelBundle, zeKernel}, syclQueue.get_context());
  syclKernel = new sycl::kernel(kernel);
  return syclKernel;
}

static sycl::event enqueueKernel(sycl::queue queue, sycl::kernel *kernel,
                                 sycl::nd_range<3> NdRange, ParamDesc *params,
                                 size_t sharedMemBytes, EventDesc *depEvents) {
  auto depEventsCount = countUntil(depEvents, EventDesc{nullptr});
  auto paramsCount = countUntil(params, ParamDesc{nullptr, 0});
  // The assumption is, if there is a param for the shared local memory,
  // then that will always be the last argument.
  if (sharedMemBytes) {
    paramsCount = paramsCount - 1;
  }
  sycl::event event = queue.submit([&](sycl::handler &cgh) {
    for (size_t i = 0; i < depEventsCount; i++) {
      // Depend on any event that was specified by the caller.
      cgh.depends_on(*(static_cast<sycl::event *>(depEvents[i].event)));
    }
    for (size_t i = 0; i < paramsCount; i++) {
      auto param = params[i];
      cgh.set_arg(static_cast<uint32_t>(i),
                  *(static_cast<void **>(param.data)));
    }
    if (sharedMemBytes) {
      // TODO: Handle other data types
      using share_mem_t =
          sycl::accessor<float, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>;
      share_mem_t local_buffer =
          share_mem_t(sharedMemBytes / sizeof(float), cgh);
      cgh.set_arg(paramsCount, local_buffer);
      cgh.parallel_for(NdRange, *kernel);
    } else {
      cgh.parallel_for(NdRange, *kernel);
    }
  });

  return event;
}

static sycl::event *launchKernel(GPUSYCLQUEUE *queue, sycl::kernel *kernel,
                                 size_t gridX, size_t gridY, size_t gridZ,
                                 size_t blockX, size_t blockY, size_t blockZ,
                                 size_t sharedMemBytes, ParamDesc *params,
                                 EventDesc *depEvents) {
  auto syclQueue = queue->syclQueue_;
  auto syclGlobalRange =
      ::sycl::range<3>(blockZ * gridZ, blockY * gridY, blockX * gridX);
  auto syclLocalRange = ::sycl::range<3>(blockZ, blockY, blockX);
  sycl::nd_range<3> syclNdRange(
      sycl::nd_range<3>(syclGlobalRange, syclLocalRange));

  if (getenv("IMEX_ENABLE_PROFILING")) {
    auto executionTime = 0.0f;
    auto maxTime = 0.0f;
    auto minTime = FLT_MAX;
    auto rounds = 100;
    auto warmups = 3;

    // Before each run we need to flush the L3 cache (global memory cache) to
    // make sure each profiling run has the same cache state. This is done by
    // writing 'zero' to a global device memory buffer larger than the L3 cache
    // size. This way, any data in the cache from previous run is evicted and we
    // get a cold cache behavior. The write to the buffer is 'cached' in other
    // words, the write goes through the L3 (global memory cache).
    size_t cacheSize =
        queue->syclDevice_
            .get_info<sycl::info::device::global_mem_cache_size>();

    // Allocate a device memory buffer twice the size of the
    // L3 cache (global memory cache) of the device.
    // The buffer can be host_shared or device-only, for our use-case we choose
    // it to be device-only, it removes the need for extra copy from/to host.
    // More importantly, it removes the possiblity of accidentally doing the
    // flush in the host-side using a host side function.
    auto *cache = allocDeviceMemory(queue, 2 * cacheSize, 64, false);

    if (getenv("IMEX_PROFILING_RUNS")) {
      auto runs = strtol(getenv("IMEX_PROFILING_RUNS"), NULL, 10L);
      if (runs)
        rounds = runs;
    }

    if (getenv("IMEX_PROFILING_WARMUPS")) {
      auto runs = strtol(getenv("IMEX_PROFILING_WARMUPS"), NULL, 10L);
      if (warmups)
        warmups = runs;
    }

    // warmups
    for (int r = 0; r < warmups; r++) {
      auto e = enqueueKernel(syclQueue, kernel, syclNdRange, params,
                             sharedMemBytes, depEvents);
      e.wait();
    }

    for (int r = 0; r < rounds; r++) {
      // Flush the L3 cache (global memory cache).
      if (getenv("IMEX_ENABLE_CACHE_FLUSHING")) {
        int init_val = 0;
        syclQueue.memset(cache, init_val, cacheSize);
      }
      sycl::event event = enqueueKernel(syclQueue, kernel, syclNdRange, params,
                                        sharedMemBytes, depEvents);
      event.wait();

      auto startTime =
          event
              .get_profiling_info<sycl::info::event_profiling::command_start>();
      auto endTime =
          event.get_profiling_info<sycl::info::event_profiling::command_end>();
      auto gap = float(endTime - startTime) / 1000000.0f;
      executionTime += gap;
      if (gap > maxTime)
        maxTime = gap;
      if (gap < minTime)
        minTime = gap;
    }

    deallocDeviceMemory(queue, cache);
    fprintf(stdout,
            "the kernel execution time is (ms):"
            "avg: %.4f, min: %.4f, max: %.4f (over %d runs)\n",
            executionTime / rounds, minTime, maxTime, rounds);
  }

  auto event = enqueueKernel(syclQueue, kernel, syclNdRange, params,
                             sharedMemBytes, depEvents);

  sycl::event *syclEvent = new sycl::event(event);

  return syclEvent;
}

// Wrappers

extern "C" SYCL_RUNTIME_EXPORT GPUSYCLQUEUE *gpuCreateStream(void *device,
                                                             void *context) {
  auto propList = sycl::property_list{};
  if (getenv("IMEX_ENABLE_PROFILING")) {
    propList = sycl::property_list{sycl::property::queue::enable_profiling()};
  }
  return catchAll([&]() {
    if (!device && !context) {
      return new GPUSYCLQUEUE(propList);
    } else if (device && context) {
      // TODO: Check if the pointers/address is valid and holds the correct
      // device and context
      return new GPUSYCLQUEUE(static_cast<sycl::device *>(device),
                              static_cast<sycl::context *>(context), propList);
    } else if (device && !context) {
      return new GPUSYCLQUEUE(static_cast<sycl::device *>(device), propList);
    } else {
      return new GPUSYCLQUEUE(static_cast<sycl::context *>(context), propList);
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT void gpuStreamDestroy(GPUSYCLQUEUE *queue) {
  catchAll([&]() { delete queue; });
}

extern "C" SYCL_RUNTIME_EXPORT void *
gpuMemAlloc(GPUSYCLQUEUE *queue, size_t size, size_t alignment, bool isShared) {
  return catchAll([&]() {
    if (queue) {
      return allocDeviceMemory(queue, size, alignment, isShared);
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT void gpuMemFree(GPUSYCLQUEUE *queue, void *ptr) {
  catchAll([&]() {
    if (queue && ptr) {
      deallocDeviceMemory(queue, ptr);
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT void
gpuMemCopy(GPUSYCLQUEUE *queue, void *dstPtr, void *srcPtr, size_t size) {
  return catchAll([&]() { memoryCopy(queue, dstPtr, srcPtr, size); });
}

extern "C" SYCL_RUNTIME_EXPORT ze_module_handle_t
gpuModuleLoad(GPUSYCLQUEUE *queue, const void *data, size_t dataSize) {
  return catchAll([&]() {
    if (queue) {
      return loadModule(queue, data, dataSize);
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT sycl::kernel *
gpuKernelGet(GPUSYCLQUEUE *queue, ze_module_handle_t module, const char *name) {
  return catchAll([&]() {
    if (queue) {
      return getKernel(queue, module, name);
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT sycl::event *
gpuLaunchKernel(GPUSYCLQUEUE *queue, sycl::kernel *kernel, size_t gridX,
                size_t gridY, size_t gridZ, size_t blockX, size_t blockY,
                size_t blockZ, size_t sharedMemBytes, void *params,
                void *depEvents) {
  return catchAll([&]() {
    if (queue) {
      return launchKernel(queue, kernel, gridX, gridY, gridZ, blockX, blockY,
                          blockZ, sharedMemBytes,
                          static_cast<ParamDesc *>(params),
                          static_cast<EventDesc *>(depEvents));
    }
  });
}

extern "C" SYCL_RUNTIME_EXPORT void gpuWait(GPUSYCLQUEUE *queue) {

  catchAll([&]() {
    if (queue) {
      queue->syclQueue_.wait();
    }
  });
}
