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
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <level_zero/ze_api.h>

#ifdef _WIN32
#define LEVEL_ZERO_RUNTIME_EXPORT __declspec(dllexport)
#else
#define LEVEL_ZERO_RUNTIME_EXPORT
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

inline void checkResult(ze_result_t res, const char *func) {
  if (res != ZE_RESULT_SUCCESS)
    throw std::runtime_error(std::string(func) +
                             " failed: " + std::to_string(res));
}

#define CHECK_ZE_RESULT(expr) checkResult((expr), #expr)

} // namespace

struct SpirvModule {
  ze_module_handle_t module = nullptr;
  ~SpirvModule();
};

namespace {
// Create a Map for the spirv module lookup
std::map<const void *, SpirvModule> moduleCache;
std::mutex mutexLock;
} // namespace

SpirvModule::~SpirvModule() {
  CHECK_ZE_RESULT(zeModuleDestroy(SpirvModule::module));
}

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

static std::pair<ze_driver_handle_t, ze_device_handle_t>
getDriverAndDevice(ze_device_type_t deviceType = ZE_DEVICE_TYPE_GPU) {

  CHECK_ZE_RESULT(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t driverCount = 0;
  CHECK_ZE_RESULT(zeDriverGet(&driverCount, nullptr));

  std::vector<ze_driver_handle_t> allDrivers{driverCount};
  CHECK_ZE_RESULT(zeDriverGet(&driverCount, allDrivers.data()));

  // Find a driver instance with a GPU device
  std::vector<ze_device_handle_t> devices;
  for (uint32_t i = 0; i < driverCount; ++i) {
    uint32_t deviceCount = 0;
    CHECK_ZE_RESULT(zeDeviceGet(allDrivers[i], &deviceCount, nullptr));
    if (deviceCount == 0)
      continue;
    devices.resize(deviceCount);
    CHECK_ZE_RESULT(zeDeviceGet(allDrivers[i], &deviceCount, devices.data()));
    for (uint32_t d = 0; d < deviceCount; ++d) {
      ze_device_properties_t device_properties = {};
      CHECK_ZE_RESULT(zeDeviceGetProperties(devices[d], &device_properties));
      if (deviceType == device_properties.type) {
        auto driver = allDrivers[i];
        auto device = devices[d];
        return {driver, device};
      }
    }
  }
  throw std::runtime_error("getDevice failed");
}

#define _IMEX_PROFILING_TRAITS_SPEC(Desc)                                      \
  struct Desc {};

namespace imex {
namespace profiling {
// defining two types representing kernel start and kernel end
_IMEX_PROFILING_TRAITS_SPEC(command_start)
_IMEX_PROFILING_TRAITS_SPEC(command_end)
} // namespace profiling
} // namespace imex

// A Timestamp event pool management class. It currently simply represents
// a event pool with fixed 256 slots. Currently for each run we just need
// one timing event, but we definity need a sophisticated event system in
// the future for programs with multiple kernels.
struct EventPool {
  ze_event_pool_handle_t zeEventPool;

  EventPool(ze_context_handle_t zeContext_) {
    ze_event_pool_desc_t tsEventPoolDesc = {
        ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
        ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP, 256};
    CHECK_ZE_RESULT(zeEventPoolCreate(zeContext_, &tsEventPoolDesc, 0, nullptr,
                                      &zeEventPool));
  }

  ~EventPool() { CHECK_ZE_RESULT(zeEventPoolDestroy(zeEventPool)); }
};

// A wrapper to ze_event_handle_t providing timestamp queries
class Event {
private:
  uint64_t zeTimestampMaxValue_;
  uint64_t zeTimerResolution_;

public:
  ze_event_handle_t zeEvent;

  Event(ze_context_handle_t zeContext_, ze_device_handle_t zeDevice_) {
    static EventPool pool(zeContext_);

    // timestamp and timer resolution is a device properties.
    // They are required to compute the final wall time.
    ze_device_properties_t deviceProperties{};
    deviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    CHECK_ZE_RESULT(zeDeviceGetProperties(zeDevice_, &deviceProperties));
    zeTimestampMaxValue_ =
        ((1ULL << deviceProperties.kernelTimestampValidBits) - 1ULL);
    zeTimerResolution_ = deviceProperties.timerResolution;

    ze_event_desc_t eventDesc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr,
        0, // index
        0, // no additional memory/cache coherency required on signal
        0  // no additional memory/cache coherency required on wait
    };
    CHECK_ZE_RESULT(zeEventCreate(pool.zeEventPool, &eventDesc, &zeEvent));
  }

  // query the kernel start or end (specified via Param) timestamp
  template <typename Param> uint64_t get_profiling_info() {
    ze_kernel_timestamp_result_t tsResult;
    CHECK_ZE_RESULT(zeEventQueryKernelTimestamp(zeEvent, &tsResult));

    if constexpr (std::is_same_v<Param, imex::profiling::command_start>) {
      uint64_t startTime =
          (tsResult.global.kernelStart & zeTimestampMaxValue_) *
          zeTimerResolution_;
      return startTime;
    }

    if constexpr (std::is_same_v<Param, imex::profiling::command_end>) {
      uint64_t startTime = tsResult.global.kernelStart & zeTimestampMaxValue_;
      uint64_t endTime = tsResult.global.kernelEnd & zeTimestampMaxValue_;

      if (endTime < startTime)
        endTime += zeTimestampMaxValue_;

      endTime *= zeTimerResolution_;
      return endTime;
    }
  }

  ~Event() { CHECK_ZE_RESULT(zeEventDestroy(zeEvent)); }
};

struct GPUL0QUEUE {

  ze_driver_handle_t zeDriver_ = nullptr;
  ze_device_handle_t zeDevice_ = nullptr;
  ze_context_handle_t zeContext_ = nullptr;
  ze_command_list_handle_t zeCommandList_ = nullptr;

  GPUL0QUEUE() {
    auto driverAndDevice = getDriverAndDevice();
    zeDriver_ = driverAndDevice.first;
    zeDevice_ = driverAndDevice.second;

    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr,
                                     0};
    CHECK_ZE_RESULT(zeContextCreate(zeDriver_, &contextDesc, &zeContext_));

    uint32_t numQueueGroups = 0;
    CHECK_ZE_RESULT(zeDeviceGetCommandQueueGroupProperties(
        zeDevice_, &numQueueGroups, nullptr));

    std::vector<ze_command_queue_group_properties_t> queueProperties(
        numQueueGroups);
    CHECK_ZE_RESULT(zeDeviceGetCommandQueueGroupProperties(
        zeDevice_, &numQueueGroups, queueProperties.data()));

    ze_command_queue_desc_t desc = {};
    desc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
    for (uint32_t i = 0; i < numQueueGroups; i++) {
      if (queueProperties[i].flags &
          ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
        desc.ordinal = i;
      }
    }
    CHECK_ZE_RESULT(zeCommandListCreateImmediate(zeContext_, zeDevice_, &desc,
                                                 &zeCommandList_));
  }

  GPUL0QUEUE(ze_device_type_t *deviceType, ze_context_handle_t context) {
    auto driverAndDevice = getDriverAndDevice(*deviceType);
    zeDriver_ = driverAndDevice.first;
    zeDevice_ = driverAndDevice.second;

    zeContext_ = context;

    uint32_t numQueueGroups = 0;
    CHECK_ZE_RESULT(zeDeviceGetCommandQueueGroupProperties(
        zeDevice_, &numQueueGroups, nullptr));

    std::vector<ze_command_queue_group_properties_t> queueProperties(
        numQueueGroups);
    CHECK_ZE_RESULT(zeDeviceGetCommandQueueGroupProperties(
        zeDevice_, &numQueueGroups, queueProperties.data()));

    ze_command_queue_desc_t desc = {};
    desc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
    for (uint32_t i = 0; i < numQueueGroups; i++) {
      if (queueProperties[i].flags &
          ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
        desc.ordinal = i;
      }
    }
    CHECK_ZE_RESULT(zeCommandListCreateImmediate(zeContext_, zeDevice_, &desc,
                                                 &zeCommandList_));
  }

  GPUL0QUEUE(ze_device_type_t *deviceType) {

    auto driverAndDevice = getDriverAndDevice(*deviceType);
    zeDriver_ = driverAndDevice.first;
    zeDevice_ = driverAndDevice.second;

    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr,
                                     0};
    CHECK_ZE_RESULT(zeContextCreate(zeDriver_, &contextDesc, &zeContext_));

    uint32_t numQueueGroups = 0;
    CHECK_ZE_RESULT(zeDeviceGetCommandQueueGroupProperties(
        zeDevice_, &numQueueGroups, nullptr));

    std::vector<ze_command_queue_group_properties_t> queueProperties(
        numQueueGroups);
    CHECK_ZE_RESULT(zeDeviceGetCommandQueueGroupProperties(
        zeDevice_, &numQueueGroups, queueProperties.data()));

    ze_command_queue_desc_t desc = {};
    desc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
    for (uint32_t i = 0; i < numQueueGroups; i++) {
      if (queueProperties[i].flags &
          ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
        desc.ordinal = i;
      }
    }
    CHECK_ZE_RESULT(zeCommandListCreateImmediate(zeContext_, zeDevice_, &desc,
                                                 &zeCommandList_));
  }

  GPUL0QUEUE(ze_context_handle_t context) {

    auto driverAndDevice = getDriverAndDevice();
    zeDriver_ = driverAndDevice.first;
    zeDevice_ = driverAndDevice.second;
    zeContext_ = context;

    uint32_t numQueueGroups = 0;
    CHECK_ZE_RESULT(zeDeviceGetCommandQueueGroupProperties(
        zeDevice_, &numQueueGroups, nullptr));

    std::vector<ze_command_queue_group_properties_t> queueProperties(
        numQueueGroups);
    CHECK_ZE_RESULT(zeDeviceGetCommandQueueGroupProperties(
        zeDevice_, &numQueueGroups, queueProperties.data()));

    ze_command_queue_desc_t desc = {};
    desc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
    for (uint32_t i = 0; i < numQueueGroups; i++) {
      if (queueProperties[i].flags &
          ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
        desc.ordinal = i;
      }
    }
    CHECK_ZE_RESULT(zeCommandListCreateImmediate(zeContext_, zeDevice_, &desc,
                                                 &zeCommandList_));
  }

  ~GPUL0QUEUE() {
    // Device and Driver resource management is dony by L0.
    // Just release context and commandList.
    // TODO: Use unique ptrs.
    if (zeContext_)
      CHECK_ZE_RESULT(zeContextDestroy(zeContext_));

    if (zeCommandList_)
      CHECK_ZE_RESULT(zeCommandListDestroy(zeCommandList_));
  }
};

static void *allocDeviceMemory(GPUL0QUEUE *queue, size_t size, size_t alignment,
                               bool isShared) {

  void *ret = nullptr;
  ze_device_mem_alloc_desc_t devDesc = {};
  devDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  if (isShared) {
    ze_host_mem_alloc_desc_t hostDesc = {};
    hostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    CHECK_ZE_RESULT(zeMemAllocShared(queue->zeContext_, &devDesc, &hostDesc,
                                     size, alignment, queue->zeDevice_, &ret));
  } else {
    devDesc.flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT;
    CHECK_ZE_RESULT(zeMemAllocDevice(queue->zeContext_, &devDesc, size,
                                     alignment, queue->zeDevice_, &ret));
  }
  return ret;
}

static void deallocDeviceMemory(GPUL0QUEUE *queue, void *ptr) {
  CHECK_ZE_RESULT(zeMemFree(queue->zeContext_, ptr));
}
// @TODO: Add support for async copy
// Currnetly, we only support synchrnous copy?
static void memoryCopy(GPUL0QUEUE *queue, void *dstPtr, void *srcPtr,
                       size_t size) {
  CHECK_ZE_RESULT(zeCommandListAppendMemoryCopy(
      queue->zeCommandList_, dstPtr, srcPtr, size, nullptr, 0, nullptr));
}
static ze_module_handle_t loadModule(GPUL0QUEUE *queue, const void *data,
                                     size_t dataSize) {
  assert(data);
  auto gpuL0Queue = queue;
  ze_module_handle_t zeModule;

  auto it = moduleCache.find((const void *)data);
  // Check the map if the module is present/cached.
  if (it != moduleCache.end()) {
    return it->second.module;
  }
  ze_module_desc_t desc = {};

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
    ;
  }

  desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  desc.pInputModule = static_cast<const uint8_t *>(data);
  desc.inputSize = dataSize;
  desc.pBuildFlags = build_flags.c_str();
  CHECK_ZE_RESULT(zeModuleCreate(gpuL0Queue->zeContext_, gpuL0Queue->zeDevice_,
                                 &desc, &zeModule, nullptr));
  std::lock_guard<std::mutex> entryLock(mutexLock);
  moduleCache[(const void *)data].module = zeModule;
  return zeModule;
}

static ze_kernel_handle_t
getKernel(GPUL0QUEUE *queue, ze_module_handle_t module, const char *name) {
  assert(module);
  assert(name);
  ze_kernel_desc_t desc = {};
  ze_kernel_handle_t zeKernel;
  desc.pKernelName = name;
  CHECK_ZE_RESULT(zeKernelCreate(module, &desc, &zeKernel));
  return zeKernel;
}

static void enqueueKernel(ze_command_list_handle_t zeCommandList,
                          ze_kernel_handle_t kernel,
                          const ze_group_count_t *pLaunchArgs,
                          ParamDesc *params, size_t sharedMemBytes,
                          ze_event_handle_t waitEvent = nullptr,
                          uint32_t numWaitEvents = 0,
                          ze_event_handle_t *phWaitEvents = nullptr) {
  auto paramsCount = countUntil(params, ParamDesc{nullptr, 0});

  if (sharedMemBytes) {
    paramsCount = paramsCount - 1;
  }

  for (size_t i = 0; i < paramsCount; ++i) {
    auto param = params[i];
    CHECK_ZE_RESULT(zeKernelSetArgumentValue(kernel, static_cast<uint32_t>(i),
                                             param.size, param.data));
  }

  if (sharedMemBytes) {
    CHECK_ZE_RESULT(
        zeKernelSetArgumentValue(kernel, paramsCount, sharedMemBytes, nullptr));
  }

  CHECK_ZE_RESULT(zeCommandListAppendLaunchKernel(zeCommandList, kernel,
                                                  pLaunchArgs, waitEvent,
                                                  numWaitEvents, phWaitEvents));
}

// Utility to discover the Global memory cache (L3) size of the device
static size_t getGlobalMemoryCacheSize(ze_device_handle_t zeDevice) {
  static constexpr unsigned MaxPropertyEntries = 16;
  uint32_t CachePropCount = MaxPropertyEntries;
  ze_device_cache_properties_t CacheProperties[MaxPropertyEntries];
  for (uint32_t i = 0; i < MaxPropertyEntries; ++i) {
    CacheProperties[i].stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
    CacheProperties[i].pNext = nullptr;
  }
  CHECK_ZE_RESULT(
      zeDeviceGetCacheProperties(zeDevice, &CachePropCount, CacheProperties));
  size_t globaMemoryCacheSize = 0;
  for (uint32_t i = 0; i < CachePropCount; ++i) {
    // find largest cache that is not user-controlled
    if ((CacheProperties[i].flags &
         ZE_DEVICE_CACHE_PROPERTY_FLAG_USER_CONTROL) != 0u) {
      continue;
    }
    if (globaMemoryCacheSize < CacheProperties[i].cacheSize) {
      globaMemoryCacheSize = CacheProperties[i].cacheSize;
    }
  }
  return globaMemoryCacheSize;
}

static void launchKernel(GPUL0QUEUE *queue, ze_kernel_handle_t kernel,
                         size_t gridX, size_t gridY, size_t gridZ,
                         size_t blockX, size_t blockY, size_t blockZ,
                         size_t sharedMemBytes, ParamDesc *params) {
  assert(kernel);

  auto castSz = [](size_t val) { return static_cast<uint32_t>(val); };

  CHECK_ZE_RESULT(zeKernelSetGroupSize(kernel, castSz(blockX), castSz(blockY),
                                       castSz(blockZ)));
  ze_group_count_t launchArgs = {castSz(gridX), castSz(gridY), castSz(gridZ)};

  if (getenv("IMEX_ENABLE_PROFILING")) {
    auto executionTime = 0.0f;
    auto maxTime = 0.0f;
    auto minTime = FLT_MAX;
    auto rounds = 1000;
    auto warmups = 3;

    // Before each run we need to flush the L3 cache (global memory cache) to
    // make sure each profiling run has the same cache state. This is done by
    // writing 'zero' to a global device memory buffer larger than the L3 cache
    // size. This way, any data in the cache from previous run is evicted and we
    // get a cold cache behavior. The write to the buffer is 'cached' in other
    // words, the write goes through the L3 (global memory cache).
    size_t cacheSize = getGlobalMemoryCacheSize(queue->zeDevice_);

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

    // warmup
    for (int r = 0; r < warmups; r++) {
      enqueueKernel(queue->zeCommandList_, kernel, &launchArgs, params,
                    sharedMemBytes, nullptr, 0, nullptr);
    }

    // profiling using timestamp event privided by level-zero
    for (int r = 0; r < rounds; r++) {
      Event event(queue->zeContext_, queue->zeDevice_);
      // Flush the L3 cache (global memory cache).
      if (getenv("IMEX_ENABLE_CACHE_FLUSHING")) {
        int init_val = 0;
        CHECK_ZE_RESULT(zeCommandListAppendMemoryFill(
            queue->zeCommandList_, cache, &init_val, 1, cacheSize, NULL, 0,
            NULL));
      }

      enqueueKernel(queue->zeCommandList_, kernel, &launchArgs, params,
                    sharedMemBytes, event.zeEvent, 0, nullptr);

      auto startTime =
          event.get_profiling_info<imex::profiling::command_start>();
      auto endTime = event.get_profiling_info<imex::profiling::command_end>();
      auto duration = float(endTime - startTime) / 1000000.0f;
      executionTime += duration;
      if (duration > maxTime)
        maxTime = duration;
      if (duration < minTime)
        minTime = duration;
    }
    deallocDeviceMemory(queue, cache);
    fprintf(stdout,
            "the kernel execution time is (ms, on L0 runtime):"
            "avg: %.4f, min: %.4f, max: %.4f (over %d runs)\n",
            executionTime / rounds, minTime, maxTime, rounds);
  } else {
    enqueueKernel(queue->zeCommandList_, kernel, &launchArgs, params,
                  sharedMemBytes, nullptr, 0, nullptr);
  }
}

// Wrappers
extern "C" LEVEL_ZERO_RUNTIME_EXPORT GPUL0QUEUE *
gpuCreateStream(void *device, void *context) {
  return catchAll([&]() {
    if (!device && !context) {
      return new GPUL0QUEUE();
    } else if (device && context) {
      // TODO: Check if the pointers/address is valid and holds the correct
      // device and context
      return new GPUL0QUEUE(static_cast<ze_device_type_t *>(device),
                            static_cast<ze_context_handle_t>(context));
    } else if (device && !context) {
      return new GPUL0QUEUE(static_cast<ze_device_type_t *>(device));
    } else {
      return new GPUL0QUEUE(static_cast<ze_context_handle_t>(context));
    }
  });
}

extern "C" LEVEL_ZERO_RUNTIME_EXPORT void gpuStreamDestroy(GPUL0QUEUE *queue) {
  catchAll([&]() { delete queue; });
}

extern "C" LEVEL_ZERO_RUNTIME_EXPORT void *
gpuMemAlloc(GPUL0QUEUE *queue, size_t size, size_t alignment, bool isShared) {
  return catchAll(
      [&]() { return allocDeviceMemory(queue, size, alignment, isShared); });
}

extern "C" LEVEL_ZERO_RUNTIME_EXPORT void gpuMemFree(GPUL0QUEUE *queue,
                                                     void *ptr) {
  catchAll([&]() { deallocDeviceMemory(queue, ptr); });
}

extern "C" LEVEL_ZERO_RUNTIME_EXPORT void
gpuMemCopy(GPUL0QUEUE *queue, void *dstPtr, void *srcPtr, size_t size) {
  return catchAll([&]() { memoryCopy(queue, dstPtr, srcPtr, size); });
}

extern "C" LEVEL_ZERO_RUNTIME_EXPORT ze_module_handle_t
gpuModuleLoad(GPUL0QUEUE *queue, const void *data, size_t dataSize) {
  return catchAll([&]() { return loadModule(queue, data, dataSize); });
}

extern "C" LEVEL_ZERO_RUNTIME_EXPORT ze_kernel_handle_t
gpuKernelGet(GPUL0QUEUE *queue, ze_module_handle_t module, const char *name) {
  return catchAll([&]() { return getKernel(queue, module, name); });
}

extern "C" LEVEL_ZERO_RUNTIME_EXPORT void
gpuLaunchKernel(GPUL0QUEUE *queue, ze_kernel_handle_t kernel, size_t gridX,
                size_t gridY, size_t gridZ, size_t blockX, size_t blockY,
                size_t blockZ, size_t sharedMemBytes, void *params) {
  return catchAll([&]() {
    launchKernel(queue, kernel, gridX, gridY, gridZ, blockX, blockY, blockZ,
                 sharedMemBytes, static_cast<ParamDesc *>(params));
  });
}

extern "C" LEVEL_ZERO_RUNTIME_EXPORT void gpuWait(GPUL0QUEUE *queue) {
  catchAll([&]() {
    // TODO: Find out ze Wait for host.
    return;
  });
}
