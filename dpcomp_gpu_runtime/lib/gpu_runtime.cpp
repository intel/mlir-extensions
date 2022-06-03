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

#include "level_zero_printing.hpp"
#include "level_zero_wrapper.hpp"

typedef void (*MemInfoDtorFunction)(void *ptr, size_t size, void *info);
using AllocFuncT = void *(*)(void *, size_t, MemInfoDtorFunction, void *);

#if 0 // Log functions
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

namespace {
static bool printGpuInfoEnabled() {
  static bool value = []() {
    auto env = std::getenv("DPCOMP_PRINT_GPU_INFO");
    return env != nullptr && std::atoi(env) != 0;
  }();
  return value;
}

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

static AllocFuncT AllocFunc = nullptr;

struct DeviceDesc {
  ze_driver_handle_t driver = nullptr;
  ze_device_handle_t device = nullptr;
};

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

template <typename CheckFunc>
static DeviceDesc getDevice(CheckFunc &&checkFunc) {
  uint32_t driverCount = 0;
  CHECK_ZE_RESULT(zeDriverGet(&driverCount, nullptr));

  auto drivers = std::make_unique<ze_driver_handle_t[]>(driverCount);
  CHECK_ZE_RESULT(zeDriverGet(&driverCount, drivers.get()));

  std::vector<ze_device_handle_t>
      devices; // vector here so we can reuse memory between iterations
  for (uint32_t i = 0; i < driverCount; ++i) {
    auto driver = drivers[i];
    assert(driver);
    uint32_t deviceCount = 0;
    CHECK_ZE_RESULT(zeDeviceGet(driver, &deviceCount, nullptr));
    devices.resize(deviceCount);
    CHECK_ZE_RESULT(zeDeviceGet(driver, &deviceCount, devices.data()));
    for (uint32_t i = 0; i < deviceCount; ++i) {
      auto device = devices[i];
      assert(device);
      if (checkFunc(device))
        return {driver, device};
    }
  }
  throw std::runtime_error("getDevice failed");
}

static DeviceDesc getDevice() {
  CHECK_ZE_RESULT(zeInit(0));
  return getDevice([](ze_device_handle_t device) {
    ze_device_properties_t props = {};
    CHECK_ZE_RESULT(zeDeviceGetProperties(device, &props));
    return props.type == ZE_DEVICE_TYPE_GPU;
  });
}

struct Printer {
  void operator()(const char *str) const { fprintf(stdout, "%s", str); }

  void operator()(const std::string &str) const {
    this->operator()(str.c_str());
  }

  void operator()(int64_t val) const { this->operator()(std::to_string(val)); }
};

static void printDriverProps(ze_driver_handle_t driver) {
  assert(driver);
  ze_api_version_t version = {};
  CHECK_ZE_RESULT(zeDriverGetApiVersion(driver, &version));
  auto major = static_cast<int>(ZE_MAJOR_VERSION(version));
  auto minor = static_cast<int>(ZE_MINOR_VERSION(version));
  fprintf(stdout, "Driver API version: %d.%d\n", major, minor);

  ze_driver_properties_t props = {};
  props.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
  CHECK_ZE_RESULT(zeDriverGetProperties(driver, &props));
  fprintf(stdout, "Driver version: %d\n",
          static_cast<int>(props.driverVersion));
}

static void printDeviceProps(ze_device_handle_t device) {
  assert(device);
  Printer printer;
  ze_device_properties_t deviceProperties = {};
  deviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  CHECK_ZE_RESULT(zeDeviceGetProperties(device, &deviceProperties));
  printer("\nDevice properties:\n");
  print(deviceProperties, printer);

  ze_device_compute_properties_t computeProperties = {};
  computeProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
  CHECK_ZE_RESULT(zeDeviceGetComputeProperties(device, &computeProperties));
  printer("\nDevice compute properties:\n");
  print(computeProperties, printer);

  ze_device_module_properties_t moduleProperties = {};
  moduleProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
  CHECK_ZE_RESULT(zeDeviceGetModuleProperties(device, &moduleProperties));
  printer("\nDevice module properties:\n");
  print(moduleProperties, printer);
}

static auto countEvents(ze_event_handle_t *events) {
  assert(events);
  return static_cast<uint32_t>(
      countUntil(events, static_cast<ze_event_handle_t>(nullptr)));
}

struct Stream {
  Stream(size_t eventsCount) {
    auto driverAndDevice = getDevice();
    driver = driverAndDevice.driver;
    device = driverAndDevice.device;

    if (printGpuInfoEnabled()) {
      printDriverProps(driver);
      printDeviceProps(device);
    }

    ze_context_desc_t contextDesc = {};
    context = ze::Context::create(driver, contextDesc);

    ze_command_queue_desc_t desc = {};
    desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    commandList = ze::CommandList::createImmediate(context, device, desc);

    if (eventsCount > 0) {
      ze_event_pool_desc_t poolDesc = {};
      poolDesc.count = static_cast<uint32_t>(eventsCount);
      eventPool = ze::EventPool::create(context, poolDesc, 0, nullptr);
      events = std::make_unique<ze::Event[]>(eventsCount);
    }
  }

  void retain() { ++refcout; }

  void release() {
    if (--refcout == 0)
      delete this;
  }

  struct Releaser {
    Releaser(Stream *s) : stream(s) { assert(stream); }

    ~Releaser() { stream->release(); }

  private:
    Stream *stream;
  };

  ze_module_handle_t loadModule(const void *data, size_t dataSize) {
    assert(data);
    ze_module_desc_t desc = {};
    desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    desc.pInputModule = static_cast<const uint8_t *>(data);
    desc.inputSize = dataSize;
    auto module = ze::Module::create(context, device, desc).first;
    return module.release();
  }

  static auto destroyModule(ze_module_handle_t module) {
    assert(module);
    ze::Module temp(module);
  }

  static ze_kernel_handle_t getKernel(ze_module_handle_t module,
                                      const char *name) {
    assert(module);
    assert(name);
    ze_kernel_desc_t desc = {};
    desc.pKernelName = name;
    auto kernel = ze::Kernel::create(module, desc);
    return kernel.release();
  }

  static void destroyKernel(ze_kernel_handle_t kernel) {
    assert(kernel);
    ze::Kernel temp(kernel);
  }

  ze_event_handle_t launchKernel(ze_kernel_handle_t kernel, size_t gridX,
                                 size_t gridY, size_t gridZ, size_t blockX,
                                 size_t blockY, size_t blockZ,
                                 ze_event_handle_t *events, ParamDesc *params,
                                 size_t eventIndex) {
    assert(kernel);
    auto eventsCount = countEvents(events);
    auto paramsCount = countUntil(params, ParamDesc{nullptr, 0});

    auto castSz = [](size_t val) { return static_cast<uint32_t>(val); };

    CHECK_ZE_RESULT(zeKernelSetGroupSize(kernel, castSz(blockX), castSz(blockY),
                                         castSz(blockZ)));
    for (size_t i = 0; i < paramsCount; ++i) {
      auto param = params[i];
      CHECK_ZE_RESULT(zeKernelSetArgumentValue(kernel, static_cast<uint32_t>(i),
                                               param.size, param.data));
    }

    auto event = getEvent(eventIndex);
    ze_group_count_t launchArgs = {castSz(gridX), castSz(gridY), castSz(gridZ)};
    CHECK_ZE_RESULT(zeCommandListAppendLaunchKernel(
        commandList.get(), kernel, &launchArgs, event, eventsCount, events));
    return event;
  }

  static void waitEvent(ze_event_handle_t event) {
    assert(event);
    CHECK_ZE_RESULT(zeEventHostSynchronize(event, UINT64_MAX));
  }

  std::tuple<void *, void *, ze_event_handle_t>
  allocBuffer(size_t size, size_t alignment, bool shared,
              ze_event_handle_t *events, size_t eventIndex,
              AllocFuncT allocFunc) {
    // Alloc is always sync for now, synchronize
    auto eventsCount = countEvents(events);
    for (decltype(eventsCount) i = 0; i < eventsCount; ++i) {
      CHECK_ZE_RESULT(zeEventHostSynchronize(events[i], UINT64_MAX));
    }

    auto dtor = [](void *ptr, size_t /*size*/, void *info) {
      assert(ptr);
      assert(info);
      auto *stream = static_cast<Stream *>(info);
      Releaser r(stream);
      CHECK_ZE_RESULT(zeMemFree(stream->context.get(), ptr));
    };

    auto event = getEvent(eventIndex);
    if (event)
      CHECK_ZE_RESULT(zeEventHostSignal(event));

    auto mem = [&]() -> void * {
      void *ret = nullptr;
      ze_device_mem_alloc_desc_t devDesc = {};
      devDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
      if (shared) {
        ze_host_mem_alloc_desc_t hostDesc = {};
        hostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
        CHECK_ZE_RESULT(zeMemAllocShared(context.get(), &devDesc, &hostDesc,
                                         size, alignment, device, &ret));
      } else {
        devDesc.flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT;
        CHECK_ZE_RESULT(zeMemAllocDevice(context.get(), &devDesc, size,
                                         alignment, device, &ret));
      }
      return ret;
    }();
    assert(mem);

    auto info = [&]() -> void * {
      if (allocFunc)
        return allocFunc(mem, size, dtor, this);

      return mem;
    }();

    if (!info) {
      zeMemFree(context.get(), mem);
      throw std::runtime_error("Failed to allocate MemInfo");
    }

    #if defined(IMEX_ENABLE_NUMBA_HOTFIX)
       retain();
    #endif
    return {info, mem, event};
  }

  void deallocBuffer(void *ptr) { zeMemFree(context.get(), ptr); }

  void suggestBlockSize(ze_kernel_handle_t kernel, const uint32_t *gridSize,
                        uint32_t *blockSize, size_t numDims) {
    assert(kernel);
    assert(numDims > 0 && numDims <= 3);
    uint32_t gSize[3] = {};
    uint32_t *bSize[3] = {};
    for (size_t i = 0; i < numDims; ++i) {
      gSize[i] = gridSize[i];
      bSize[i] = &blockSize[i];
    }

    CHECK_ZE_RESULT(zeKernelSuggestGroupSize(
        kernel, gSize[0], gSize[1], gSize[2], bSize[0], bSize[1], bSize[2]));
  }

private:
  std::atomic<unsigned> refcout = {1};
  ze_driver_handle_t driver = nullptr;
  ze_device_handle_t device = nullptr;
  ze::Context context;
  ze::CommandList commandList;

  ze::EventPool eventPool;
  std::unique_ptr<ze::Event[]> events;

  static const constexpr size_t NoEvent = static_cast<size_t>(-1);

  ze_event_handle_t getEvent(size_t index) {
    if (index == NoEvent)
      return nullptr;

    assert(eventPool);
    if (events[index] != nullptr) {
      auto ev = events[index].get();
      CHECK_ZE_RESULT(zeEventHostReset(ev));
      return ev;
    }

    ze_event_desc_t desc = {};
    desc.index = static_cast<uint32_t>(index);
    desc.signal = ZE_EVENT_SCOPE_FLAG_DEVICE | ZE_EVENT_SCOPE_FLAG_HOST;
    desc.wait = ZE_EVENT_SCOPE_FLAG_DEVICE | ZE_EVENT_SCOPE_FLAG_HOST;
    auto event = ze::Event::create(eventPool, desc);
    auto ev = event.get();
    events[index] = std::move(event);
    return ev;
  }
};
} // namespace

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void
dpcompGpuSetMemInfoAllocFunc(void *func) {
  LOG_FUNC();
  AllocFunc = reinterpret_cast<AllocFuncT>(func);
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *
dpcompGpuStreamCreate(size_t eventsCount) {
  LOG_FUNC();
  return catchAll([&]() { return new Stream(eventsCount); });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void dpcompGpuStreamDestroy(void *stream) {
  LOG_FUNC();
  catchAll([&]() { static_cast<Stream *>(stream)->release(); });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *
dpcompGpuModuleLoad(void *stream, const void *data, size_t dataSize) {
  LOG_FUNC();
  return catchAll([&]() {
    return static_cast<Stream *>(stream)->loadModule(data, dataSize);
  });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void dpcompGpuModuleDestroy(void *module) {
  LOG_FUNC();
  catchAll([&]() {
    Stream::destroyModule(static_cast<ze_module_handle_t>(module));
  });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *
dpcompGpuKernelGet(void *module, const char *name) {
  LOG_FUNC();
  return catchAll([&]() {
    return Stream::getKernel(static_cast<ze_module_handle_t>(module), name);
  });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void dpcompGpuKernelDestroy(void *kernel) {
  LOG_FUNC();
  catchAll([&]() {
    Stream::destroyKernel(static_cast<ze_kernel_handle_t>(kernel));
  });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *
dpcompGpuLaunchKernel(void *stream, void *kernel, size_t gridX, size_t gridY,
                      size_t gridZ, size_t blockX, size_t blockY, size_t blockZ,
                      void *events, void *params, size_t eventIndex) {
  LOG_FUNC();
  return catchAll([&]() {
    return static_cast<Stream *>(stream)->launchKernel(
        static_cast<ze_kernel_handle_t>(kernel), gridX, gridY, gridZ, blockX,
        blockY, blockZ, static_cast<ze_event_handle_t *>(events),
        static_cast<ParamDesc *>(params), eventIndex);
  });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void dpcompGpuWait(void *event) {
  LOG_FUNC();
  catchAll([&]() { Stream::waitEvent(static_cast<ze_event_handle_t>(event)); });
}

struct AllocResult {
  void *info;
  void *ptr;
  void *event;
};

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void
dpcompGpuAlloc(void *stream, size_t size, size_t alignment, int shared,
               void *events, size_t eventIndex, AllocResult *ret) {
  LOG_FUNC();
  catchAll([&]() {
    auto res = static_cast<Stream *>(stream)->allocBuffer(
        size, alignment, shared != 0, static_cast<ze_event_handle_t *>(events),
        eventIndex, AllocFunc);
    *ret = AllocResult{std::get<0>(res), std::get<1>(res), std::get<2>(res)};
  });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void dpcompGpuDeAlloc(void *stream,
                                                           void *ptr) {
  LOG_FUNC();
  catchAll([&]() { static_cast<Stream *>(stream)->deallocBuffer(ptr); });
}

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void
dpcompGpuSuggestBlockSize(void *stream, void *kernel, const uint32_t *gridSize,
                          uint32_t *blockSize, size_t numDims) {
  LOG_FUNC();
  catchAll([&]() {
    static_cast<Stream *>(stream)->suggestBlockSize(
        static_cast<ze_kernel_handle_t>(kernel), gridSize, blockSize, numDims);
  });
}
