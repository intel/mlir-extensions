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
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "dpcomp-gpu-runtime_export.h"

#include "level_zero_wrapper.hpp"

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

namespace {

template <typename CheckFunc>
std::pair<ze_driver_handle_t, ze_device_handle_t>
getDevice(CheckFunc &&checkFunc) {
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

struct L0State {
  static std::unique_ptr<L0State> get() {
    CHECK_ZE_RESULT(zeInit(0));

    auto driverAndDev = getDevice([](ze_device_handle_t device) {
      ze_device_properties_t props = {};
      CHECK_ZE_RESULT(zeDeviceGetProperties(device, &props));
      return props.type == ZE_DEVICE_TYPE_GPU;
    });
    auto driver = std::get<0>(driverAndDev);
    auto device = std::get<1>(driverAndDev);

    ze_context_desc_t contextDesc = {};
    auto context = ze::Context::create(driver, contextDesc);

    return std::unique_ptr<L0State>(new L0State(device, std::move(context)));
  }

  ze_command_list_handle_t getCommandList() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!commandLists.empty()) {
      auto ret = std::move(commandLists.back());
      commandLists.pop_back();
      return ret.release();
    }

    ze_command_queue_desc_t desc = {};
    desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    return ze::CommandList::createImmediate(context, device, desc).release();
  }

  void returnCommandList(ze_command_list_handle_t handle) {
    assert(handle);
    std::lock_guard<std::mutex> lock(mutex);
    commandLists.push_back(ze::CommandList(handle));
  }

  ze_module_handle_t getModule(const void *data, size_t dataSize) {
    // We assume data is statically allocated and didn't change, so we can use
    // ptr value as key.
    assert(data);
    std::lock_guard<std::mutex> lock(mutex);
    auto it = modules.find(data);
    if (it != modules.end())
      return it->second.get();

    ze_module_desc_t desc = {};
    desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    desc.pInputModule = static_cast<const uint8_t *>(data);
    desc.inputSize = dataSize;
    auto mod = ze::Module::create(context, device, desc).first;
    auto modPtr = mod.get();
    modules[data] = std::move(mod);
    return modPtr;
  }

  void returnModule(ze_module_handle_t mod) {
    assert(mod);
    (void)mod;
    // Nothing To do, modules always owned by context
  }

  ze_kernel_handle_t getKernel(ze_module_handle_t module, const char* name) {
    assert(module);
    assert(name);
    auto key = std::make_pair(module, name);
    auto it = kernels.find(key);
    if (it != kernels.end())
      return it->second.get();

    ze_kernel_desc_t desc = {};
    auto kernel = ze::Kernel::create(module, desc);
    auto ret = kernel.get();
    kernels[key] = std::move(kernel);
    return ret;
  }

private:
  L0State(ze_device_handle_t dev, ze::Context ctx)
      : device(dev), context(std::move(ctx)) {}

  std::mutex mutex;
  ze_device_handle_t device;
  ze::Context context;

  std::vector<ze::CommandList> commandLists;

  std::unordered_map<const void *, ze::Module> modules;

  using KernelKey = std::pair<ze_module_handle_t, const char*>;
  struct KernelHasher
  {
    size_t operator()(const KernelKey val) const {
      return std::hash<decltype (val.first)>()(val.first) |
             std::hash<decltype (val.second)>()(val.second);
    }
  };

  std::unordered_map<KernelKey, ze::Kernel, KernelHasher> kernels;
};
} // namespace

extern "C" DPCOMP_GPU_RUNTIME_EXPORT void *mgpuModuleLoad(const void *ptr) {
  LOG_FUNC();
  auto data = static_cast<const uint32_t *>(ptr);
  auto size = data[0] * sizeof(uint32_t);
  ++data;
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
