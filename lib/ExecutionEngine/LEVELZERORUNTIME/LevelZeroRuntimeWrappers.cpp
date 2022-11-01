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
#include <iostream>
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
get_driver_and_device(ze_device_type_t device_type = ZE_DEVICE_TYPE_GPU) {

  CHECK_ZE_RESULT(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t driverCount = 0;
  CHECK_ZE_RESULT(zeDriverGet(&driverCount, nullptr));

  std::vector<ze_driver_handle_t> allDrivers{driverCount};
  CHECK_ZE_RESULT(zeDriverGet(&driverCount, allDrivers.data()));

  // Find a driver instance with a GPU device
  for (uint32_t i = 0; i < driverCount; ++i) {
    uint32_t deviceCount = 0;
    CHECK_ZE_RESULT(zeDeviceGet(allDrivers[i], &deviceCount, nullptr));
    if (deviceCount == 0)
      continue;
    ze_device_handle_t *allDevices =
        (ze_device_handle_t *)malloc(deviceCount * sizeof(ze_device_handle_t));
    CHECK_ZE_RESULT(zeDeviceGet(allDrivers[i], &deviceCount, allDevices));
    for (uint32_t d = 0; d < deviceCount; ++d) {
      ze_device_properties_t device_properties;
      CHECK_ZE_RESULT(zeDeviceGetProperties(allDevices[d], &device_properties));
      if (device_type == device_properties.type) {
        auto driver = allDrivers[i];
        auto device = allDevices[d];
        return {driver, device};
      }
    }
  }
  throw std::runtime_error("getDevice failed");
}

struct GPU_L0_QUEUE {

  ze_driver_handle_t ze_driver_ = nullptr;
  ze_device_handle_t ze_device_ = nullptr;
  ze_context_handle_t ze_context_ = nullptr;
  ze_command_list_handle_t ze_commandList_ = nullptr;

  GPU_L0_QUEUE() {
    auto driverAndDevice = get_driver_and_device();
    ze_driver_ = driverAndDevice.first;
    ze_device_ = driverAndDevice.second;

    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr,
                                     0};
    CHECK_ZE_RESULT(zeContextCreate(ze_driver_, &contextDesc, &ze_context_));

    ze_command_queue_desc_t desc = {};
    desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    CHECK_ZE_RESULT(zeCommandListCreateImmediate(ze_context_, ze_device_, &desc,
                                                 &ze_commandList_));
  }
  GPU_L0_QUEUE(ze_device_type_t *device_type, ze_context_handle_t context) {
    auto driverAndDevice = get_driver_and_device(*device_type);
    ze_driver_ = driverAndDevice.first;
    ze_device_ = driverAndDevice.second;

    ze_context_ = context;

    ze_command_queue_desc_t desc = {};
    desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    CHECK_ZE_RESULT(zeCommandListCreateImmediate(ze_context_, ze_device_, &desc,
                                                 &ze_commandList_));
  }
  GPU_L0_QUEUE(ze_device_type_t *device_type) {

    auto driverAndDevice = get_driver_and_device(*device_type);
    ze_driver_ = driverAndDevice.first;
    ze_device_ = driverAndDevice.second;

    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr,
                                     0};
    CHECK_ZE_RESULT(zeContextCreate(ze_driver_, &contextDesc, &ze_context_));

    ze_command_queue_desc_t desc = {};
    desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    CHECK_ZE_RESULT(zeCommandListCreateImmediate(ze_context_, ze_device_, &desc,
                                                 &ze_commandList_));
  }
  GPU_L0_QUEUE(ze_context_handle_t context) {

    auto driverAndDevice = get_driver_and_device();
    ze_driver_ = driverAndDevice.first;
    ze_device_ = driverAndDevice.second;
    ze_context_ = context;

    ze_command_queue_desc_t desc = {};
    desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;

    CHECK_ZE_RESULT(zeCommandListCreateImmediate(ze_context_, ze_device_, &desc,
                                                 &ze_commandList_));
  }
};

void *alloc_device_memory(GPU_L0_QUEUE *queue, size_t size, size_t alignment,
                          bool is_shared) {

  void *ret = nullptr;
  auto gpu_l0_queue = queue;
  ze_device_mem_alloc_desc_t devDesc = {};
  devDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  if (is_shared) {
    ze_host_mem_alloc_desc_t hostDesc = {};
    hostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    CHECK_ZE_RESULT(zeMemAllocShared(gpu_l0_queue->ze_context_, &devDesc,
                                     &hostDesc, size, alignment,
                                     gpu_l0_queue->ze_device_, &ret));
  } else {
    devDesc.flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT;
    CHECK_ZE_RESULT(zeMemAllocDevice(gpu_l0_queue->ze_context_, &devDesc, size,
                                     alignment, gpu_l0_queue->ze_device_,
                                     &ret));
  }
  return ret;
}

void dealloc_device_memory(GPU_L0_QUEUE *queue, void *ptr) {
  zeMemFree(queue->ze_context_, ptr);
}

ze_module_handle_t loadModule(GPU_L0_QUEUE *queue, const void *data,
                              size_t dataSize) {
  assert(data);
  auto gpu_l0_queue = queue;
  ze_module_handle_t ze_module;
  ze_module_desc_t desc = {};
  desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  desc.pInputModule = static_cast<const uint8_t *>(data);
  desc.inputSize = dataSize;
  CHECK_ZE_RESULT(zeModuleCreate(gpu_l0_queue->ze_context_,
                                 gpu_l0_queue->ze_device_, &desc, &ze_module,
                                 nullptr));
  return ze_module;
}

ze_kernel_handle_t getKernel(GPU_L0_QUEUE *queue, ze_module_handle_t module,
                             const char *name) {
  assert(module);
  assert(name);
  ze_kernel_desc_t desc = {};
  ze_kernel_handle_t ze_kernel;
  desc.pKernelName = name;
  CHECK_ZE_RESULT(zeKernelCreate(module, &desc, &ze_kernel));
  return ze_kernel;
}

void launchKernel(GPU_L0_QUEUE *queue, ze_kernel_handle_t kernel, size_t gridX,
                  size_t gridY, size_t gridZ, size_t blockX, size_t blockY,
                  size_t blockZ, size_t sharedMemBytes, ParamDesc *params) {
  assert(kernel);
  auto paramsCount = countUntil(params, ParamDesc{nullptr, 0});

  auto castSz = [](size_t val) { return static_cast<uint32_t>(val); };

  CHECK_ZE_RESULT(zeKernelSetGroupSize(kernel, castSz(blockX), castSz(blockY),
                                       castSz(blockZ)));
  for (size_t i = 0; i < paramsCount; ++i) {
    auto param = params[i];
    CHECK_ZE_RESULT(zeKernelSetArgumentValue(kernel, static_cast<uint32_t>(i),
                                             param.size, param.data));
  }

  ze_group_count_t launchArgs = {castSz(gridX), castSz(gridY), castSz(gridZ)};
  CHECK_ZE_RESULT(zeCommandListAppendLaunchKernel(
      queue->ze_commandList_, kernel, &launchArgs, nullptr, 0, nullptr));
}

// Wrappers
extern "C" LEVEL_ZERO_RUNTIME_EXPORT GPU_L0_QUEUE *
gpuCreateStream(void *device, void *context) {
  return catchAll([&]() {
    if (!device && !context) {
      return new GPU_L0_QUEUE();
    } else if (device && context) {
      return new GPU_L0_QUEUE(static_cast<ze_device_type_t *>(device),
                              static_cast<ze_context_handle_t>(context));
    } else if (device && !context) {
      return new GPU_L0_QUEUE(static_cast<ze_device_type_t *>(device));
    } else {
      return new GPU_L0_QUEUE(static_cast<ze_context_handle_t>(context));
    }
  });
}

extern "C" LEVEL_ZERO_RUNTIME_EXPORT void
gpuStreamDestroy(GPU_L0_QUEUE *queue) {
  catchAll([&]() { delete queue; });
}

extern "C" LEVEL_ZERO_RUNTIME_EXPORT void *gpuMemAlloc(GPU_L0_QUEUE *queue,
                                                       size_t size,
                                                       size_t alignment,
                                                       bool is_shared) {
  return catchAll(
      [&]() { return alloc_device_memory(queue, size, alignment, is_shared); });
}

extern "C" LEVEL_ZERO_RUNTIME_EXPORT void gpuMemFree(GPU_L0_QUEUE *queue,
                                                     void *ptr) {
  catchAll([&]() { dealloc_device_memory(queue, ptr); });
}

extern "C" LEVEL_ZERO_RUNTIME_EXPORT ze_module_handle_t
gpuModuleLoad(GPU_L0_QUEUE *queue, const void *data, size_t dataSize) {
  return catchAll([&]() { return loadModule(queue, data, dataSize); });
}

extern "C" LEVEL_ZERO_RUNTIME_EXPORT ze_kernel_handle_t
gpuKernelGet(GPU_L0_QUEUE *queue, ze_module_handle_t module, const char *name) {
  return catchAll([&]() { return getKernel(queue, module, name); });
}

extern "C" LEVEL_ZERO_RUNTIME_EXPORT void
gpuLaunchKernel(GPU_L0_QUEUE *queue, ze_kernel_handle_t kernel, size_t gridX,
                size_t gridY, size_t gridZ, size_t blockX, size_t blockY,
                size_t blockZ, size_t sharedMemBytes, void *params) {
  return catchAll([&]() {
    launchKernel(queue, kernel, gridX, gridY, gridZ, blockX, blockY, blockZ,
                 sharedMemBytes, static_cast<ParamDesc *>(params));
  });
}

extern "C" LEVEL_ZERO_RUNTIME_EXPORT void gpuWait(GPU_L0_QUEUE *queue) {
  catchAll([&]() {
    // TODO: Find out ze Wait for host.
    return;
  });
}
