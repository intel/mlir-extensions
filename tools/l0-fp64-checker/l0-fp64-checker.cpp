//===- l0-fp64-checker.cpp --------------------------------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines gpu fp64 support utility.
//
//===----------------------------------------------------------------------===//

#include <level_zero/ze_api.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#define VALIDATECALL(zeCall)                                                   \
  if (zeCall != ZE_RESULT_SUCCESS) {                                           \
    exit(1);                                                                   \
  }

// Find the first GPU device and check fp64 support
int main() {
  // Initialization
  VALIDATECALL(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  // Get drivers
  uint32_t driverCount = 0;
  VALIDATECALL(zeDriverGet(&driverCount, nullptr));

  if (driverCount == 0) {
    return 1;
  }

  std::vector<ze_driver_handle_t> drivers(driverCount);
  VALIDATECALL(zeDriverGet(&driverCount, drivers.data()));

  for (uint32_t i = 0; i < driverCount; ++i) {
    auto driver = drivers[i];
    // Get the device
    uint32_t deviceCount = 0;
    VALIDATECALL(zeDeviceGet(driver, &deviceCount, nullptr));

    if (deviceCount == 0) {
      continue;
    }

    std::vector<ze_device_handle_t> devices(deviceCount);
    VALIDATECALL(zeDeviceGet(driver, &deviceCount, devices.data()));
    for (uint32_t j = 0; j < deviceCount; ++j) {
      auto device = devices[j];
      ze_device_properties_t deviceProperties;
      deviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      VALIDATECALL(zeDeviceGetProperties(device, &deviceProperties));
      if (ZE_DEVICE_TYPE_GPU == deviceProperties.type) {
        ze_device_module_properties_t moduleProperties;
        moduleProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
        VALIDATECALL(zeDeviceGetModuleProperties(device, &moduleProperties));

        // GPU does not have fp64
        if (moduleProperties.fp64flags == 0) {
          return 1;
        } else {
          return 0;
        }
      }
    }
  }
  return 1;
}
