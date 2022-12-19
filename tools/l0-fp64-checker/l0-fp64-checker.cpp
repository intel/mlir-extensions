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

#include <cstring>
#include <iostream>

#define VALIDATECALL(zeCall) \
    if (zeCall != ZE_RESULT_SUCCESS){ \
	exit(1); \
    }

int main() {

    // Initialization
    VALIDATECALL(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    // Get the driver
    uint32_t driverCount = 0;
    VALIDATECALL(zeDriverGet(&driverCount, nullptr));

    if(driverCount == 0) {
	exit(1);
    }
    ze_driver_handle_t driverHandle;
    VALIDATECALL(zeDriverGet(&driverCount, &driverHandle));

    // Get the device
    uint32_t deviceCount = 0;
    VALIDATECALL(zeDeviceGet(driverHandle, &deviceCount, nullptr));

    if(deviceCount == 0) {
	exit(1);
    }
    ze_device_handle_t device;
    VALIDATECALL(zeDeviceGet(driverHandle, &deviceCount, &device));

    ze_device_module_properties_t moduleProperties;
    VALIDATECALL(zeDeviceGetModuleProperties(device, &moduleProperties));

    if(moduleProperties.fp64flags == 0)
	exit(1);
    return 0;
}
