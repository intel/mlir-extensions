// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>
#include <string>

#include <llvm/ADT/Optional.h>

// Must be kept in sync with gpu_runtime version.
struct OffloadDeviceCapabilities {
  uint16_t spirvMajorVersion;
  uint16_t spirvMinorVersion;
  bool hasFP16;
  bool hasFP64;
};

// TODO: device name
llvm::Optional<OffloadDeviceCapabilities>
getOffloadDeviceCapabilities(const std::string &name);

llvm::Optional<std::string> getDefaultDevice();
