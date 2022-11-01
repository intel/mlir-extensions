// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <optional>
#include <string_view>

struct DeviceDesc {
  std::string_view backend;
  std::string_view name;
  int index = -1;
};

std::optional<DeviceDesc> parseFilterString(std::string_view filterString);
