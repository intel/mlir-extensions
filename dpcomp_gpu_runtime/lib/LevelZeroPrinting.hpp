// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <type_traits>

#include <level_zero/ze_api.h>

template <typename T, typename F>
auto print(T val, F &&printer) ->
    typename std::enable_if<std::is_arithmetic<T>::value>::type {
  printer(std::to_string(val));
}

template <typename T, size_t N, typename F>
void print(T (&array)[N], F &&printer) {
  printer("[");
  for (size_t i = 0; i < N; ++i) {
    if (i != 0)
      printer(", ");

    print(array[i], printer);
  }
  printer("]");
}

template <typename F> void print(const char *str, F &&printer) { printer(str); }

template <typename F> void print(ze_device_type_t val, F &&printer) {
  auto index = static_cast<size_t>(val);
  std::array<const char *, 6> res;
  if (index > res.size()) {
    printer("?");
    return;
  }

  std::fill(res.begin(), res.end(), "?");

#define ENUM_ELEMENT(elem) res[static_cast<size_t>(elem)] = #elem
  ENUM_ELEMENT(ZE_DEVICE_TYPE_GPU);
  ENUM_ELEMENT(ZE_DEVICE_TYPE_CPU);
  ENUM_ELEMENT(ZE_DEVICE_TYPE_FPGA);
  ENUM_ELEMENT(ZE_DEVICE_TYPE_MCA);
  ENUM_ELEMENT(ZE_DEVICE_TYPE_VPU);
#undef ENUM_ELEM

  printer(res[index]);
}

template <typename F> void print(ze_device_property_flag_t flags, F &&printer) {
  printer("[");
  bool first = true;
  auto printFlag = [&](auto val, auto name) {
    if (flags & val) {
      if (!first)
        printer("|");

      printer(name);
      first = false;
    }
  };
#define PRINT_FLAG(flag) printFlag(flag, #flag)
  PRINT_FLAG(ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
  PRINT_FLAG(ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE);
  PRINT_FLAG(ZE_DEVICE_PROPERTY_FLAG_ECC);
  PRINT_FLAG(ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING);
#undef PRINT_FLAG
  printer("]");
}

template <typename F>
void print(const ze_device_properties_t val, F &&printer) {
  auto write = [&](auto desc, auto val) {
    printer(desc);
    print(val, printer);
    printer("\n");
  };

  write("Device::properties_t::type : ", val.type);
  write("Device::properties_t::vendorId : ", val.vendorId);
  write("Device::properties_t::deviceId : ", val.deviceId);
  write("Device::properties_t::flags : ",
        static_cast<ze_device_property_flag_t>(val.flags));
  write("Device::properties_t::subdeviceId : ", val.subdeviceId);
  write("Device::properties_t::coreClockRate : ", val.coreClockRate);
  write("Device::properties_t::maxMemAllocSize : ", val.maxMemAllocSize);
  write("Device::properties_t::maxHardwareContexts : ",
        val.maxHardwareContexts);
  write("Device::properties_t::maxCommandQueuePriority : ",
        val.maxCommandQueuePriority);
  write("Device::properties_t::numThreadsPerEU : ", val.numThreadsPerEU);
  write("Device::properties_t::physicalEUSimdWidth : ",
        val.physicalEUSimdWidth);
  write("Device::properties_t::numEUsPerSubslice : ", val.numEUsPerSubslice);
  write("Device::properties_t::numSubslicesPerSlice : ",
        val.numSubslicesPerSlice);
  write("Device::properties_t::numSlices : ", val.numSlices);
  write("Device::properties_t::timerResolution : ", val.timerResolution);
  write("Device::properties_t::timestampValidBits : ", val.timestampValidBits);
  write("Device::properties_t::kernelTimestampValidBits : ",
        val.kernelTimestampValidBits);
  write("Device::properties_t::name : ", val.name);
}

template <typename F>
void print(const ze_device_compute_properties_t val, F &&printer) {
  auto write = [&](auto desc, auto val) {
    printer(desc);
    print(val, printer);
    printer("\n");
  };
  write("maxTotalGroupSize : ", val.maxTotalGroupSize);
  write("maxGroupSizeX : ", val.maxGroupSizeX);
  write("maxGroupSizeY : ", val.maxGroupSizeY);
  write("maxGroupSizeZ : ", val.maxGroupSizeZ);
  write("maxGroupCountX : ", val.maxGroupCountX);
  write("maxGroupCountY : ", val.maxGroupCountY);
  write("maxGroupCountZ : ", val.maxGroupCountZ);
  write("maxSharedLocalMemory : ", val.maxSharedLocalMemory);
  write("numSubGroupSizes : ", val.numSubGroupSizes);
  printer("subGroupSizes : ");
  print(val.subGroupSizes, printer);
  printer("\n");
}

template <typename F> void print(ze_device_module_flag_t flags, F &&printer) {
  printer("[");
  bool first = true;
  auto printFlag = [&](auto val, auto name) {
    if (flags & val) {
      if (!first)
        printer("|");

      printer(name);
      first = false;
    }
  };
#define PRINT_FLAG(flag) printFlag(flag, #flag)
  PRINT_FLAG(ZE_DEVICE_MODULE_FLAG_FP16);
  PRINT_FLAG(ZE_DEVICE_MODULE_FLAG_FP64);
  PRINT_FLAG(ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS);
  PRINT_FLAG(ZE_DEVICE_MODULE_FLAG_DP4A);
#undef PRINT_FLAG
  printer("]");
}

template <typename F> void print(ze_device_fp_flag_t flags, F &&printer) {
  printer("[");
  bool first = true;
  auto printFlag = [&](auto val, auto name) {
    if (flags & val) {
      if (!first)
        printer("|");

      printer(name);
      first = false;
    }
  };
#define PRINT_FLAG(flag) printFlag(flag, #flag)
  PRINT_FLAG(ZE_DEVICE_FP_FLAG_DENORM);
  PRINT_FLAG(ZE_DEVICE_FP_FLAG_INF_NAN);
  PRINT_FLAG(ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST);
  PRINT_FLAG(ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO);
  PRINT_FLAG(ZE_DEVICE_FP_FLAG_ROUND_TO_INF);
  PRINT_FLAG(ZE_DEVICE_FP_FLAG_FMA);
  PRINT_FLAG(ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT);
  PRINT_FLAG(ZE_DEVICE_FP_FLAG_SOFT_FLOAT);
#undef PRINT_FLAG
  printer("]");
}

template <typename F>
void print(ze_device_module_properties_t properties, F &&printer) {
  auto write = [&](auto desc, auto val) {
    printer(desc);
    print(val, printer);
    printer("\n");
  };

  printer("Supported spir-v version: ");
  printer(ZE_MAJOR_VERSION(properties.spirvVersionSupported));
  printer(".");
  printer(ZE_MINOR_VERSION(properties.spirvVersionSupported));
  printer("\n");

  write("Flags: ", static_cast<ze_device_module_flag_t>(properties.flags));
  write("FP16 flags: ", static_cast<ze_device_fp_flag_t>(properties.fp16flags));
  write("FP32 flags: ", static_cast<ze_device_fp_flag_t>(properties.fp32flags));
  write("FP64 flags: ", static_cast<ze_device_fp_flag_t>(properties.fp64flags));
  write("Max argument size: ", properties.maxArgumentsSize);
  write("Print buffer size: ", properties.printfBufferSize);
}
