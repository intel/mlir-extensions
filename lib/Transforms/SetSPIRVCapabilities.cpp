//===- SetSPIRVCapabilities.cpp - SetSPIRVCapabilities Pass  -------*- C++
//-*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file adds various capabilties & extensions for the SPIRV execution.
///
//===----------------------------------------------------------------------===//

#include <imex/Transforms/Passes.h>

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Pass/Pass.h>

namespace imex {
#define GEN_PASS_DEF_SETSPIRVCAPABILITIES
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

namespace {
struct SetSPIRVCapabilitiesPass
    : public imex::impl::SetSPIRVCapabilitiesBase<SetSPIRVCapabilitiesPass> {
public:
  explicit SetSPIRVCapabilitiesPass() { m_clientAPI = "vulkan"; }
  explicit SetSPIRVCapabilitiesPass(const mlir::StringRef &clientAPI)
      : m_clientAPI(clientAPI) {}

  mlir::LogicalResult initializeOptions(mlir::StringRef options) override {
    if (failed(Pass::initializeOptions(options)))
      return mlir::failure();

    if (clientAPI == "opencl") {
      m_clientAPI = "opencl";
    }

    if (clientAPI != "vulkan" && clientAPI != "opencl")
      return mlir::failure();

    return mlir::success();
  }

  void runOnOperation() override {
    namespace spirv = mlir::spirv;
    auto context = &getContext();
    spirv::Capability caps_opencl[] = {
        // clang-format off
        spirv::Capability::Addresses,
        spirv::Capability::Float16Buffer,
        spirv::Capability::Int64,
        spirv::Capability::Int16,
        spirv::Capability::Int8,
        spirv::Capability::Kernel,
        spirv::Capability::Linkage,
        spirv::Capability::Vector16,
        spirv::Capability::GenericPointer,
        spirv::Capability::Groups,
        spirv::Capability::Float16,
        spirv::Capability::Float64,
        spirv::Capability::AtomicFloat32AddEXT,
        spirv::Capability::ExpectAssumeKHR,
        // clang-format on
    };
    spirv::Capability caps_vulkan[] = {
        // clang-format off
        spirv::Capability::Shader,
        // clang-format on
    };
    spirv::Extension exts_opencl[] = {
        spirv::Extension::SPV_EXT_shader_atomic_float_add,
        spirv::Extension::SPV_KHR_expect_assume};
    spirv::Extension exts_vulkan[] = {
        spirv::Extension::SPV_KHR_storage_buffer_storage_class};
    if (m_clientAPI == "opencl") {
      auto triple = spirv::VerCapExtAttr::get(
          spirv::Version::V_1_0, caps_opencl, exts_opencl, context);
      auto attr = spirv::TargetEnvAttr::get(
          triple, spirv::Vendor::Unknown, spirv::DeviceType::Unknown,
          spirv::TargetEnvAttr::kUnknownDeviceID,
          spirv::getDefaultResourceLimits(context));
      auto op = getOperation();
      op->walk([&](mlir::gpu::GPUModuleOp op) {
        op->setAttr(spirv::getTargetEnvAttrName(), attr);
      });
    } else if (m_clientAPI == "vulkan") {
      auto triple = spirv::VerCapExtAttr::get(
          spirv::Version::V_1_0, caps_vulkan, exts_vulkan, context);
      auto attr = spirv::TargetEnvAttr::get(
          triple, spirv::Vendor::Unknown, spirv::DeviceType::Unknown,
          spirv::TargetEnvAttr::kUnknownDeviceID,
          spirv::getDefaultResourceLimits(context));
      auto op = getOperation();
      op->walk([&](mlir::gpu::GPUModuleOp op) {
        op->setAttr(spirv::getTargetEnvAttrName(), attr);
      });
    }
  }

private:
  mlir::StringRef m_clientAPI;
};

} // namespace

namespace imex {
std::unique_ptr<mlir::Pass> createSetSPIRVCapabilitiesPass() {
  return std::make_unique<SetSPIRVCapabilitiesPass>();
}
} // namespace imex
