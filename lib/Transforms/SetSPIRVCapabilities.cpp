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
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
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
  explicit SetSPIRVCapabilitiesPass(const mlir::StringRef &clientAPI)
      : m_clientAPI(clientAPI) {}

  mlir::LogicalResult
  initializeOptions(mlir::StringRef options,
                    mlir::function_ref<mlir::LogicalResult(const llvm::Twine &)>
                        errorHandler) override {
    if (mlir::failed(Pass::initializeOptions(options, errorHandler)))
      return mlir::failure();

    if (clientAPI != "vulkan" && clientAPI != "opencl")
      return errorHandler(llvm::Twine("Invalid clienAPI: ") + clientAPI);
    m_clientAPI = clientAPI;

    return mlir::success();
  }

  void runOnOperation() override {
    namespace spirv = mlir::spirv;
    auto context = &getContext();
    spirv::Capability caps_opencl[] = {
        // clang-format off
        spirv::Capability::Addresses,
        spirv::Capability::Bfloat16ConversionINTEL,
        spirv::Capability::BFloat16TypeKHR,
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
        spirv::Capability::VectorAnyINTEL,
        spirv::Capability::VectorComputeINTEL,
        // clang-format on
    };
    spirv::Capability caps_vulkan[] = {
        // clang-format off
        spirv::Capability::Shader,
        // clang-format on
    };
    spirv::Extension exts_opencl[] = {
        // clang-format off
        spirv::Extension::SPV_EXT_shader_atomic_float_add,
        spirv::Extension::SPV_KHR_bfloat16,
        spirv::Extension::SPV_KHR_expect_assume,
        spirv::Extension::SPV_INTEL_bfloat16_conversion,
        spirv::Extension::SPV_INTEL_vector_compute
        // clang-format on
    };
    spirv::Extension exts_vulkan[] = {
        spirv::Extension::SPV_KHR_storage_buffer_storage_class};
    auto op = getOperation();
    op->walk([&](mlir::gpu::GPUModuleOp gmod) {
      auto oldAttr = gmod->getAttrOfType<mlir::spirv::TargetEnvAttr>(
          spirv::getTargetEnvAttrName());
      if (!oldAttr) {
        if (m_clientAPI == "opencl") {
          auto triple = spirv::VerCapExtAttr::get(
              spirv::Version::V_1_4, caps_opencl, exts_opencl, context);
          auto attr = spirv::TargetEnvAttr::get(
              triple, spirv::getDefaultResourceLimits(context),
              spirv::ClientAPI::OpenCL, spirv::Vendor::Unknown,
              spirv::DeviceType::Unknown,
              spirv::TargetEnvAttr::kUnknownDeviceID);
          gmod->setAttr(spirv::getTargetEnvAttrName(), attr);
        } else if (m_clientAPI == "vulkan") {
          auto triple = spirv::VerCapExtAttr::get(
              spirv::Version::V_1_4, caps_vulkan, exts_vulkan, context);
          auto attr = spirv::TargetEnvAttr::get(
              triple, spirv::getDefaultResourceLimits(context),
              spirv::ClientAPI::Vulkan, spirv::Vendor::Unknown,
              spirv::DeviceType::Unknown,
              spirv::TargetEnvAttr::kUnknownDeviceID);
          gmod->setAttr(spirv::getTargetEnvAttrName(), attr);
        }
      }
    });
  }

private:
  mlir::StringRef m_clientAPI;
};

} // namespace

namespace imex {
std::unique_ptr<mlir::Pass>
createSetSPIRVCapabilitiesPass(mlir::StringRef api) {
  return std::make_unique<SetSPIRVCapabilitiesPass>(api);
}
std::unique_ptr<mlir::Pass> createSetSPIRVCapabilitiesPass() {
  return createSetSPIRVCapabilitiesPass("opencl");
}
} // namespace imex
