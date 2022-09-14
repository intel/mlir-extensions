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

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Pass/Pass.h>

namespace imex {
struct SetSPIRVCapabilitiesPass
    : public mlir::PassWrapper<SetSPIRVCapabilitiesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    namespace spirv = mlir::spirv;
    auto context = &getContext();
    spirv::Capability caps[] = {
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
    spirv::Extension exts[] = {
        spirv::Extension::SPV_EXT_shader_atomic_float_add,
        spirv::Extension::SPV_KHR_expect_assume};
    auto triple =
        spirv::VerCapExtAttr::get(spirv::Version::V_1_0, caps, exts, context);
    auto attr = spirv::TargetEnvAttr::get(
        triple, spirv::Vendor::Unknown, spirv::DeviceType::Unknown,
        spirv::TargetEnvAttr::kUnknownDeviceID,
        spirv::getDefaultResourceLimits(context));
    auto module = getOperation();
    module->setAttr(spirv::getTargetEnvAttrName(), attr);
  }
};

} // namespace imex

std::unique_ptr<mlir::Pass> imex::createSetSPIRVCapabilitiesPass() {
  return std::make_unique<SetSPIRVCapabilitiesPass>();
}
