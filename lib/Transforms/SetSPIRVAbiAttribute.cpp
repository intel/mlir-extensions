//===- SetSPIRVAbiAttribute.cpp - SetSPIRVAbiAttribute Pass  -------*- C++
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
/// This file adds a kernel attribute called spv.entry_point_abi to the kernel
/// function.
///
//===----------------------------------------------------------------------===//

#include "imex/Transforms/Passes.h"

#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace imex {
#define GEN_PASS_DEF_SETSPIRVABIATTRIBUTE
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

namespace {
class SetSPIRVAbiAttributePass
    : public imex::impl::SetSPIRVAbiAttributeBase<SetSPIRVAbiAttributePass> {
public:
  explicit SetSPIRVAbiAttributePass() { m_clientAPI = "vulkan"; }
  explicit SetSPIRVAbiAttributePass(const mlir::StringRef &clientAPI)
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
    auto gpuModule = getOperation();
    auto *context = &getContext();
    auto attrName =
        mlir::StringAttr::get(context, mlir::spirv::getEntryPointABIAttrName());
    if (m_clientAPI == "opencl") {
      auto abi = mlir::spirv::getEntryPointABIAttr(llvm::None, context);
      for (auto gpuFunc : gpuModule.getOps<mlir::gpu::GPUFuncOp>()) {
        if (!mlir::gpu::GPUDialect::isKernel(gpuFunc) ||
            gpuFunc->getAttr(attrName))
          continue;

        gpuFunc->setAttr(attrName, abi);
      }
    } else if (m_clientAPI == "vulkan") {
      llvm::SmallVector<int32_t, 3> workgroupVec{1, 1, 1};
      auto abi = mlir::spirv::getEntryPointABIAttr(workgroupVec, context);
      for (auto gpuFunc : gpuModule.getOps<mlir::gpu::GPUFuncOp>()) {
        if (!mlir::gpu::GPUDialect::isKernel(gpuFunc) ||
            gpuFunc->getAttr(attrName))
          continue;

        gpuFunc->setAttr(attrName, abi);
      }
    }
  }

private:
  mlir::StringRef m_clientAPI;
};

} // namespace

namespace imex {
std::unique_ptr<mlir::Pass> createSetSPIRVAbiAttributePass() {
  return std::make_unique<SetSPIRVAbiAttributePass>();
}
} // namespace imex
