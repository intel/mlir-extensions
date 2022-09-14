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

#include <imex/Transforms/Passes.h>

#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>

namespace imex {
struct SetSPIRVAbiAttribute
    : public mlir::PassWrapper<SetSPIRVAbiAttribute,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    auto gpuModule = getOperation();
    auto *context = &getContext();
    auto attrName =
        mlir::StringAttr::get(context, mlir::spirv::getEntryPointABIAttrName());
    auto abi = mlir::spirv::getEntryPointABIAttr(llvm::None, context);
    for (auto gpuFunc : gpuModule.getOps<mlir::gpu::GPUFuncOp>()) {
      if (!mlir::gpu::GPUDialect::isKernel(gpuFunc) ||
          gpuFunc->getAttr(attrName))
        continue;

      gpuFunc->setAttr(attrName, abi);
    }
  }
};

} // namespace imex

std::unique_ptr<mlir::Pass> imex::createSetSPIRVAbiAttribute() {
  return std::make_unique<SetSPIRVAbiAttribute>();
}
