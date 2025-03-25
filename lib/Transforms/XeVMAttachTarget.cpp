//===-- XeVMAttachTarget.cpp - DESC -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `GpuXeVMAttachTarget` pass, attaching `#xevm.target`
// attributes to GPU modules.
//
//===----------------------------------------------------------------------===//

#include "imex/Dialect/LLVMIR/XeVMDialect.h"
#include "imex/Target/LLVM/XeVM/Target.h"
#include "imex/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Regex.h"

namespace imex {
#define GEN_PASS_DEF_GPUXEVMATTACHTARGET
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

using namespace mlir;
using namespace imex::xevm;

namespace {
struct XeVMAttachTarget
    : public imex::impl::GpuXeVMAttachTargetBase<XeVMAttachTarget> {
  using Base::Base;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<imex::xevm::XeVMDialect>();
  }
};
} // namespace

void XeVMAttachTarget::runOnOperation() {
  OpBuilder builder(&getContext());
  auto target =
      builder.getAttr<imex::xevm::XeVMTargetAttr>(optLevel, triple, chip);
  llvm::Regex matcher(moduleMatcher);
  // Check if the name of the module matches.
  auto gpuModule = cast<gpu::GPUModuleOp>(getOperation());
  if (!moduleMatcher.empty() && !matcher.match(gpuModule.getName()))
    return;
  // Create the target array.
  SmallVector<Attribute> targets;
  if (std::optional<ArrayAttr> attrs = gpuModule.getTargets())
    targets.append(attrs->getValue().begin(), attrs->getValue().end());
  targets.push_back(target);
  // Remove any duplicate targets.
  targets.erase(llvm::unique(targets), targets.end());
  // Update the target attribute array.
  gpuModule.setTargetsAttr(builder.getArrayAttr(targets));
}
