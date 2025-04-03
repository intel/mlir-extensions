//===-- Utils.h - MLIR XeVM target utils ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files declares XeVM target related utility classes and functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVM_XEVM_UTILS_H
#define MLIR_TARGET_LLVM_XEVM_UTILS_H

#include "imex/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Target/LLVM/ModuleToObject.h"

namespace imex {
namespace xevm {

/// Base class for all XeVM serializations from GPU modules into binary strings.
/// By default this class serializes into LLVM bitcode.
class SerializeGPUModuleBase : public mlir::LLVM::ModuleToObject {
public:
  SerializeGPUModuleBase(mlir::Operation &module, XeVMTargetAttr target,
                         const mlir::gpu::TargetOptions &targetOptions = {});

  static void init();
  XeVMTargetAttr getTarget() const;

protected:
  XeVMTargetAttr target;
};
} // namespace xevm
} // namespace imex

#endif // MLIR_TARGET_LLVM_XEVM_UTILS_H
