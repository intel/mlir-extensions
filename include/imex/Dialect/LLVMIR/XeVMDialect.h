//===-- XeVMDialect.h - MLIR XeVM target definitions ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_XEVMDIALECT_H_
#define MLIR_DIALECT_LLVMIR_XEVMDIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include <imex/Dialect/LLVMIR/XeVMOpsEnums.h.inc>

namespace imex::xevm {

enum class XeVMAddrSpace : uint32_t {
  kPrivate = 0,  // OpenCL Workitem address space, SPIRV function
  kGlobal = 1,   // OpenCL Global memory, SPIRV crossworkgroup
  kConstant = 2, // OpenCL Constant memory, SPIRV uniform constant
  kShared = 3,   // OpenCL Local memory, SPIRV workgroup
  kGeneric = 4   // OpenCL Generic memory, SPIRV generic
};

} // namespace imex::xevm

#define GET_ATTRDEF_CLASSES
#include <imex/Dialect/LLVMIR/XeVMOpsAttributes.h.inc>

#define GET_OP_CLASSES
#include <imex/Dialect/LLVMIR/XeVMOps.h.inc>

#include <imex/Dialect/LLVMIR/XeVMOpsDialect.h.inc>

#endif /* MLIR_DIALECT_LLVMIR_XEVMDIALECT_H_ */
