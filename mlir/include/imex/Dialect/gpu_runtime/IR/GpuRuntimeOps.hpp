// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

#include <mlir/Dialect/GPU/IR/GPUDialect.h>

#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOpsDialect.h.inc"

#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOps.h.inc"

namespace gpu_runtime {
mlir::StringRef getGpuAccessibleAttrName();
}
