// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
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

#include "imex/Dialect/imex_util/ImexUtilOpsDialect.h.inc"
#include "imex/Dialect/imex_util/ImexUtilOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "imex/Dialect/imex_util/ImexUtilOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "imex/Dialect/imex_util/ImexUtilOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "imex/Dialect/imex_util/ImexUtilOps.h.inc"

namespace imex {
namespace util {
namespace attributes {
llvm::StringRef getFastmathName();
llvm::StringRef getJumpMarkersName();
llvm::StringRef getParallelName();
llvm::StringRef getMaxConcurrencyName();
llvm::StringRef getForceInlineName();
llvm::StringRef getOptLevelName();
} // namespace attributes
} // namespace util
} // namespace imex
