// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

class OpaqueType : public ::mlir::Type::TypeBase<OpaqueType, ::mlir::Type,
                                                 ::mlir::TypeStorage> {
public:
  using Base::Base;

  static OpaqueType get(mlir::MLIRContext *context);
};

} // namespace util
} // namespace imex
