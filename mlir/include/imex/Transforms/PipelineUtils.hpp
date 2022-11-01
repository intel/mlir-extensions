// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace mlir {
class ArrayAttr;
class ModuleOp;
class StringAttr;
} // namespace mlir

namespace imex {
mlir::ArrayAttr getPipelineJumpMarkers(mlir::ModuleOp module);
void addPipelineJumpMarker(mlir::ModuleOp module, mlir::StringAttr name);
void removePipelineJumpMarker(mlir::ModuleOp module, mlir::StringAttr name);
} // namespace imex
