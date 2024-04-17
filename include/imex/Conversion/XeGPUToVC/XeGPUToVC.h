//===- XeGPUToVC.h - Conversion---------------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements conversion of the XeGPU dialect operations into Func
/// dialect calls to vc-intrinsics functions
///
//===----------------------------------------------------------------------===//
#ifndef IMEX_CONVERSION_XEGPUTOVC_H
#define IMEX_CONVERSION_XEGPUTOVC_H
#include <mlir/Dialect/Vector/IR/VectorOps.h>

#include "imex/Dialect/XeGPU/IR/XeGPU.h"
#include "imex/Dialect/XeTile/IR/XeTileOps.h"
#include "imex/Utils/XeCommon.h"

namespace mlir {

class ConversionTarget;
class LLVMTypeConverter;
class Pass;
class Operation;
class RewritePatternSet;
template <typename T> class OperationPass;

namespace gpu {
class GPUModuleOp;
} // namespace gpu

} // namespace mlir

namespace imex {

std::unique_ptr<::mlir::OperationPass<::mlir::gpu::GPUModuleOp>>
createConvertXeGPUToVCPass();

} // namespace imex
#endif
