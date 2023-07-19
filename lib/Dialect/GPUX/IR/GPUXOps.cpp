//===- GPUXOps.cpp - GPUX dialect -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the GPUX dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/GPUX/IR/GPUXOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/InliningUtils.h>

#include <optional>

namespace imex {
namespace gpux {

void GPUXDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/GPUX/IR/GPUXOpsTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <imex/Dialect/GPUX/IR/GPUXOps.cpp.inc>
      >();
}

void CreateStreamOp::build(::mlir::OpBuilder &odsBuilder,
                           ::mlir::OperationState &odsState,
                           std::optional<::mlir::Value> device,
                           std::optional<::mlir::Value> context) {
  CreateStreamOp::build(odsBuilder, odsState, odsBuilder.getType<StreamType>(),
                        device.value_or(mlir::Value{}),
                        context.value_or(mlir::Value{}));
}

void LaunchFuncOp::build(
    ::mlir::OpBuilder &builder, ::mlir::OperationState &result,
    ::mlir::Value stream, ::mlir::gpu::GPUFuncOp kernelFunc,
    ::mlir::gpu::KernelDim3 gridSize, ::mlir::gpu::KernelDim3 blockSize,
    ::mlir::Value dynamicSharedMemorySize, ::mlir::ValueRange kernelOperands,
    ::mlir::Type asyncTokenType, ::mlir::ValueRange asyncDependencies) {
  result.addOperands(asyncDependencies);
  result.addOperands(stream);
  if (asyncTokenType)
    result.types.push_back(builder.getType<mlir::gpu::AsyncTokenType>());

  // Add grid and block sizes as op operands, followed by the data operands.
  result.addOperands({gridSize.x, gridSize.y, gridSize.z, blockSize.x,
                      blockSize.y, blockSize.z});
  if (dynamicSharedMemorySize)
    result.addOperands(dynamicSharedMemorySize);
  result.addOperands(kernelOperands);
  auto kernelModule = kernelFunc->getParentOfType<mlir::gpu::GPUModuleOp>();
  auto kernelSymbol = mlir::SymbolRefAttr::get(
      kernelModule.getNameAttr(),
      {mlir::SymbolRefAttr::get(kernelFunc.getNameAttr())});
  // Verify this
  result.addAttribute(getKernelAttrName(result.name), kernelSymbol);
  mlir::SmallVector<int32_t, 10> segmentSizes(10, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes[segmentSizes.size() - 2] = dynamicSharedMemorySize ? 1 : 0;
  segmentSizes.back() = static_cast<int32_t>(kernelOperands.size());
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(segmentSizes));
}

mlir::StringAttr LaunchFuncOp::getKernelModuleName() {
  return getKernel().getRootReference();
}

mlir::StringAttr LaunchFuncOp::getKernelName() {
  return getKernel().getLeafReference();
}

} // namespace gpux
} // namespace imex

#include <imex/Dialect/GPUX/IR/GPUXOpsDialect.cpp.inc>

#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/GPUX/IR/GPUXOpsTypes.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/GPUX/IR/GPUXOps.cpp.inc>
