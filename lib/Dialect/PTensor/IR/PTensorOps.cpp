//===- PTensor.cpp - PTensor dialect  --------------------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the PTensor dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/internal/PassUtils.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/DialectImplementation.h>

namespace imex {
namespace ptensor {

void PTensorDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/PTensor/IR/PTensorOpsTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <imex/Dialect/PTensor/IR/PTensorOps.cpp.inc>
      >();
}

} // namespace ptensor
} // namespace imex

#include <imex/Dialect/PTensor/IR/PTensorOpsDialect.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/PTensor/IR/PTensorOpsTypes.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/PTensor/IR/PTensorOps.cpp.inc>

#if 0
void imex::ptensor::ExtractSliceOp::build(
    ::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
    ::mlir::Type resultType, ::mlir::Value source,
    ::mlir::ArrayRef<::mlir::OpFoldResult> offsets,
    ::mlir::ArrayRef<::mlir::OpFoldResult> sizes,
    ::mlir::ArrayRef<::mlir::OpFoldResult> strides,
    ::mlir::ArrayRef<::mlir::NamedAttribute> attrs)
{
    ::mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
    ::mlir::SmallVector<::mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
    dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets,
                                ::mlir::ShapedType::kDynamicStrideOrOffset);
    dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes,
                                ::mlir::ShapedType::kDynamicSize);
    dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides,
                                ::mlir::ShapedType::kDynamicStrideOrOffset);
    build(odsBuilder, odsState, resultType, source, dynamicOffsets, dynamicSizes,
          dynamicStrides, odsBuilder.getI64ArrayAttr(staticOffsets),
          odsBuilder.getI64ArrayAttr(staticSizes), odsBuilder.getI64ArrayAttr(staticStrides));
    odsState.addAttributes(attrs);
}
#endif
