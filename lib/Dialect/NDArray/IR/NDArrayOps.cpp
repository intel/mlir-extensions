//===- NDArrayOps.cpp - NDArray dialect  -----------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the NDArray dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Utils/PassUtils.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/DialectImplementation.h>

namespace imex {
namespace ndarray {

void NDArrayDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/NDArray/IR/NDArrayOpsTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <imex/Dialect/NDArray/IR/NDArrayOps.cpp.inc>
      >();
}

} // namespace ndarray
} // namespace imex

namespace imex {
namespace ndarray {

NDArrayType NDArrayType::get(::mlir::MLIRContext *context,
                             ::llvm::ArrayRef<int64_t> shape,
                             ::mlir::Type elementType,
                             ::llvm::ArrayRef<::mlir::Attribute> environments,
                             ::mlir::StringAttr layout) {
  ::mlir::SmallVector<::mlir::Attribute> envs(environments);
  struct {
    bool operator()(::mlir::Attribute a_, ::mlir::Attribute b_) const {
      return ::mlir::hash_value(a_) < ::mlir::hash_value(b_);
    }
  } attrSort;
  std::sort(envs.begin(), envs.end(), attrSort);
  return Base::get(context, std::move(shape), std::move(elementType),
                   std::move(envs), std::move(layout));
}

NDArrayType NDArrayType::get(::llvm::ArrayRef<int64_t> shape,
                             ::mlir::Type elementType,
                             ::mlir::ArrayRef<::mlir::Attribute> environments,
                             std::optional<::llvm::StringRef> layout) {
  auto ctx = elementType.getContext();
  auto l =
      layout ? ::mlir::StringAttr::get(ctx, *layout) : ::mlir::StringAttr{};
  return get(ctx, shape, elementType, environments, l);
}

NDArrayType NDArrayType::get(::llvm::ArrayRef<int64_t> shape,
                             ::mlir::Type elementType,
                             ::mlir::ArrayRef<::mlir::Attribute> environments,
                             ::mlir::StringAttr layout) {
  auto ctx = elementType.getContext();
  return get(ctx, shape, elementType, environments, layout);
}

::mlir::MemRefType NDArrayType::getMemRefType() const {
  return ::imex::getMemRefType(getContext(), getShape(), getElementType());
}

::mlir::RankedTensorType NDArrayType::getTensorType() const {
  return ::imex::getTensorType(getContext(), getShape(), getElementType());
}

NDArrayType NDArrayType::cloneWithDynDims() const {
  if (hasZeroSize() || hasUnitSize()) {
    return cloneWith(getShape(), getElementType()).cast<NDArrayType>();
  }
  return NDArrayType::get(
      ::mlir::SmallVector<int64_t>(getRank(), ::mlir::ShapedType::kDynamic),
      getElementType(), getEnvironments(), getLayout());
}

} // namespace ndarray
} // namespace imex

bool imex::ndarray::NDArrayBase::hasRank() const { return true; }

llvm::ArrayRef<int64_t> imex::ndarray::NDArrayBase::getShape() const {
  return cast<NDArrayType>().getShape();
}

imex::ndarray::NDArrayBase imex::ndarray::NDArrayBase::cloneWith(
    std::optional<llvm::ArrayRef<int64_t>> shape, Type elementType) const {
  auto t = cast<NDArrayType>();
  return NDArrayType::get(shape.value_or(getShape()), elementType,
                          t.getEnvironments(), t.getLayout());
}

imex::ndarray::NDArrayBase
imex::ndarray::NDArrayBase::cloneWithEnv(::mlir::Attribute env) const {
  auto t = cast<NDArrayType>();
  ::mlir::SmallVector<::mlir::Attribute> envs(t.getEnvironments());
  envs.emplace_back(env);
  return NDArrayType::get(t.getShape(), t.getElementType(), envs,
                          t.getLayout());
};

bool imex::ndarray::NDArrayBase::isValidElementType(Type type) {
  return type.isIntOrIndexOrFloat();
}

bool imex::ndarray::isUnitShape(const llvm::ArrayRef<int64_t> shp) {
  for (auto d : shp) {
    if (d != 1)
      return false;
  }
  return true;
}

bool imex::ndarray::NDArrayType::hasUnitSize() const {
  return isUnitShape(getShape());
}

bool imex::ndarray::NDArrayType::hasZeroSize() const {
  for (auto d : getShape()) {
    if (d == 0)
      return true;
  }
  return false;
}

#include <imex/Dialect/NDArray/IR/NDArrayOpsDialect.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/NDArray/IR/NDArrayOpsTypes.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/NDArray/IR/NDArrayOps.cpp.inc>
