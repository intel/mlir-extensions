// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/ConstUtils.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>

mlir::Attribute imex::getConstVal(mlir::Operation *op) {
  assert(op);
  if (!op->hasTrait<mlir::OpTrait::ConstantLike>())
    return {};

  return op->getAttr("value");
}

mlir::Attribute imex::getConstVal(mlir::Value op) {
  assert(op);
  if (auto parent_op = op.getDefiningOp())
    return getConstVal(parent_op);

  return {};
}

mlir::Attribute imex::getConstAttr(mlir::Type type, double val) {
  assert(type);
  if (type.isa<mlir::FloatType>())
    return mlir::FloatAttr::get(type, val);

  if (type.isa<mlir::IntegerType, mlir::IndexType>())
    return mlir::IntegerAttr::get(type, static_cast<int64_t>(val));

  return {};
}

int64_t imex::getIntAttrValue(mlir::IntegerAttr attr) {
  assert(attr);
  auto attrType = attr.getType();
  if (attrType.isa<mlir::IndexType>())
    return attr.getInt();

  auto type = attrType.cast<mlir::IntegerType>();
  if (type.isSigned()) {
    return attr.getSInt();
  } else if (type.isUnsigned()) {
    return static_cast<int64_t>(attr.getUInt());
  } else {
    assert(type.isSignless());
    return attr.getInt();
  }
}
