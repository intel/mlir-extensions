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

#include "mlir-extensions/Transforms/const_utils.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>

mlir::Attribute plier::getConstVal(mlir::Operation *op) {
  assert(op);
  if (!op->hasTrait<mlir::OpTrait::ConstantLike>()) {
    return {};
  }

  return op->getAttr("value");
}

mlir::Attribute plier::getConstVal(mlir::Value op) {
  assert(op);
  if (auto parent_op = op.getDefiningOp()) {
    return getConstVal(parent_op);
  }
  return {};
}

mlir::Attribute plier::getConstAttr(mlir::Type type, double val) {
  assert(type);
  if (type.isa<mlir::FloatType>()) {
    return mlir::FloatAttr::get(type, val);
  }
  if (type.isa<mlir::IntegerType, mlir::IndexType>()) {
    return mlir::IntegerAttr::get(type, static_cast<int64_t>(val));
  }
  return {};
}

int64_t plier::getIntAttrValue(mlir::IntegerAttr attr) {
  assert(attr);
  auto attrType = attr.getType();
  if (attrType.isa<mlir::IndexType>()) {
    return attr.getInt();
  }
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
