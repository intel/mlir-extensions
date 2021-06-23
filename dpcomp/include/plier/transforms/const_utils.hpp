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

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Value.h>

namespace mlir {
class Operation;
class IntegerAttr;
} // namespace mlir

namespace plier {
mlir::Attribute getConstVal(mlir::Operation *op);
mlir::Attribute getConstVal(mlir::Value op);

template <typename T> T getConstVal(mlir::Operation *op) {
  return getConstVal(op).dyn_cast_or_null<T>();
}

template <typename T> T getConstVal(mlir::Value op) {
  return getConstVal(op).dyn_cast_or_null<T>();
}

mlir::Attribute getConstAttr(mlir::Type type, double val);

int64_t getIntAttrValue(mlir::IntegerAttr attr);
} // namespace plier
