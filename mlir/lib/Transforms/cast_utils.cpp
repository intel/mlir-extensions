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

#include "mlir-extensions/Transforms/cast_utils.hpp"
#include "mlir-extensions/Dialect/plier_util/dialect.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>

namespace {
mlir::Type makeSignless(mlir::Type type) {
  if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    if (!intType.isSignless()) {
      return mlir::IntegerType::get(intType.getContext(), intType.getWidth());
    }
  }
  return type;
}
} // namespace

mlir::Value plier::indexCast(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value val, mlir::Type dstType) {
  auto srcType = val.getType();
  assert(srcType.isa<mlir::IndexType>() || dstType.isa<mlir::IndexType>());
  if (srcType == dstType)
    return val;

  auto newSrcType = makeSignless(srcType);
  if (newSrcType != srcType)
    val = builder.createOrFold<plier::SignCastOp>(loc, newSrcType, val);

  auto newDstType = makeSignless(dstType);
  val = builder.createOrFold<mlir::arith::IndexCastOp>(loc, newDstType, val);
  if (newDstType != dstType)
    val = builder.createOrFold<plier::SignCastOp>(loc, dstType, val);

  return val;
}

mlir::Value plier::indexCast(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value src) {
  return indexCast(builder, loc, src,
                   mlir::IndexType::get(builder.getContext()));
}

mlir::Type plier::makeSignlessType(mlir::Type type) {
  if (auto intType = type.dyn_cast<mlir::IntegerType>())
    return makeSignlessType(intType);

  return type;
}

mlir::IntegerType plier::makeSignlessType(mlir::IntegerType type) {
  if (!type.isSignless())
    return mlir::IntegerType::get(type.getContext(), type.getWidth());

  return type;
}
