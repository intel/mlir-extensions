// Copyright 2022 Intel Corporation
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

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Region.h>
#include <mlir/Interfaces/CopyOpInterface.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

namespace mlir {

/// Return the list of Range (i.e. offset, size, stride). Each Range
/// entry contains either the dynamic value or a ConstantIndexOp constructed
/// with `b` at location `loc`.
SmallVector<Range, 8> getOrCreateRanges(OffsetSizeAndStrideOpInterface op,
                                        OpBuilder &b, Location loc);

} // namespace mlir

namespace imex {
namespace ntensor {
class NTensorType;
class SliceType;
} // namespace ntensor
} // namespace imex

#include "imex/Dialect/ntensor/IR/NTensorOpsDialect.h.inc"
#include "imex/Dialect/ntensor/IR/NTensorOpsEnums.h.inc"

namespace imex {
namespace ntensor {
class NTensorBase : public mlir::Type,
                    public mlir::ShapedType::Trait<NTensorBase> {
public:
  using Type::Type;

  /// Returns the element type of this tensor type.
  mlir::Type getElementType() const;

  /// Returns if this type is ranked, i.e. it has a known number of dimensions.
  bool hasRank() const;

  /// Returns the shape of this tensor type.
  llvm::ArrayRef<int64_t> getShape() const;

  /// Clone this type with the given shape and element type. If the
  /// provided shape is `None`, the current shape of the type is used.
  NTensorBase cloneWith(llvm::Optional<llvm::ArrayRef<int64_t>> shape,
                        mlir::Type elementType) const;

  /// Return true if the specified element type is ok in a tensor.
  static bool isValidElementType(Type type);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Allow implicit conversion to ShapedType.
  operator mlir::ShapedType() const { return cast<mlir::ShapedType>(); }
};

} // namespace ntensor
} // namespace imex

#define GET_TYPEDEF_CLASSES
#include "imex/Dialect/ntensor/IR/NTensorOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "imex/Dialect/ntensor/IR/NTensorOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "imex/Dialect/ntensor/IR/NTensorOps.h.inc"
