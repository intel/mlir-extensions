//===- NDArrayOps.h - NDArray dialect  -------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the NDArray dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#ifndef _NDArray_OPS_H_INCLUDED_
#define _NDArray_OPS_H_INCLUDED_

#include <imex/Dialect/Region/IR/RegionOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/ShapedOpInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

#include "NDArrayDefs.h"

namespace imex {
namespace ndarray {

class NDArrayBase : public mlir::Type,
                    public mlir::ShapedType::Trait<NDArrayBase> {
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
  NDArrayBase cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
                        mlir::Type elementType) const;

  /// Clone this type with the given environment.
  NDArrayBase cloneWithEnv(::mlir::Attribute) const;

  /// Return true if the specified element type is ok in a tensor.
  static bool isValidElementType(Type type);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Allow implicit conversion to ShapedType.
  operator mlir::ShapedType() const {
    return mlir::cast<mlir::ShapedType>(*this);
  }
};

} // namespace ndarray
} // namespace imex

#include <imex/Dialect/NDArray/IR/NDArrayOpsDialect.h.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/NDArray/IR/NDArrayOpsTypes.h.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/NDArray/IR/NDArrayOps.h.inc>

#include <imex/Dialect/NDArray/Utils/Utils.h>

namespace imex {
namespace ndarray {

/// @return true if given NDArrayTYpe has this specific environment attribute
template <typename T> bool hasEnv(const ::imex::ndarray::NDArrayType &t) {
  for (auto a : t.getEnvironments()) {
    if (::mlir::isa<T>(a)) {
      return true;
    }
  }
  return false;
}

inline bool hasGPUEnv(const ::mlir::Type &t) {
  auto ptType = mlir::dyn_cast<::imex::ndarray::NDArrayType>(t);
  return ptType ? ::imex::ndarray::hasEnv<::imex::region::GPUEnvAttr>(ptType)
                : false;
}

inline ::imex::region::GPUEnvAttr getGPUEnv(const ::mlir::Type &t) {
  auto ptType = mlir::dyn_cast<::imex::ndarray::NDArrayType>(t);
  if (ptType) {
    for (auto a : ptType.getEnvironments()) {
      if (auto g = ::mlir::dyn_cast<::imex::region::GPUEnvAttr>(a)) {
        return g;
      }
    }
  }
  return {};
}

// Determine whether CastOp casts to a nore dynamic version of the source tensor
bool canFoldIntoConsumerOp(CastOp castOp);
bool canFoldIntoConsumerOp(::mlir::tensor::CastOp castOp);

/// Performs folding of any operand of `op` if it comes from a ndarray::CastOp
/// that can be folded.
mlir::LogicalResult foldArrayCast(mlir::Operation *op);

/// @return true if shape is known to span exactly one element
bool isUnitShape(const llvm::ArrayRef<int64_t> shp);

} // namespace ndarray
} // namespace imex

#endif // _NDArray_OPS_H_INCLUDED_
