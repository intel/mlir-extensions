//===---------------------- XeTileOps.h - XeTile dialect  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the XeTile dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _XETILE_OPS_H_INCLUDED_
#define _XETILE_OPS_H_INCLUDED_
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/ShapedOpInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <numeric>

namespace imex {
namespace xetile {

class TileType;

} // namespace xetile
} // namespace imex

namespace imex {
namespace xetile {

// TODO : TileBase is  similar to that of XeGPU dialect, can we use a common
// TileBase and derive from that?
class TileBase : public mlir::Type, public mlir::ShapedType::Trait<TileBase> {
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
  TileBase cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
                     mlir::Type elementType, mlir::Attribute encoding) const;
};

} // namespace xetile
} // namespace imex

#include <imex/Dialect/XeTile/IR/XeTileOpsDialect.h.inc>
#include <imex/Dialect/XeTile/IR/XeTileOpsEnums.h.inc>
#define GET_ATTRDEF_CLASSES
#include <imex/Dialect/XeTile/IR/XeTileOpsAttrs.h.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/XeTile/IR/XeTileOpsTypes.h.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/XeTile/IR/XeTileOps.h.inc>

#endif // _XETILE_OPS_H_INCLUDED_
