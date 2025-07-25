//===- Utils.h - Utils for NDArray dialect  ---------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the utils for the ndarray dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _NDArray_UTILS_H_INCLUDED_
#define _NDArray_UTILS_H_INCLUDED_

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Dialect/Region/IR/RegionOps.h>
#include <imex/Utils/PassUtils.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>

namespace imex {

// *******************************
// ***** Some helper functions ***
// *******************************

namespace ndarray {

inline ::mlir::Type toMLIR(::mlir::OpBuilder &b, DType dt) {
  switch (dt) {
  case F64:
    return b.getF64Type();
  case F32:
    return b.getF32Type();
  case I64:
    return b.getI64Type();
  case U64:
    return b.getI64Type();
  case I32:
    return b.getI32Type();
  case U32:
    return b.getI32Type();
  case I16:
    return b.getI16Type();
  case U16:
    return b.getI16Type();
  case I8:
    return b.getI8Type();
  case U8:
    return b.getI8Type();
  case I1:
    return b.getI1Type();
  default:
    assert(false && "Cannot handle unknown DType");
  };
  return {};
}

inline DType fromMLIR(const ::mlir::Type &typ) {
  if (typ.isF64())
    return F64;
  else if (typ.isF32())
    return F32;
  else if (typ.isIndex())
    return I64;
  else if (typ.isIntOrIndexOrFloat()) {
    auto w = typ.getIntOrFloatBitWidth();
    auto u = typ.isUnsignedInteger();
    switch (w) {
    case 64:
      return u ? U64 : I64;
    case 32:
      return u ? U32 : I32;
    case 16:
      return u ? U16 : I16;
    case 8:
      return u ? U8 : I8;
    case 1:
      return I1;
    };
  }
  assert(false && "Type not supported by NDArray");
}

inline ::mlir::Value createDType(::mlir::Location &loc,
                                 ::mlir::OpBuilder &builder, ::mlir::Type mt) {
  return createInt(loc, builder,
                   static_cast<int>(::imex::ndarray::fromMLIR(mt)),
                   sizeof(int) * 8);
}

inline ::mlir::Value createDType(::mlir::Location &loc,
                                 ::mlir::OpBuilder &builder,
                                 ::mlir::MemRefType mrt) {
  return createDType(loc, builder, mrt.getElementType());
}

template <typename T = ::imex::ValVec>
auto createShapeOf(::mlir::Location loc, ::mlir::OpBuilder &builder,
                   ::mlir::Value lPTnsr) {
  auto arType = mlir::dyn_cast<::mlir::RankedTensorType>(lPTnsr.getType());
  assert(arType);
  auto rank = arType.getRank();
  T dims;

  for (int64_t i = 0; i < rank; ++i) {
    dims.emplace_back(
        builder.createOrFold<::mlir::tensor::DimOp>(loc, lPTnsr, i));
  }

  return dims;
}

// convert an unranked memref from a NDArray
inline ::mlir::Value mkURMemRef(::mlir::Location loc,
                                ::mlir::OpBuilder &builder, ::mlir::Value src) {
  auto srcArType = mlir::cast<::mlir::RankedTensorType>(src.getType());
  auto bMRTyp = getMemRefType(srcArType);
  auto bMRef = createToMemRef(loc, builder, src, bMRTyp);
  return createUnrankedMemRefCast(builder, loc, bMRef);
}

} // namespace ndarray
} // namespace imex

#endif //  _NDArray_UTILS_H_INCLUDED_
