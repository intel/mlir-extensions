//===-- EWOp.h - NDArray pass details ---------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header file defines utilities for ew op canonicalization.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Utils/PassUtils.h>

/// @brief replace operand of op with defining cast op
/// @return true if replacement succeeded, false otherwise
template <typename OpType>
bool replaceOperandInplaceWithCast(::mlir::PatternRewriter &rewriter,
                                   unsigned idx, ::mlir::Value arg, OpType op) {
  auto ptTyp = mlir::dyn_cast<::imex::ndarray::NDArrayType>(arg.getType());
  if (ptTyp && !ptTyp.hasStaticShape()) {
    auto defOp = arg.getDefiningOp<::imex::ndarray::CastOp>();
    if (defOp) {
      auto src = defOp.getSource();
      auto srcPtTyp = mlir::cast<::imex::ndarray::NDArrayType>(src.getType());
      if (srcPtTyp.hasStaticShape()) {
        rewriter.modifyOpInPlace(op, [&]() { op->setOperand(idx, src); });
        return true;
      }
    }
  }
  return false;
};
