//===- ArithOpConversion.h - XeTileToXeGPU conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the ArithOpConversionPattern, used in XeTileToXeGPU
/// conversion, converting the Arith Ops.
///
//===----------------------------------------------------------------------===//
#ifndef _ArithOpConversion_H_INCLUDED_
#define _ArithOpConversion_H_INCLUDED_

#include "imex/Conversion/XeTileToXeGPU/XeTileToXeGPU.h"
#include "imex/Conversion/XeTileToXeGPU/XeTileToXeGPUConversion.h"

namespace imex {
bool isLegalArithOp(mlir::Operation *op);

void populateArithOpConversionPatterns(imex::XeOneToNTypeConverter &converter,
                                       mlir::RewritePatternSet &patterns,
                                       TileUsageAnalysis &analysis);

} // namespace imex
#endif
