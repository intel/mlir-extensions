//===- XeTileOpConversion.h - XeTileToXeGPU conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines ConversionPatterns for XeTileOps, used in XeTileToXeGPU
/// conversion, converting the XeTile dialect to the XeGPU dialect.
///
//===----------------------------------------------------------------------===//
#ifndef _XeTileOpConversion_H_INCLUDED_
#define _XeTileOpConversion_H_INCLUDED_

#include "imex/Conversion/XeTileToXeGPU/XeTileToXeGPUConversion.h"

namespace imex {

void populateXeTileOpConversionPatterns(imex::XeGPUTypeConverter &converter,
                                        mlir::RewritePatternSet &patterns);

} // namespace imex

#endif
