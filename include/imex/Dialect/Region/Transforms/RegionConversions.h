//===-- RegionConversions.h - Region conversion declaration file -*- C++ -*-==//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header file defines prototypes that expose conversion type patterns
///     for region ops.
///
//===----------------------------------------------------------------------===//

#ifndef _Region_Conversions_H_INCLUDED_
#define _Region_Conversions_H_INCLUDED_

namespace mlir {
class TypeConverter;
class RewritePatternSet;
} // namespace mlir

namespace imex {

/// @brief add type conversion patters for op in region dialect
void populateRegionTypeConversionPatterns(::mlir::RewritePatternSet &patterns,
                                          ::mlir::TypeConverter &typeConverter);

} // namespace imex

#endif // _Region_Conversions_H_INCLUDED_
