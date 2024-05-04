
//===- RegionUtils.h - Region dialect  --------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides utils to deal with Region Dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _Region_UTILS_H_INCLUDED_
#define _Region_UTILS_H_INCLUDED_

#include <imex/Dialect/Region/IR/RegionOps.h>

namespace imex {
namespace region {

inline bool isGpuRegion(::imex::region::EnvironmentRegionOp op) {
  return op && mlir::isa<::imex::region::GPUEnvAttr>(op.getEnvironment());
}

inline bool isInGpuRegion(::mlir::Operation *op) {
  if (!op)
    return false;
  auto p = op->getParentOfType<::imex::region::EnvironmentRegionOp>();
  return p && isGpuRegion(p);
}

} // namespace region
} // namespace imex

#endif // _Region_UTILS_H_INCLUDED_
