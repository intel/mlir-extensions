//===- RegionParallelLoopToGpu.h -------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Adds the conversion pattern from `scf.parallel` within `region.env_region`
/// to `gpu.launch`.
///
//===----------------------------------------------------------------------===//

#ifndef _RegionParallelLoopToGpu_H_INCLUDED_
#define _RegionParallelLoopToGpu_H_INCLUDED_

#include <mlir/IR/PatternMatch.h>

namespace mlir {
class Pass;
} // namespace mlir

namespace imex {
#define GEN_PASS_DECL_CONVERTREGIONPARALLELLOOPTOGPU
#include "imex/Conversion/Passes.h.inc"

/// Create a pass to convert the Region dialect to the GPU dialect.
std::unique_ptr<::mlir::Pass> createConvertRegionParallelLoopToGpuPass();

} // namespace imex

#endif // _RegionParallelLoopToGpu_H_INCLUDED_
