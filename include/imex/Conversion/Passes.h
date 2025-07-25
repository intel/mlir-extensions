//===- Passes.h - Conversion Pass Construction and Registration -------===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file includes all IMEX conversion pass declarations.
///
//===----------------------------------------------------------------------===//

#ifndef _IMEX_CONVERSION_PASSES_H_INCLUDED_
#define _IMEX_CONVERSION_PASSES_H_INCLUDED_

#include "mlir/Pass/Pass.h"

#include <imex/Conversion/ArithToVC/ArithToVC.h>
#include <imex/Conversion/DropRegions/DropRegions.h>
#include <imex/Conversion/GPUToGPUX/GPUToGPUX.h>
#include <imex/Conversion/GPUToSPIRV/GPUToSPIRVPass.h>
#include <imex/Conversion/GPUXToLLVM/GPUXToLLVMPass.h>
#include <imex/Conversion/MathToVC/MathToVC.h>
#include <imex/Conversion/NDArrayToLinalg/NDArrayToLinalg.h>
#include <imex/Conversion/RegionParallelLoopToGpu/RegionParallelLoopToGpu.h>
#include <imex/Conversion/XeGPUToVC/XeGPUToVC.h>
#include <imex/Conversion/XeGPUToXeVM/XeGPUToXeVM.h>
#include <imex/Conversion/XeTileToXeGPU/XeTileToXeGPU.h>

namespace imex {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include <imex/Conversion/Passes.h.inc>

} // namespace imex

#endif // _IMEX_CONVERSION_PASSES_H_INCLUDED_
