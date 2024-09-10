//===- GPUToGPUX.h - GPUToGPUX conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the GPUToGPUX conversion, converting the GPU
/// dialect to the GPUX dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _GPUToGPUX_H_INCLUDED_
#define _GPUToGPUX_H_INCLUDED_

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
class LLVMTypeConverter;
class MLIRContext;
template <typename T> class OperationPass;
} // namespace mlir

namespace imex {
#define GEN_PASS_DECL_CONVERTGPUTOGPUX
#include "imex/Conversion/Passes.h.inc"

/// Create a pass to convert the GPU dialect to the GPUX dialect.
std::unique_ptr<::mlir::OperationPass<void>> createConvertGPUToGPUXPass();

} // namespace imex

#endif // _GPUToGPUX_H_INCLUDED_
