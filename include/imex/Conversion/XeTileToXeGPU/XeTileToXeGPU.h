//===- XeTileToXeGPU.h - XeTileToXeGPU conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the XeTileToXeGPU conversion, converting the XeTile
/// dialect to the XeGPU dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _XeTileToXeGPU_H_INCLUDED_
#define _XeTileToXeGPU_H_INCLUDED_

#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include "XeTileToXeGPUConversion.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
namespace gpu {
class GPUModuleOp;
}
template <typename T> class OperationPass;
class RewritePatternSet;
} // namespace mlir

namespace imex {
#define GEN_PASS_DECL_CONVERTXETILETOXEGPU
#include "imex/Conversion/Passes.h.inc"

class XeOneToNTypeConverter;

/// Populate the given list with patterns rewrite XeTile Ops
void populateXeTileToXeGPUConversionPatterns(XeOneToNTypeConverter &converter,
                                             mlir::RewritePatternSet &patterns,
                                             imex::TileUsageAnalysis &analysis);

/// Create a pass to convert the XeTile dialect to the XeGPU dialect.
std::unique_ptr<mlir::OperationPass<mlir::gpu::GPUModuleOp>>
createConvertXeTileToXeGPUPass(const std::string &device = "pvc");

} // namespace imex

#endif // _XeTileToXeGPU_H_INCLUDED_
