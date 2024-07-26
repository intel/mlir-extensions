//===- VectorToXeGPU.h - VectorToXeGPU conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the VectorToXeGPU conversion, converting the Vector
/// dialect to the XeGPU dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _VectorToXeGPU_H_INCLUDED_
#define _VectorToXeGPU_H_INCLUDED_

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;
}

namespace imex {
/// Create a pass to convert the Vector dialect to the XeGPU dialect.
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>> createConvertVectorToXeGPUPass();

} // namespace imex

#endif // _VectorToXeGPU_H_INCLUDED_
