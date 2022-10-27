//===- GPUXToLLVMPass.h - GPUXToLLVM conversion  ---------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the GPUXToLLVM conversion,
///
//===----------------------------------------------------------------------===//

#ifndef IMEX_GPUXTOLLVM_PASS_H_
#define IMEX_GPUXTOLLVM_PASS_H_

#include <memory>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
namespace mlir {
class ConversionTarget;
class LLVMTypeConverter;
class Location;
struct LogicalResult;
class Pass;
class Operation;
class RewritePatternSet;
class ModuleOp;
template <typename T> class OperationPass;

} // namespace mlir

namespace imex {

// std::unique_ptr<mlir::Pass> createEnumerateEventsPass();

/// Populates the given list with patterns that convert from GPUX dialect to
/// LLVM.
void populateGpuxToLLVMPatternsAndLegality(mlir::LLVMTypeConverter &converter,
                                           mlir::RewritePatternSet &patterns,
                                           mlir::ConversionTarget &target);
/// Creates a pass to convert a GPU operations into a sequence of GPU runtime
/// calls.
///
/// This pass does not generate code to call GPU runtime APIs directly but
/// instead uses a small wrapper library (currently Level-zero wrapper) that
/// exports a stable and conveniently typed ABI on top of GPU runtimes
/// such as Level-Zero or SYCL
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertGPUXToLLVMPass();

} // namespace imex

#endif // IMEX_GPUXTOLLVM_PASS_H_
