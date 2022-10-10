//===- GPUToSPIRVPass.h - GPUToSPIRV conversion  ---------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the GPUToSPIRV conversion,
///
//===----------------------------------------------------------------------===//

#ifndef IMEX_GPUTOSPIRV_PASS_H_
#define IMEX_GPUTOSPIRV_PASS_H_

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
class Pass;
struct ScfToSPIRVContextImpl;
class ModuleOp;
template <typename T> class OperationPass;

} // namespace mlir

namespace imex {
/// Create a pass
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertGPUXToSPIRVPass(bool mapMemorySpace = false);

} // namespace imex

#endif // IMEX_GPUTOSPIRV_PASS_H_
