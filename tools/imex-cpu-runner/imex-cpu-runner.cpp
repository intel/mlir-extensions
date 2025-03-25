//===- imex-cpu-runner.cpp ---------------------------------------------*- C++
//-*-===//
//
// Copyright 2022 Intel Corporation
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the IMEX optimizer driver.
//
//===----------------------------------------------------------------------===//

//===- imex-cpu-runner.cpp - MLIR CPU Execution Driver---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry point to a command line utility that executes an MLIR file on the
// CPU by  translating MLIR to LLVM IR before JIT-compiling and executing the
// latter.
//
//===----------------------------------------------------------------------===//

// This file is copied from upstream mlir-cpu-runner
// https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-cpu-runner/mlir-cpu-runner.cpp

#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include "imex/Target/LLVMIR/Dialect/XeVM/XeVMToLLVMIRTranslation.h"

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  imex::xevm::registerXeVMDialectTranslation(registry);

  return mlir::JitRunnerMain(argc, argv, registry);
}
