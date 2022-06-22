//===- imex-opt.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <imex/InitIMEXDialects.h>
#include <imex/InitIMEXPasses.h>

int main(int argc, char **argv) {
  ::mlir::registerAllPasses();
  ::imex::registerAllPasses();
  // TODO: Register imex passes here.

  ::mlir::DialectRegistry registry;
  ::mlir::registerAllDialects(registry);
  ::imex::registerAllDialects(registry);

  return ::mlir::asMainReturnCode(
      ::mlir::MlirOptMain(argc, argv, "Imex optimizer driver\n", registry));
}
