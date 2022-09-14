//===- imex-opt.cpp ---------------------------------------------*- C++ -*-===//
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
#include <imex/Transforms/Passes.h>

int main(int argc, char **argv) {
  ::mlir::registerAllPasses();
  ::imex::registerAllPasses();

  ::mlir::DialectRegistry registry;
  ::mlir::registerAllDialects(registry);
  ::imex::registerAllDialects(registry);

  return ::mlir::asMainReturnCode(
      ::mlir::MlirOptMain(argc, argv, "Imex optimizer driver\n", registry));
}

static mlir::PassPipelineRegistration<> InsertGpuAlloc(
    "insert-gpu-alloc", "Converts memref alloc to gpu alloc",
    [](mlir::OpPassManager &pm) {
      pm.addNestedPass<mlir::func::FuncOp>(imex::createInsertGPUAllocsPass());
    });

static mlir::PassPipelineRegistration<>
    SetSPIRVCapabilities("set-spirv-capablilities", "Sets Spirv capabilities",
                         [](mlir::OpPassManager &pm) {
                           pm.addPass(imex::createSetSPIRVCapabilitiesPass());
                         });

static mlir::PassPipelineRegistration<>
    SetSPIRVAbiAttribute("set-spirv-abi-attrs", "Create AbiAttrs Pass",
                         [](mlir::OpPassManager &pm) {
                           pm.addNestedPass<mlir::gpu::GPUModuleOp>(
                               imex::createSetSPIRVAbiAttribute());
                         });
