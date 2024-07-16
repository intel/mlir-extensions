//===-- Passes.h - XeTile pass declaration file --------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header file defines prototypes that expose pass constructors for the
/// XeTile dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _XeTile_PASSES_H_INCLUDED_
#define _XeTile_PASSES_H_INCLUDED_

#include "imex/Utils/XeArch.h"
#include <mlir/Pass/Pass.h>

namespace mlir {
class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T> class OperationPass;
class RewritePatternSet;
} // namespace mlir

namespace imex {

class XeTypeConverter;

//===----------------------------------------------------------------------===//
/// XeTile passes.
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createXeTileInitDuplicatePass();

std::unique_ptr<mlir::Pass>
createXeTileBlockingPass(const std::string &device = "pvc");
std::unique_ptr<mlir::Pass> createXeTileBlockAligningPass();
std::unique_ptr<mlir::Pass> createXeTileWgToSgPass();
std::unique_ptr<mlir::Pass> createXeTileOptimizeTransposePass();

///
void populateXeTileInitDuplicatePatterns(imex::XeTypeConverter &converter,
                                         mlir::RewritePatternSet &patterns);

///
void populateXeTileBlockingPatterns(imex::XeTypeConverter &converter,
                                    mlir::RewritePatternSet &patterns,
                                    std::shared_ptr<XeuArchInterface> ptruArch);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include <imex/Dialect/XeTile/Transforms/Passes.h.inc>

} // namespace imex

#endif // _XeTile_PASSES_H_INCLUDED_
