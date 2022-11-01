// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "pipelines/PlierToScf.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "imex/Compiler/PipelineRegistry.hpp"
#include "imex/Conversion/CfgToScf.hpp"
#include "imex/Dialect/plier/Dialect.hpp"
#include "imex/Transforms/ArgLowering.hpp"
#include "imex/Transforms/RewriteWrapper.hpp"

#include "BasePipeline.hpp"

namespace {

/// Convert plier::ArgOp into direct function argument access. ArgOp is just an
/// artifact of Numba IR conversion and doesn't really have any functional
/// meaning so we can get rid of it early.
struct LowerArgOps
    : public imex::RewriteWrapperPass<
          LowerArgOps, void, imex::DependentDialectsList<plier::PlierDialect>,
          imex::ArgOpLowering> {};

static void populatePlierToScfPipeline(mlir::OpPassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<LowerArgOps>());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(imex::createCFGToSCFPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
}
} // namespace

void registerPlierToScfPipeline(imex::PipelineRegistry &registry) {
  registry.registerPipeline([](auto sink) {
    auto stage = getHighLoweringStage();
    sink(plierToScfPipelineName(), {stage.begin}, {stage.end}, {},
         &populatePlierToScfPipeline);
  });
}

llvm::StringRef plierToScfPipelineName() { return "plier_to_scf"; }
