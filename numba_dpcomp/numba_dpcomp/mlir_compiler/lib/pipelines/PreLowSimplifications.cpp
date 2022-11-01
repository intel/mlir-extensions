// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "pipelines/PreLowSimplifications.hpp"

#include "pipelines/BasePipeline.hpp"

#include "imex/Compiler/PipelineRegistry.hpp"
#include "imex/Transforms/ExpandTuple.hpp"
#include "imex/Transforms/MakeSignless.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

static void populateUntuplePipeline(mlir::OpPassManager &pm) {
  pm.addPass(imex::createExpandTuplePass());
  pm.addPass(mlir::createCanonicalizerPass());
}

static void populateRemoveSignPipeline(mlir::OpPassManager &pm) {
  pm.addPass(imex::createMakeSignlessPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void registerPreLowSimpleficationsPipeline(imex::PipelineRegistry &registry) {
  registry.registerPipeline([](auto sink) {
    auto stage = getHighLoweringStage();
    sink(untuplePipelineName(), {stage.begin}, {stage.end}, {},
         &populateUntuplePipeline);
  });
  registry.registerPipeline([](auto sink) {
    auto stage = getHighLoweringStage();
    sink(removeSignPipelineName(), {untuplePipelineName(), stage.begin},
         {stage.end}, {}, &populateRemoveSignPipeline);
  });
}

llvm::StringRef untuplePipelineName() { return "pre_low_untuple"; }

llvm::StringRef removeSignPipelineName() { return "pre_low_remove_sign"; }
