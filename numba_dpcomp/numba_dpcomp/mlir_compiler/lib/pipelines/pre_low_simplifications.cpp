// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pipelines/pre_low_simplifications.hpp"

#include "pipelines/base_pipeline.hpp"

#include "imex/Transforms/MakeSignless.hpp"
#include "imex/Transforms/expand_tuple.hpp"
#include "imex/compiler/pipeline_registry.hpp"

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
