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
