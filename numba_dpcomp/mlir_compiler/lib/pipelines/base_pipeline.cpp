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

#include "pipelines/base_pipeline.hpp"

#include "plier/compiler/pipeline_registry.hpp"

namespace {
const constexpr llvm::StringRef passes[] = {
    "init",
    "lowering",
    "terminate",
};

void dummy_pass_func(mlir::OpPassManager &) {}
} // namespace

void registerBasePipeline(plier::PipelineRegistry &registry) {
  for (std::size_t i = 0; i < llvm::array_lengthof(passes); ++i) {
    registry.registerPipeline([i](auto sink) {
      if (0 == i) {
        sink(passes[i], {}, {}, {}, dummy_pass_func);
      } else {
        sink(passes[i], {passes[i - 1]}, {}, {}, dummy_pass_func);
      }
    });
  }
}

PipelineStage getHighLoweringStage() { return {passes[0], passes[1]}; }

PipelineStage getLowerLoweringStage() { return {passes[1], passes[2]}; }
