// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "pipelines/BasePipeline.hpp"

#include "imex/Compiler/PipelineRegistry.hpp"

namespace {
static const constexpr llvm::StringLiteral passes[] = {
    "init",
    "lowering",
    "terminate",
};

void dummyPassFunc(mlir::OpPassManager &) {}
} // namespace

void registerBasePipeline(imex::PipelineRegistry &registry) {
  for (std::size_t i = 0; i < std::size(passes); ++i) {
    registry.registerPipeline([i](auto sink) {
      if (0 == i) {
        sink(passes[i], {}, {}, {}, dummyPassFunc);
      } else {
        sink(passes[i], {passes[i - 1]}, {}, {}, dummyPassFunc);
      }
    });
  }
}

PipelineStage getHighLoweringStage() { return {passes[0], passes[1]}; }

PipelineStage getLowerLoweringStage() { return {passes[1], passes[2]}; }
