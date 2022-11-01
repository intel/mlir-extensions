// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/StringRef.h>

namespace imex {
class PipelineRegistry;
}

void registerBasePipeline(imex::PipelineRegistry &registry);

struct PipelineStage {
  llvm::StringRef begin;
  llvm::StringRef end;
};

PipelineStage getHighLoweringStage();  // TODO: better name
PipelineStage getLowerLoweringStage(); // TODO: better name
