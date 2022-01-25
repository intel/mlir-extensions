// Copyright 2022 Intel Corporation
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

#pragma once

#include "../../numba_dpcomp/numba_dpcomp/mlir_compiler/lib/pipelines/base_pipeline.hpp"
#include "../../numba_dpcomp/numba_dpcomp/mlir_compiler/lib/pipelines/lower_to_llvm.hpp"

namespace plier {
class PipelineRegistry;
}

namespace llvm {
class StringRef;
}

void registerLowerToGPURuntimePipeline(plier::PipelineRegistry &registry);

llvm::StringRef lowerToGPURuntimePipelineNameHigh();
llvm::StringRef lowerToGPURuntimePipelineNameLow();
