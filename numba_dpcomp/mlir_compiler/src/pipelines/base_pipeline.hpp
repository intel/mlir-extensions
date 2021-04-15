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

#pragma once

#include <llvm/ADT/StringRef.h>

namespace plier
{
class PipelineRegistry;
}

void register_base_pipeline(plier::PipelineRegistry& registry);

struct PipelineStage
{
    llvm::StringRef begin;
    llvm::StringRef end;
};

PipelineStage get_high_lowering_stage(); // TODO: better name
PipelineStage get_lower_lowering_stage(); // TODO: better name
