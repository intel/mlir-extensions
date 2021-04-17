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

#include <memory>

namespace mlir
{
class MLIRContext;
class ModuleOp;
}

namespace plier
{
class PipelineRegistry;

class CompilerContext
{
public:
    struct Settings
    {
        bool verify = false;
        bool pass_statistics = false;
        bool pass_timings = false;
        bool ir_printing = false;
    };

    class CompilerContextImpl;

    CompilerContext(mlir::MLIRContext& ctx, const Settings& settings,
                    const PipelineRegistry& registry);
    ~CompilerContext();

    CompilerContext(CompilerContext&&) = default;

    void run(mlir::ModuleOp module);

private:
    std::unique_ptr<CompilerContextImpl> impl;
};
}
