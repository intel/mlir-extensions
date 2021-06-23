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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>

#include <functional>
#include <vector>

namespace mlir {
class OpPassManager;
}

namespace plier {
class PipelineRegistry {
public:
  PipelineRegistry() = default;
  PipelineRegistry(const PipelineRegistry &) = delete;

  using pipeline_funt_t = void (*)(mlir::OpPassManager &);
  using registry_entry_sink_t =
      void(llvm::StringRef pipeline_name,
           llvm::ArrayRef<llvm::StringRef> prev_pipelines,
           llvm::ArrayRef<llvm::StringRef> next_pipelines,
           llvm::ArrayRef<llvm::StringRef> jumps, pipeline_funt_t func);
  using registry_entry_t =
      std::function<void(llvm::function_ref<registry_entry_sink_t>)>;

  void register_pipeline(registry_entry_t func);

  using fill_stage_sink_t = llvm::function_ref<void(
      llvm::StringRef name, llvm::ArrayRef<llvm::StringRef> jumps,
      llvm::function_ref<void(mlir::OpPassManager &)>)>;
  using populate_pass_manager_sink_t =
      llvm::function_ref<void(fill_stage_sink_t)>;
  using populate_pass_manager_t =
      llvm::function_ref<void(populate_pass_manager_sink_t)>;
  void populate_pass_manager(populate_pass_manager_t result_sink) const;

private:
  std::vector<registry_entry_t> pipelines;
};
} // namespace plier
