// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>

#include <functional>
#include <vector>

namespace mlir {
class OpPassManager;
}

namespace imex {
class PipelineRegistry {
public:
  PipelineRegistry() = default;
  PipelineRegistry(const PipelineRegistry &) = delete;

  using pipeline_funt_t = void (*)(mlir::OpPassManager &);
  using registry_entry_sink_t =
      void(llvm::StringRef pipelineName,
           llvm::ArrayRef<llvm::StringRef> prevPipelines,
           llvm::ArrayRef<llvm::StringRef> nextPipelines,
           llvm::ArrayRef<llvm::StringRef> jumps, pipeline_funt_t func);
  using registry_entry_t =
      std::function<void(llvm::function_ref<registry_entry_sink_t>)>;

  void registerPipeline(registry_entry_t func);

  using fill_stage_sink_t = llvm::function_ref<void(
      llvm::StringRef name, llvm::ArrayRef<llvm::StringRef> jumps,
      llvm::function_ref<void(mlir::OpPassManager &)>)>;
  using populate_pass_manager_sink_t =
      llvm::function_ref<void(fill_stage_sink_t)>;
  using populate_pass_manager_t =
      llvm::function_ref<void(populate_pass_manager_sink_t)>;
  void populatePassManager(populate_pass_manager_t resultSink) const;

private:
  std::vector<registry_entry_t> pipelines;
};
} // namespace imex
