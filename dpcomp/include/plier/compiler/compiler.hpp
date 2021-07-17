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

#include <functional>
#include <memory>
#include <string>

#include <llvm/ADT/Optional.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace plier {
class PipelineRegistry;

class CompilerContext {
public:
  struct Settings {
    struct IRPrintingSettings {
      llvm::SmallVector<std::string, 1> printBefore;
      llvm::SmallVector<std::string, 1> printAfter;
      llvm::raw_ostream *out;
    };

    bool verify = false;
    bool passStatistics = false;
    bool passTimings = false;
    bool irDumpStderr = false;

    llvm::Optional<IRPrintingSettings> irPrinting;
  };

  class CompilerContextImpl;

  CompilerContext(mlir::MLIRContext &ctx, const Settings &settings,
                  const PipelineRegistry &registry);
  ~CompilerContext();

  CompilerContext(CompilerContext &&) = default;

  void run(mlir::ModuleOp module);

private:
  std::unique_ptr<CompilerContextImpl> impl;
};
} // namespace plier
