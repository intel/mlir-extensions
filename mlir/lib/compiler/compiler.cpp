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

#include "mlir-extensions/compiler/compiler.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <llvm/Support/raw_ostream.h>

#include <unordered_map>

#include "mlir-extensions/compiler/pipeline_registry.hpp"
#include "mlir-extensions/Transforms/pipeline_utils.hpp"
#include "mlir-extensions/utils.hpp"

namespace {
struct PassManagerStage {
  template <typename F>
  PassManagerStage(mlir::MLIRContext &ctx,
                   const plier::CompilerContext::Settings &settings,
                   F &&init_func)
      : pm(&ctx) {
    pm.enableVerifier(settings.verify);

    if (settings.passStatistics) {
      pm.enableStatistics();
    }
    if (settings.passTimings) {
      pm.enableTiming();
    }
    if (settings.irDumpStderr) {
      ctx.disableMultithreading();
      pm.enableIRPrinting();
    }
    if (settings.irPrinting) {
      struct Checker {
        llvm::SmallVector<std::string, 1> names;

        bool operator()(mlir::Pass *pass, mlir::Operation *) const {
          auto name = pass->getName();
          name.consume_front("`anonymous-namespace'::");
          name.consume_front("{anonymous}::");
          name.consume_front("(anonymous namespace)::");
          return llvm::is_contained(names, name);
        }
      };

      ctx.disableMultithreading();
      pm.enableIRPrinting(Checker{settings.irPrinting->printBefore},
                          Checker{settings.irPrinting->printAfter},
                          /*printModuleScope*/ true,
                          /*printAfterOnlyOnChange*/ false,
                          /*printAfterOnlyOnFailure*/ false,
                          *(settings.irPrinting->out));
    }

    init_func(pm);
  }

  void add_jump(mlir::StringAttr name, PassManagerStage *stage) {
    assert(!name.getValue().empty());
    assert(nullptr != stage);
    jumps.emplace_back(name, stage);
  }

  std::pair<PassManagerStage *, mlir::StringAttr>
  get_jump(mlir::ArrayAttr names) const {
    if (names) {
      for (auto &it : jumps) {
        for (auto name : names) {
          auto str = name.cast<mlir::StringAttr>();
          if (it.first == str) {
            return {it.second, str};
          }
        }
      }
    }
    return {nullptr, nullptr};
  }

  void set_next_stage(PassManagerStage *stage) {
    assert(nullptr == next_stage);
    assert(nullptr != stage);
    next_stage = stage;
  }

  PassManagerStage *get_next_stage() const { return next_stage; }

  mlir::LogicalResult run(mlir::ModuleOp op) { return pm.run(op); }

private:
  mlir::PassManager pm;
  llvm::SmallVector<std::pair<mlir::StringAttr, PassManagerStage *>, 1> jumps;
  PassManagerStage *next_stage = nullptr;
};

struct PassManagerSchedule {
  PassManagerSchedule(mlir::MLIRContext &ctx,
                      const plier::CompilerContext::Settings &settings,
                      const plier::PipelineRegistry &registry) {
    auto func = [&](auto sink) {
      struct StageDesc {
        llvm::StringRef name;
        llvm::ArrayRef<llvm::StringRef> jumps;
        std::unique_ptr<PassManagerStage> stage;
      };

      assert(nullptr == stages);
      llvm::SmallVector<StageDesc, 64> stagesTemp;
      std::unordered_map<const void *, PassManagerStage *> stages_map;

      auto add_stage = [&](llvm::StringRef name,
                           llvm::ArrayRef<llvm::StringRef> jumps,
                           auto pm_init_func) {
        assert(!name.empty());
        auto prevStage =
            (stages_map.empty() ? nullptr : stagesTemp.back().stage.get());
        stagesTemp.push_back(
            {name, jumps,
             std::make_unique<PassManagerStage>(ctx, settings, pm_init_func)});
        assert(stages_map.count(name.data()) == 0);
        stages_map.insert({name.data(), stagesTemp.back().stage.get()});
        if (nullptr != prevStage) {
          prevStage->set_next_stage(stagesTemp.back().stage.get());
        }
      };

      sink(add_stage);

      for (auto &stage : stagesTemp) {
        for (auto jump : stage.jumps) {
          assert(!jump.empty());
          auto it = stages_map.find(jump.data());
          assert(it != stages_map.end());
          assert(nullptr != it->second);
          auto name = mlir::StringAttr::get(&ctx, jump);
          stage.stage->add_jump(name, it->second);
        }
      }

      stages = std::make_unique<std::unique_ptr<PassManagerStage>[]>(
          stagesTemp.size());
      for (auto it : llvm::enumerate(stagesTemp)) {
        stages[it.index()] = std::move(it.value().stage);
      }
    };
    registry.populatePassManager(func);
  }

  mlir::LogicalResult run(mlir::ModuleOp module) {
    assert(nullptr != stages);
    auto current = stages[0].get();
    do {
      assert(nullptr != current);
      if (mlir::failed(current->run(module))) {
        return mlir::failure();
      }
      auto markers = plier::getPipelineJumpMarkers(module);
      auto jumpTarget = current->get_jump(markers);
      if (nullptr != jumpTarget.first) {
        plier::removePipelineJumpMarker(module, jumpTarget.second);
        current = jumpTarget.first;
      } else {
        current = current->get_next_stage();
      }
    } while (nullptr != current);
    return mlir::success();
  }

private:
  std::unique_ptr<std::unique_ptr<PassManagerStage>[]> stages;
};

static void printDiag(llvm::raw_ostream &os, const mlir::Diagnostic &diag) {
  os << diag;
  for (auto &note : diag.getNotes())
    os << "\n" << note;
}

} // namespace

class plier::CompilerContext::CompilerContextImpl {
public:
  CompilerContextImpl(mlir::MLIRContext &ctx,
                      const CompilerContext::Settings &settings,
                      const plier::PipelineRegistry &registry)
      : schedule(ctx, settings, registry), dumpDiag(settings.diagDumpStderr) {}

  void run(mlir::ModuleOp module) {
    std::string err;
    llvm::raw_string_ostream errStream(err);
    auto diagHandler = [&](const mlir::Diagnostic &diag) {
      if (dumpDiag)
        printDiag(llvm::errs(), diag);

      if (diag.getSeverity() == mlir::DiagnosticSeverity::Error)
        printDiag(errStream, diag);
    };

    plier::scopedDiagHandler(*module.getContext(), diagHandler, [&]() {
      if (mlir::failed(schedule.run(module))) {
        errStream << "\n";
        module.print(errStream);
        errStream.flush();
        plier::reportError(llvm::Twine("MLIR pipeline failed\n") + err);
      }
    });
  }

private:
  PassManagerSchedule schedule;
  bool dumpDiag = false;
};

plier::CompilerContext::CompilerContext(mlir::MLIRContext &ctx,
                                        const Settings &settings,
                                        const PipelineRegistry &registry)
    : impl(std::make_unique<CompilerContextImpl>(ctx, settings, registry)) {}

plier::CompilerContext::~CompilerContext() {}

void plier::CompilerContext::run(mlir::ModuleOp module) { impl->run(module); }
