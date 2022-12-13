// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Compiler/Compiler.hpp"

#include "imex/Compiler/PipelineRegistry.hpp"
#include "imex/Transforms/PipelineUtils.hpp"
#include "imex/Utils.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <llvm/Support/raw_ostream.h>

#include <unordered_map>

namespace {
struct PassManagerStage {
  template <typename F>
  PassManagerStage(mlir::MLIRContext &ctx,
                   const imex::CompilerContext::Settings &settings,
                   F &&initFunc)
      : pm(&ctx) {
    pm.enableVerifier(settings.verify);

    if (settings.passStatistics)
      pm.enableStatistics();

    if (settings.passTimings)
      pm.enableTiming();

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

    initFunc(pm);
  }

  void addJump(mlir::StringAttr name, PassManagerStage *stage) {
    assert(!name.getValue().empty());
    assert(nullptr != stage);
    jumps.emplace_back(name, stage);
  }

  std::pair<PassManagerStage *, mlir::StringAttr>
  getJump(mlir::ArrayAttr names) const {
    if (names) {
      for (auto &it : jumps) {
        for (auto name : names) {
          auto str = name.cast<mlir::StringAttr>();
          if (it.first == str)
            return {it.second, str};
        }
      }
    }
    return {nullptr, nullptr};
  }

  void setNextStage(PassManagerStage *stage) {
    assert(nullptr == nextStage);
    assert(nullptr != stage);
    nextStage = stage;
  }

  PassManagerStage *getNextStage() const { return nextStage; }

  mlir::LogicalResult run(mlir::ModuleOp op) { return pm.run(op); }

private:
  mlir::PassManager pm;
  llvm::SmallVector<std::pair<mlir::StringAttr, PassManagerStage *>, 1> jumps;
  PassManagerStage *nextStage = nullptr;
};

struct PassManagerSchedule {
  PassManagerSchedule(mlir::MLIRContext &ctx,
                      const imex::CompilerContext::Settings &settings,
                      const imex::PipelineRegistry &registry) {
    auto func = [&](auto sink) {
      struct StageDesc {
        llvm::StringRef name;
        llvm::ArrayRef<llvm::StringRef> jumps;
        std::unique_ptr<PassManagerStage> stage;
      };

      assert(nullptr == stages);
      llvm::SmallVector<StageDesc, 64> stagesTemp;
      std::unordered_map<const void *, PassManagerStage *> stagesMap;

      auto addStage = [&](llvm::StringRef name,
                          llvm::ArrayRef<llvm::StringRef> jumps,
                          auto pmInitFunc) {
        assert(!name.empty());
        auto prevStage =
            (stagesMap.empty() ? nullptr : stagesTemp.back().stage.get());
        stagesTemp.push_back(
            {name, jumps,
             std::make_unique<PassManagerStage>(ctx, settings, pmInitFunc)});
        assert(stagesMap.count(name.data()) == 0);
        stagesMap.insert({name.data(), stagesTemp.back().stage.get()});
        if (nullptr != prevStage)
          prevStage->setNextStage(stagesTemp.back().stage.get());
      };

      sink(addStage);

      for (auto &stage : stagesTemp) {
        for (auto jump : stage.jumps) {
          assert(!jump.empty());
          auto it = stagesMap.find(jump.data());
          assert(it != stagesMap.end());
          assert(nullptr != it->second);
          auto name = mlir::StringAttr::get(&ctx, jump);
          stage.stage->addJump(name, it->second);
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
      if (mlir::failed(current->run(module)))
        return mlir::failure();

      auto markers = imex::getPipelineJumpMarkers(module);
      auto jumpTarget = current->getJump(markers);
      if (nullptr != jumpTarget.first) {
        imex::removePipelineJumpMarker(module, jumpTarget.second);
        current = jumpTarget.first;
      } else {
        current = current->getNextStage();
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

class imex::CompilerContext::CompilerContextImpl {
public:
  CompilerContextImpl(mlir::MLIRContext &ctx,
                      const CompilerContext::Settings &settings,
                      const imex::PipelineRegistry &registry)
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

    imex::scopedDiagHandler(*module.getContext(), diagHandler, [&]() {
      if (mlir::failed(schedule.run(module))) {
        errStream << "\n";
        module.print(errStream);
        errStream.flush();
        imex::reportError(llvm::Twine("MLIR pipeline failed\n") + err);
      }
    });
  }

private:
  PassManagerSchedule schedule;
  bool dumpDiag = false;
};

imex::CompilerContext::CompilerContext(mlir::MLIRContext &ctx,
                                       const Settings &settings,
                                       const PipelineRegistry &registry)
    : impl(std::make_unique<CompilerContextImpl>(ctx, settings, registry)) {}

imex::CompilerContext::~CompilerContext() {}

void imex::CompilerContext::run(mlir::ModuleOp module) { impl->run(module); }
