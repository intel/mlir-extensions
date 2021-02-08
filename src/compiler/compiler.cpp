#include "plier/compiler/compiler.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <mlir/IR/Diagnostics.h>

#include <llvm/Support/raw_ostream.h>

#include <unordered_map>

#include "plier/utils.hpp"

#include "plier/compiler/pipeline_registry.hpp"

#include "plier/transforms/pipeline_utils.hpp"

namespace
{
struct PassManagerStage
{
    template<typename F>
    PassManagerStage(mlir::MLIRContext& ctx,
                     const plier::CompilerContext::Settings& settings,
                     F&& init_func):
        pm(&ctx)
    {
        pm.enableVerifier(settings.verify);

        if (settings.pass_statistics)
        {
            pm.enableStatistics();
        }
        if (settings.pass_timings)
        {
            pm.enableTiming();
        }
        if (settings.ir_printing)
        {
            ctx.enableMultithreading(false);
            pm.enableIRPrinting();
        }

        init_func(pm);
    }

    void add_jump(mlir::StringAttr name, PassManagerStage* stage)
    {
        assert(!name.getValue().empty());
        assert(nullptr != stage);
        jumps.emplace_back(name, stage);
    }

    std::pair<PassManagerStage*, mlir::StringAttr> get_jump(mlir::ArrayAttr names) const
    {
        if (names)
        {
            for (auto& it : jumps)
            {
                for (auto name : names)
                {
                    auto str = name.cast<mlir::StringAttr>();
                    if (it.first == str)
                    {
                        return {it.second, str};
                    }
                }
            }
        }
        return {nullptr, nullptr};
    }

    void set_next_stage(PassManagerStage* stage)
    {
        assert(nullptr == next_stage);
        assert(nullptr != stage);
        next_stage = stage;
    }

    PassManagerStage* get_next_stage() const
    {
        return next_stage;
    }

    mlir::LogicalResult run(mlir::ModuleOp op)
    {
        return pm.run(op);
    }

private:
    mlir::PassManager pm;
    llvm::SmallVector<std::pair<mlir::StringAttr, PassManagerStage*>, 1> jumps;
    PassManagerStage* next_stage = nullptr;
};

struct PassManagerSchedule
{
    PassManagerSchedule(mlir::MLIRContext& ctx,
                        const plier::CompilerContext::Settings& settings,
                        const plier::PipelineRegistry& registry)
    {
        auto func = [&](auto sink)
        {
            struct StageDesc
            {
                llvm::StringRef name;
                llvm::ArrayRef<llvm::StringRef> jumps;
                std::unique_ptr<PassManagerStage> stage;
            };

            assert(nullptr == stages);
            llvm::SmallVector<StageDesc, 64> stages_temp;
            std::unordered_map<const void*, PassManagerStage*> stages_map;

            auto add_stage = [&](llvm::StringRef name, llvm::ArrayRef<llvm::StringRef> jumps, auto pm_init_func)
            {
                assert(!name.empty());
                auto prev_stage = (stages_map.empty() ? nullptr : stages_temp.back().stage.get());
                stages_temp.push_back({name, jumps, std::make_unique<PassManagerStage>(ctx, settings, pm_init_func)});
                assert(stages_map.count(name.data()) == 0);
                stages_map.insert({name.data(), stages_temp.back().stage.get()});
                if (nullptr != prev_stage)
                {
                    prev_stage->set_next_stage(stages_temp.back().stage.get());
                }
            };

            sink(add_stage);

            for (auto& stage : stages_temp)
            {
                for (auto jump : stage.jumps)
                {
                    assert(!jump.empty());
                    auto it = stages_map.find(jump.data());
                    assert(it != stages_map.end());
                    assert(nullptr != it->second);
                    auto name = mlir::StringAttr::get(jump, &ctx);
                    stage.stage->add_jump(name, it->second);
                }
            }

            stages = std::make_unique<std::unique_ptr<PassManagerStage>[]>(stages_temp.size());
            for (auto it : llvm::enumerate(stages_temp))
            {
                stages[it.index()] = std::move(it.value().stage);
            }
        };
        registry.populate_pass_manager(func);
    }

    mlir::LogicalResult run(mlir::ModuleOp module)
    {
        assert(nullptr != stages);
        auto current = stages[0].get();
        do
        {
            assert(nullptr != current);
            if (mlir::failed(current->run(module)))
            {
                return mlir::failure();
            }
            auto markers = plier::get_pipeline_jump_markers(module);
            auto jump_target = current->get_jump(markers);
            if (nullptr != jump_target.first)
            {
                plier::remove_pipeline_jump_marker(module, jump_target.second);
                current = jump_target.first;
            }
            else
            {
                current = current->get_next_stage();
            }
        }
        while (nullptr != current);
        return mlir::success();
    }

private:
    std::unique_ptr<std::unique_ptr<PassManagerStage>[]> stages;
};
}

class plier::CompilerContext::CompilerContextImpl
{
public:
    CompilerContextImpl(mlir::MLIRContext& ctx,
                        const CompilerContext::Settings& settings,
                        const plier::PipelineRegistry& registry):
        schedule(ctx, settings, registry) {}

    void run(mlir::ModuleOp module)
    {
        std::string err;
        llvm::raw_string_ostream err_stream(err);
        auto diag_handler = [&](mlir::Diagnostic& diag)
        {
            if (diag.getSeverity() == mlir::DiagnosticSeverity::Error)
            {
                err_stream << diag;
            }
        };

        plier::scoped_diag_handler(*module.getContext(), diag_handler, [&]()
        {
            if (mlir::failed(schedule.run(module)))
            {
                err_stream << "\n";
                module.print(err_stream);
                err_stream.flush();
                plier::report_error(llvm::Twine("MLIR pipeline failed\n") + err);
            }
        });
    }
private:
    PassManagerSchedule schedule;
};

plier::CompilerContext::CompilerContext(mlir::MLIRContext& ctx,
                                        const Settings& settings,
                                        const PipelineRegistry& registry):
    impl(std::make_unique<CompilerContextImpl>(ctx, settings, registry))
{

}

plier::CompilerContext::~CompilerContext()
{

}

void plier::CompilerContext::run(mlir::ModuleOp module)
{
    impl->run(module);
}
