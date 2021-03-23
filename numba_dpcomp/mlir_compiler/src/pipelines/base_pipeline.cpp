#include "pipelines/base_pipeline.hpp"

#include "plier/compiler/pipeline_registry.hpp"

namespace
{
const constexpr llvm::StringRef passes[] ={
    "init",
    "lowering",
    "terminate",
};

void dummy_pass_func(mlir::OpPassManager&) {}
}

void register_base_pipeline(plier::PipelineRegistry& registry)
{
    for (std::size_t i = 0; i < llvm::array_lengthof(passes); ++i)
    {
        registry.register_pipeline([i](auto sink)
        {
            if (0 == i)
            {
                sink(passes[i], {}, {}, {}, dummy_pass_func);
            }
            else
            {
                sink(passes[i], {passes[i - 1]}, {}, {}, dummy_pass_func);
            }
        });
    }
}

PipelineStage get_high_lowering_stage()
{
    return {passes[0], passes[1]};
}

PipelineStage get_lower_lowering_stage()
{
    return {passes[1], passes[2]};
}
