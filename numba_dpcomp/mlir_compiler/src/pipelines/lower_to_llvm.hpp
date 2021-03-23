#pragma once

namespace plier
{
class PipelineRegistry;
}

namespace llvm
{
class StringRef;
}

void register_lower_to_llvm_pipeline(plier::PipelineRegistry& registry);

llvm::StringRef lower_to_llvm_pipeline_name();
