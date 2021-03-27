#pragma once

namespace mlir
{
class RewritePatternSet;
class MLIRContext;
}

namespace plier
{
void populate_common_opts_patterns(mlir::MLIRContext& context, mlir::RewritePatternSet& patterns);
}
