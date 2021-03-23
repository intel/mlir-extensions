#pragma once

namespace mlir
{
class OwningRewritePatternList;
class MLIRContext;
}

namespace plier
{
void populate_common_opts_patterns(mlir::MLIRContext& context, mlir::OwningRewritePatternList& patterns);
}
