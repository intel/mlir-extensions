#pragma once

namespace mlir
{
class OwningRewritePatternList;
class MLIRContext;
}

namespace plier
{
void populate_index_propagate_patterns(mlir::MLIRContext& context, mlir::OwningRewritePatternList& patterns);
}
