#include "plier/rewrites/common_opts.hpp"

#include "plier/rewrites/force_inline.hpp"
#include "plier/rewrites/index_type_propagation.hpp"
#include "plier/rewrites/loop_rewrites.hpp"
#include "plier/rewrites/cse.hpp"
#include "plier/rewrites/if_rewrites.hpp"

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/BuiltinOps.h>

void plier::populate_common_opts_patterns(mlir::MLIRContext& context, mlir::OwningRewritePatternList& patterns)
{
    for (auto *op : context.getRegisteredOperations())
    {
        op->getCanonicalizationPatterns(patterns, &context);
    }

    patterns.insert<
        //        LoopInvariantCodeMotion, TODO
        plier::ForceInline,
        plier::CmpLoopBoundsSimplify,
        SimplifyEmptyIf,
        plier::IfOpConstCond,
        SimplifySelect,
        SimplifySelectEq,
        plier::CSERewrite<mlir::FuncOp, /*recusive*/false>
        >(&context);

    plier::populate_index_propagate_patterns(context, patterns);
}
