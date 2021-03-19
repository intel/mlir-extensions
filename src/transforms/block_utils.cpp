#include "plier/transforms/block_utils.hpp"

#include <mlir/IR/Operation.h>

namespace
{
auto collectParentOps(mlir::Operation* op)
{
    llvm::SmallVector<mlir::Operation*> ops;
    while (true)
    {
        assert(op);
        ops.emplace_back(op);
        auto parent = op->getParentOp();
        if (!parent)
        {
            break;
        }
        op = parent;
    }
    return ops;
}
}

plier::OpRelation plier::relativeTo(mlir::Operation* op, mlir::Operation* relativeTo)
{
    assert(op);
    assert(relativeTo);

    for (auto& reg : relativeTo->getRegions())
    {
        for (auto& block : reg)
        {
            if (block.findAncestorOpInBlock(*op))
            {
                return OpRelation::In;
            }
        }
    }

    auto ops1 = collectParentOps(op);
    auto ops2 = collectParentOps(relativeTo);

    for (auto op1 : ops1)
    {
        assert(op1);
        for (auto op2 : ops2)
        {
            assert(op2);
            if (op1->getBlock() == op1->getBlock())
            {
                if (op1->isBeforeInBlock(op1))
                {
                    return OpRelation::Before;
                }
                else
                {
                    return OpRelation::After;
                }
            }
        }
    }
    return OpRelation::Unknown;
}
