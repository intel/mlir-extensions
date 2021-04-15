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
