#include "plier/transforms/const_utils.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Value.h>

mlir::Attribute plier::getConstVal(mlir::Operation* op)
{
    if (!op->hasTrait<mlir::OpTrait::ConstantLike>())
    {
        return {};
    }

    return op->getAttr("value");
}

mlir::Attribute plier::getConstVal(mlir::Value op)
{
    if (auto parent_op = op.getDefiningOp())
    {
        return getConstVal(parent_op);
    }
    return {};
}
