#include "plier/transforms/cast_utils.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>

mlir::Value plier::index_cast(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value src, mlir::Type dst_type)
{
    auto src_type = src.getType();
    assert(src_type.isa<mlir::IndexType>() || dst_type.isa<mlir::IndexType>());
    if (src_type != dst_type)
    {
        return builder.create<mlir::IndexCastOp>(loc, src, dst_type);
    }
    return src;
}

mlir::Value plier::index_cast(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value src)
{
    return index_cast(builder, loc, src, mlir::IndexType::get(builder.getContext()));
}
