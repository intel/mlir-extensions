#pragma once

namespace mlir
{
class Operation;
}

namespace plier
{

enum class OpRelation
{
    Before,
    After,
    In,
    Unknown
};

OpRelation relativeTo(mlir::Operation* op, mlir::Operation* relativeTo);
}
