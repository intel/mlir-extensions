#pragma once

namespace mlir
{
class FuncOp;
struct LogicalResult;
}

namespace plier
{
mlir::LogicalResult optimizeMemoryOps(mlir::FuncOp func);
}
