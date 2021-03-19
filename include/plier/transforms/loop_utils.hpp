#pragma once

#include <llvm/ADT/STLExtras.h>

namespace mlir
{
struct LogicalResult;
class PatternRewriter;
class Value;
class Location;
class OpBuilder;
class Type;
class Region;
namespace scf
{
class ForOp;
}
}

namespace plier
{
class GetiterOp;
}

namespace plier
{
mlir::LogicalResult lower_while_to_for(plier::GetiterOp getiter, mlir::PatternRewriter& builder,
    llvm::function_ref<std::tuple<mlir::Value,mlir::Value,mlir::Value>(mlir::OpBuilder&, mlir::Location)> get_bounds,
    llvm::function_ref<mlir::Value(mlir::OpBuilder&, mlir::Location, mlir::Type, mlir::Value)> get_iter_val,
    llvm::function_ref<void(mlir::scf::ForOp)> results = nullptr);

mlir::LogicalResult naivelyFuseParallelOps(mlir::Region &region);
}
