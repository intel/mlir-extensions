#pragma once

namespace mlir
{
class Value;
class Location;
class OpBuilder;
class Type;
}

namespace plier
{
mlir::Value index_cast(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value src, mlir::Type dst_type);
mlir::Value index_cast(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value src);
}
