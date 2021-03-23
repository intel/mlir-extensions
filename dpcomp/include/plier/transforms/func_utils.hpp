#pragma once

namespace mlir
{
class ModuleOp;
class FuncOp;
class OpBuilder;
class FunctionType;
}

namespace llvm
{
class StringRef;
}

namespace plier
{
mlir::FuncOp add_function(mlir::OpBuilder& builder, mlir::ModuleOp module,
                          llvm::StringRef name, mlir::FunctionType type);
}
