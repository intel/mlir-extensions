#include "plier/transforms/func_utils.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>

#include <llvm/ADT/StringRef.h>

mlir::FuncOp plier::add_function(
    mlir::OpBuilder& builder, mlir::ModuleOp module, llvm::StringRef name,
    mlir::FunctionType type)
{
    mlir::OpBuilder::InsertionGuard guard(builder);
    // Insert before module terminator.
    builder.setInsertionPoint(module.getBody(),
                              std::prev(module.getBody()->end()));
    auto func = builder.create<mlir::FuncOp>(builder.getUnknownLoc(), name, type);
    func.setPrivate();
    return func;
}
