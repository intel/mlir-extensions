#pragma once

#include <memory>

#include <llvm/ADT/Optional.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/ArrayRef.h>

namespace llvm
{
class StringRef;
}

namespace mlir
{
class Value;
class FuncOp;
class ValueRange;
class OpBuilder;
class Location;
}

class PyLinalgResolver
{
public:
    PyLinalgResolver();
    ~PyLinalgResolver();

    using Values = llvm::SmallVector<mlir::Value, 8>;
    using KWArgs = llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>>;

    llvm::Optional<Values> rewrite_func(llvm::StringRef name, mlir::Location loc, mlir::OpBuilder& builder, mlir::ValueRange args,
                                        KWArgs kwargs);

    llvm::Optional<Values> rewrite_attr(llvm::StringRef name, mlir::Location loc, mlir::OpBuilder& builder, mlir::Value arg);

private:
    friend struct PyBuilderContext;
    struct Context;
    std::unique_ptr<Context> context;

    llvm::Optional<Values> rewrite(llvm::StringRef name, mlir::Location loc, mlir::OpBuilder& builder, mlir::ValueRange args,
                                   KWArgs kwargs);
};
