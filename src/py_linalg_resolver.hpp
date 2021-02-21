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

    llvm::Optional<Values> rewrite(llvm::StringRef name, mlir::Location loc, mlir::OpBuilder& builder, mlir::ValueRange args,
                                   llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs);

private:
    friend struct PyBuilderContext;
    struct Context;
    std::unique_ptr<Context> context;
};
