#pragma once

#include <memory>

namespace llvm
{
class StringRef;
}

namespace mlir
{
class FuncOp;
class TypeRange;
}

class PyFuncResolver
{
public:
    PyFuncResolver();
    ~PyFuncResolver();

    mlir::FuncOp get_func(llvm::StringRef name, mlir::TypeRange types);

private:
    struct Context;
    std::unique_ptr<Context> context;
};
