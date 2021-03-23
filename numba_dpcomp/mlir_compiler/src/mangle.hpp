#pragma once

#include <string>

namespace llvm
{
class StringRef;
class raw_ostream;
}

namespace mlir
{
class TypeRange;
}

bool mangle(llvm::raw_ostream& res, llvm::StringRef ident, mlir::TypeRange types);

std::string mangle(llvm::StringRef ident, mlir::TypeRange types);
