// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

namespace llvm {
class StringRef;
}

namespace mlir {
namespace func {
class FuncOp;
}
} // namespace mlir

namespace mlir {
class TypeRange;
} // namespace mlir

class PyFuncResolver {
public:
  PyFuncResolver();
  ~PyFuncResolver();

  mlir::func::FuncOp getFunc(llvm::StringRef name, mlir::TypeRange types) const;

private:
  struct Context;
  std::unique_ptr<Context> context;
};
