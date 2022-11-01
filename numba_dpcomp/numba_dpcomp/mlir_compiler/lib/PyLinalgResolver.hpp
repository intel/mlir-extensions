// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/SmallVector.h>

namespace llvm {
class StringRef;
}

namespace mlir {
class Value;
class ValueRange;
class OpBuilder;
class Location;
} // namespace mlir

namespace mlir {
namespace func {
class FuncOp;
}
} // namespace mlir

class PyLinalgResolver {
public:
  PyLinalgResolver(const char *modName, const char *regName);
  ~PyLinalgResolver();

  using Values = llvm::SmallVector<mlir::Value, 8>;
  using KWArgs = llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>>;

  llvm::Optional<Values> rewriteFunc(llvm::Twine name, mlir::Location loc,
                                     mlir::OpBuilder &builder,
                                     mlir::ValueRange args,
                                     KWArgs kwargs) const;

  llvm::Optional<Values> rewriteAttr(llvm::Twine name, mlir::Location loc,
                                     mlir::OpBuilder &builder,
                                     mlir::Value arg) const;

private:
  friend struct PyBuilderContext;
  struct Context;
  std::unique_ptr<Context> context;

  llvm::Optional<Values> rewrite(llvm::StringRef name, mlir::Location loc,
                                 mlir::OpBuilder &builder,
                                 mlir::ValueRange args, KWArgs kwargs) const;
};
