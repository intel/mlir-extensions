// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
