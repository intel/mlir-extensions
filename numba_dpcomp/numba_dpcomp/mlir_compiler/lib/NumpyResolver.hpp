// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>
#include <string>

#include <llvm/ADT/Optional.h>
#include <llvm/ADT/SmallVector.h>

namespace llvm {
class StringRef;
}

namespace mlir {
class ArrayAttr;
class Location;
class OpBuilder;
class Value;
class ValueRange;
struct LogicalResult;
} // namespace mlir

class NumpyResolver {
public:
  NumpyResolver(const char *modName, const char *mapName);
  ~NumpyResolver();

  bool hasFunc(llvm::StringRef name) const;

  mlir::LogicalResult
  resolveFuncArgs(mlir::OpBuilder &builder, mlir::Location loc,
                  llvm::StringRef name, mlir::ValueRange args,
                  mlir::ArrayAttr argsNames,
                  llvm::SmallVectorImpl<mlir::Value> &resultArgs,
                  llvm::SmallVectorImpl<mlir::Value> &outArgs, bool &viewLike);

private:
  class Impl;

  std::unique_ptr<Impl> impl;
};
