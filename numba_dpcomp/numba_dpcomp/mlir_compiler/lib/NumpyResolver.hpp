// Copyright 2022 Intel Corporation
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
                  llvm::SmallVectorImpl<mlir::Value> &resultArgs);

private:
  class Impl;

  std::unique_ptr<Impl> impl;
};
