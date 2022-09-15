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

#include "mlir-extensions/Dialect/ntensor/IR/NTensorOps.hpp"

#include <mlir/Transforms/InliningUtils.h>

namespace {
struct InlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                       mlir::BlockAndValueMapping &) const final override {
    return true;
  }
  bool isLegalToInline(mlir::Operation *op, mlir::Region *, bool,
                       mlir::BlockAndValueMapping &) const final override {
    return true;
  }
};
} // namespace

namespace imex {
namespace ntensor {

void NTensorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-extensions/Dialect/ntensor/IR/NTensorOps.cpp.inc"
      >();

  addInterfaces<InlinerInterface>();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir-extensions/Dialect/ntensor/IR/NTensorOpsAttributes.cpp.inc"
      >();
}

} // namespace ntensor
} // namespace imex

#include "mlir-extensions/Dialect/ntensor/IR/NTensorOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mlir-extensions/Dialect/ntensor/IR/NTensorOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir-extensions/Dialect/ntensor/IR/NTensorOpsAttributes.cpp.inc"

#include "mlir-extensions/Dialect/ntensor/IR/NTensorOpsEnums.cpp.inc"
