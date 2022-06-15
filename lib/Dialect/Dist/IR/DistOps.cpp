// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <mlir/Dialect/Dist/IR/DistOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>

namespace dist {

    void DistDialect::initialize()
    {
        addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Dist/IR/DistOpsTypes.cpp.inc"
            >();
        addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Dist/IR/DistOps.cpp.inc"
            >();
    }

} // namespace dist

#include "mlir/Dialect/Dist/IR/DistOpsDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Dist/IR/DistOpsTypes.cpp.inc"
#define GET_OP_CLASSES
#include "mlir/Dialect/Dist/IR/DistOps.cpp.inc"
