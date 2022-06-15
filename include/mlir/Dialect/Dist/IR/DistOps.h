// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef _Dist_OPS_H_INCLUDED_
#define _Dist_OPS_H_INCLUDED_

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>

namespace dist {

}

#include <mlir/Dialect/Dist/IR/DistOpsDialect.h.inc>
#define GET_TYPEDEF_CLASSES
#include <mlir/Dialect/Dist/IR/DistOpsTypes.h.inc>
#define GET_OP_CLASSES
#include <mlir/Dialect/Dist/IR/DistOps.h.inc>

#endif // _Dist_OPS_H_INCLUDED_
