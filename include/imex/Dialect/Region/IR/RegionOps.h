//===- RegionOps.h - Region dialect  ----------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the Region dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _Region_OPS_H_INCLUDED_
#define _Region_OPS_H_INCLUDED_

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <imex/Dialect/Region/IR/RegionOpsDialect.h.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/Region/IR/RegionOpsTypes.h.inc>
#define GET_ATTRDEF_CLASSES
#include <imex/Dialect/Region/IR/RegionOpsAttrs.h.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/Region/IR/RegionOps.h.inc>

namespace imex {
namespace region {} // namespace region
} // namespace imex

#endif // _Region_OPS_H_INCLUDED_
