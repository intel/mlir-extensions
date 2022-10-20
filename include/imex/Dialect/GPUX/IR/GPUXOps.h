//===- GPUXOps.h - GPUX dialect  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the GPUX dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _GPUX_OPS_H_INCLUDED_
#define _GPUX_OPS_H_INCLUDED_

#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <imex/Dialect/GPUX/IR/GPUXOpsDialect.h.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/GPUX/IR/GPUXOpsTypes.h.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/GPUX/IR/GPUXOps.h.inc>

#endif // _GPUX_OPS_H_INCLUDED_
