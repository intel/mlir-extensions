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
/// This file declares the GPUX dialect's StreamType.
///
//===----------------------------------------------------------------------===//

#ifndef _GPUX_OPS_H_INCLUDED_
#define _GPUX_OPS_H_INCLUDED_

#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

// Custom Types for Stream, Device and Context.
// They hold opaque pointers.

namespace imex {
namespace gpux {
class StreamType : public ::mlir::Type::TypeBase<StreamType, ::mlir::Type,
                                                 ::mlir::TypeStorage> {
public:
  using Base::Base;
};

class DeviceType : public ::mlir::Type::TypeBase<DeviceType, ::mlir::Type,
                                                 ::mlir::TypeStorage> {
public:
  using Base::Base;
};

class ContextType : public ::mlir::Type::TypeBase<ContextType, ::mlir::Type,
                                                  ::mlir::TypeStorage> {
public:
  using Base::Base;
};

} // namespace gpux
} // namespace imex

#include <imex/Dialect/GPUX/IR/GPUXOpsDialect.h.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/GPUX/IR/GPUXOpsTypes.h.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/GPUX/IR/GPUXOps.h.inc>

#endif // _GPUX_OPS_H_INCLUDED_
