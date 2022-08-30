//===- IntelGpuOps.h - IntelGpu dialect  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the IntelGpu dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _IntelGpu_OPS_H_INCLUDED_
#define _IntelGpu_OPS_H_INCLUDED_

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace imex {
namespace intel_gpu {
class OpaqueType : public ::mlir::Type::TypeBase<OpaqueType, ::mlir::Type,
                                                 ::mlir::TypeStorage> {
public:
  using Base::Base;

  static OpaqueType get(mlir::MLIRContext *context);
};

} // namespace intel_gpu
} // namespace imex

#include <imex/Dialect/IntelGpu/IR/IntelGpuOpsDialect.h.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/IntelGpu/IR/IntelGpuOpsTypes.h.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/IntelGpu/IR/IntelGpuOps.h.inc>

#endif // _IntelGpu_OPS_H_INCLUDED_
