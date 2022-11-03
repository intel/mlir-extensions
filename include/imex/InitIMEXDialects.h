//===- InitIMEXDialects.h - IMEX Dialects Registration ----------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all IMEX
// dialects and passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef _IMEX_INITALLDIALECTS_H_
#define _IMEX_INITALLDIALECTS_H_

#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/GPUX/IR/GPUXOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>

namespace imex {

/// Add all the IMEX dialects to the provided registry.
inline void registerAllDialects(::mlir::DialectRegistry &registry) {
  // clang-format off
    registry.insert<::imex::ptensor::PTensorDialect,
                    ::imex::dist::DistDialect,
                    ::imex::gpux::GPUXDialect>();
  // clang-format on
}

/// Append all the IMEX dialects to the registry contained in the given context.
inline void registerAllDialects(::mlir::MLIRContext &context) {
  ::mlir::DialectRegistry registry;
  ::imex::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace imex

#endif // _IMEX_INITALLDIALECTS_H_
