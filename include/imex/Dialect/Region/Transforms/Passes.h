//===-- Passes.h - Dist pass declaration file -------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header file defines prototypes that expose pass constructors for the
/// Region dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _Region_PASSES_H_INCLUDED_
#define _Region_PASSES_H_INCLUDED_

#include <mlir/Pass/Pass.h>

namespace imex {

//===----------------------------------------------------------------------===//
/// Dist passes.
//===----------------------------------------------------------------------===//

/// Create a RegionBufferize pass
std::unique_ptr<::mlir::Pass> createRegionBufferizePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include <imex/Dialect/Region/Transforms/Passes.h.inc>

} // namespace imex

#endif // _Region_PASSES_H_INCLUDED_
