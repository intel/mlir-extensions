//===- AddCommCacheKeys.cpp - OverlapCommAndCompute Transform *- C++ -*-//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///     This file implements adding unique keys to update_halo ops for caching.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/DistRuntime/IR/DistRuntimeOps.h>

#include "PassDetail.h"

namespace imex {
namespace distruntime {

namespace {
struct AddCommCacheKeysPass
    : public ::imex::AddCommCacheKeysBase<AddCommCacheKeysPass> {

  AddCommCacheKeysPass() = default;

  /// @brief Add unique cache key to every distruntime::GetHaloOp
  void runOnOperation() override {
    auto root = this->getOperation();
    static int64_t key = -1;

    // find all GetHaloOps and assign a unique key to each instance
    root->walk([&](::imex::distruntime::GetHaloOp op) { op.setKey(++key); });
  }
};

} // namespace
} // namespace distruntime

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::Pass> createAddCommCacheKeysPass() {
  return std::make_unique<::imex::distruntime::AddCommCacheKeysPass>();
}

} // namespace imex
