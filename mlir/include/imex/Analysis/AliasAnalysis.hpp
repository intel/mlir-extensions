// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Analysis/AliasAnalysis.h>

// TODO: this is direct copypaste from llvm LocalAliasAnalysis, upstream.
// The only difference is that function arguments respects Restrict attr.

namespace mlir {
class Operation;
}

namespace imex {
/// This class implements a local form of alias analysis that tries to identify
/// the underlying values addressed by each value and performs a few basic
/// checks to see if they alias.
class LocalAliasAnalysis {
public:
  LocalAliasAnalysis() = default;

  /// Ctor compatible with mlir analysis infra.
  LocalAliasAnalysis(mlir::Operation *){};

  /// Given two values, return their aliasing behavior.
  mlir::AliasResult alias(mlir::Value lhs, mlir::Value rhs);

  /// Return the modify-reference behavior of `op` on `location`.
  mlir::ModRefResult getModRef(mlir::Operation *op, mlir::Value location);
};

mlir::StringRef getRestrictArgName();
} // namespace imex
