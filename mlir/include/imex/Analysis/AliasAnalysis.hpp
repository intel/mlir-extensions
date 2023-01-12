// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h>

namespace mlir {
class Operation;
}

namespace imex {
/// This class implements a local form of alias analysis that tries to identify
/// the underlying values addressed by each value and performs a few basic
/// checks to see if they alias.
class LocalAliasAnalysis : public mlir::LocalAliasAnalysis {
public:
  LocalAliasAnalysis() = default;

  /// Ctor compatible with mlir analysis infra.
  LocalAliasAnalysis(mlir::Operation *){};

protected:
  mlir::AliasResult aliasImpl(mlir::Value lhs, mlir::Value rhs) override;
};

mlir::StringRef getRestrictArgName();
} // namespace imex
