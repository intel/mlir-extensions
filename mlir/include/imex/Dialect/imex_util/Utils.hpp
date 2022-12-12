// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "imex/Dialect/imex_util/Dialect.hpp"

namespace imex {
namespace util {
template <typename Builder, typename F>
static llvm::SmallVector<mlir::Value>
wrapEnvRegion(Builder &builder, mlir::Location loc, mlir::Attribute env,
              mlir::TypeRange results, F &&func) {
  if (!env) {
    auto res = func(builder, loc);
    mlir::ValueRange range(res);
    assert(range.getTypes() == results && "Invalid result types");
    return {range.begin(), range.end()};
  }

  auto bodyBuilder = [&](mlir::OpBuilder &b, mlir::Location l) {
    auto res = func(static_cast<Builder &>(b), l);
    mlir::ValueRange range(res);
    assert(range.getTypes() == results && "Invalid result types");
    b.create<imex::util::EnvironmentRegionYieldOp>(l, range);
  };

  auto res = builder
                 .template create<imex::util::EnvironmentRegionOp>(
                     loc, env, /*args*/ std::nullopt, results, bodyBuilder)
                 .getResults();
  return {res.begin(), res.end()};
}
} // namespace util
} // namespace imex
