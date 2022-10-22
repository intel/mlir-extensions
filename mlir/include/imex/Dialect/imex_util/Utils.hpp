// Copyright 2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
                     loc, env, /*args*/ llvm::None, results, bodyBuilder)
                 .getResults();
  return {res.begin(), res.end()};
}
} // namespace util
} // namespace imex
