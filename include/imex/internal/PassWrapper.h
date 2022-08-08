// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef _PassWrapper_H_INCLUDED_
#define _PassWrapper_H_INCLUDED_

#include <type_traits>

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace imex {

/// Convenience functions for filling in a pattern set with patterns provided as
/// template args
template <typename... Rewrites>
void insertPatterns(::mlir::MLIRContext &context,
                    ::mlir::FrozenRewritePatternSet &patterns) {
  ::mlir::RewritePatternSet p(&context);
  p.insert<Rewrites...>(&context);
  patterns = std::move(p);
}

} // namespace imex
#endif // _PassWrapper_H_INCLUDED_
