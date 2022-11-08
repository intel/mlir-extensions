//===- PassWrapper.h - ----------------------------------------*- C++//-*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
//===----------------------------------------------------------------------===//

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
