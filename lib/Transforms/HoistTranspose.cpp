//======-- HoistTranspose.cpp - HoistTranspose Pass  ----------*- C++-*-======//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains HoistTranspose pass.
///
//===----------------------------------------------------------------------===//

#include "imex/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <memory>
#include <utility>

namespace imex {
#define GEN_PASS_DEF_HOISTTRANSPOSE
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

namespace hoisttranspose {

// This pattern tries to hoist transpose ops before extract strided slice ops if
// possible. The following code sequence:
// clang-format off
// %0 = load ...
// %extract0 = vector.extract_strided_slice %0 ...
// %extract1 = vector.extract_strided_slice %0 ...
// ....
// %extractN = vector.extract_strided_slice %0 ...
// %transpose0 = vector.transpose %extract0 ...
// %transpose1 = vector.transpose %extract1 ...
// ....
// %transposeN = vector.transpose %extractN ...
// clang-format on
// gets converted to:
// clang-format off
// %0 = load ...
// %1 = vector.transpose %0
// %transpose0 = vector.extract_strided_slice %1 ...
// %transpose1 = vector.extract_strided_slice %1 ...
// ....
// %transposeN = vector.extract_strided_slice %1 ...
// clang-format on
struct HoistTransposeBeforeExtractStridedSliceOpPattern
    : public mlir::OpRewritePattern<mlir::vector::TransposeOp> {
  HoistTransposeBeforeExtractStridedSliceOpPattern(
      mlir::MLIRContext *context,
      llvm::SmallDenseSet<mlir::vector::TransposeOp> &candidates)
      : OpRewritePattern<mlir::vector::TransposeOp>(context),
        candidates(candidates) {}
  llvm::SmallDenseSet<mlir::vector::TransposeOp> &candidates;
  mlir::LogicalResult
  matchAndRewrite(mlir::vector::TransposeOp transposeOp,
                  mlir::PatternRewriter &rewriter) const override {
    // Check if the transpose op is a candidate for hoisting.
    if (!candidates.count(transposeOp))
      return mlir::failure();
    // Source must be a extract strided slice op.
    auto extractOp = llvm::dyn_cast<mlir::vector::ExtractStridedSliceOp>(
        transposeOp.getVector().getDefiningOp());
    if (!extractOp)
      return mlir::failure();
    auto sourceOfExtract = extractOp.getSource().getDefiningOp();
    if (!sourceOfExtract)
      return mlir::failure();
    // Check if the source is already transposed by previous application of this
    // pattern.
    mlir::vector::TransposeOp transposedLoad = nullptr;
    for (auto user : sourceOfExtract->getUsers()) {
      if (auto transposeUser =
              llvm::dyn_cast<mlir::vector::TransposeOp>(user)) {
        transposedLoad = transposeUser;
        break;
      }
    }
    // If not found, create a new transpose op.
    if (!transposedLoad)
      transposedLoad = mlir::vector::TransposeOp::create(rewriter,
          transposeOp.getLoc(), sourceOfExtract->getResult(0),
          llvm::ArrayRef<int64_t>({1, 0}));
    // Extract the required slice from the transposed load and replace the
    // original transpose op with it.
    auto swap2DArrayAttr = [](mlir::ArrayAttr arrayAttr) {
      assert(arrayAttr.size() == 2 && "Expected 2D array attribute.");
      llvm::SmallVector<mlir::Attribute> swappedAttrs{arrayAttr[1],
                                                      arrayAttr[0]};
      return mlir::ArrayAttr::get(arrayAttr.getContext(), swappedAttrs);
    };
    rewriter.replaceOpWithNewOp<mlir::vector::ExtractStridedSliceOp>(
        transposeOp, transposeOp.getType(), transposedLoad.getResult(),
        swap2DArrayAttr(extractOp.getOffsets()),
        swap2DArrayAttr(extractOp.getSizes()),
        swap2DArrayAttr(extractOp.getStrides()));
    return mlir::success();
  }
};

struct HoistTransposePass final
    : public imex::impl::HoistTransposeBase<HoistTransposePass> {
  void runOnOperation() override {
    auto *context = &getContext();
    mlir::Operation *op = getOperation();
    llvm::SmallDenseSet<mlir::vector::TransposeOp> transposeOps;

    // Visit ExtractStridedSliceOp and check if it is followed by a TransposeOp.
    auto visitExtractStridedSliceOp =
        [&](mlir::vector::ExtractStridedSliceOp extractStridedSliceOp)
        -> mlir::vector::TransposeOp {
      // If extract op has more than one user, skip.
      if (!extractStridedSliceOp->hasOneUse())
        return nullptr;
      // If the user is not a transpose op, skip.
      auto transposeOp = llvm::dyn_cast_if_present<mlir::vector::TransposeOp>(
          *extractStridedSliceOp->user_begin());
      if (!(transposeOp &&
            transposeOp.getPermutation() == llvm::ArrayRef<int64_t>({1, 0})))
        return nullptr;
      return transposeOp;
    };

    op->walk([&](mlir::xegpu::LoadNdOp loadOp) -> mlir::WalkResult {
      // Check all users of the load op are,
      // 1. ExtractStridedSliceOp -> TransposeOp chain
      // 2. ExtractOp -> ExtractStridedSliceOp -> TransposeOp chain
      for (auto user : loadOp->getUsers()) {
        if (auto extractStridedSliceOp =
                llvm::dyn_cast_if_present<mlir::vector::ExtractStridedSliceOp>(
                    user)) {
          auto found = visitExtractStridedSliceOp(extractStridedSliceOp);
          if (found)
            transposeOps.insert(found);
        } else if (auto extractOp =
                       llvm::dyn_cast_if_present<mlir::vector::ExtractOp>(
                           user)) {
          for (auto extractUser : extractOp->getUsers()) {
            if (auto extractStridedSliceOp = llvm::dyn_cast_if_present<
                    mlir::vector::ExtractStridedSliceOp>(extractUser)) {
              auto found = visitExtractStridedSliceOp(extractStridedSliceOp);
              if (found)
                transposeOps.insert(found);
            }
          }
        } else {
          return mlir::WalkResult::skip();
        }
      }
      return mlir::WalkResult::advance();
    });

    mlir::RewritePatternSet patterns(context);
    mlir::GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);
    config.setUseTopDownTraversal(true);
    config.setStrictness(mlir::GreedyRewriteStrictness::ExistingAndNewOps);
    // TODO: Currently we only support hoisting TransposeOps before
    // ExtractStridedSliceOp. We may also want to support hoisting TransposeOps
    // before element-wise ops.
    patterns.add<HoistTransposeBeforeExtractStridedSliceOpPattern>(
        context, transposeOps);
    if (failed(applyPatternsGreedily(op, std::move(patterns), config))) {
      return signalPassFailure();
    }
  }
};
} // namespace hoisttranspose

std::unique_ptr<mlir::Pass> imex::createHoistTransposePass() {
  return std::make_unique<hoisttranspose::HoistTransposePass>();
}
