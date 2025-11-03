//===- RegionOps.cpp - Region dialect -------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the Region dialect and its basic operations.
/// Ported from numba-mlir
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/Region/IR/RegionOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

namespace imex {
namespace region {

void RegionDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/Region/IR/RegionOpsTypes.cpp.inc>
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include <imex/Dialect/Region/IR/RegionOpsAttrs.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <imex/Dialect/Region/IR/RegionOps.cpp.inc>
      >();
}

//===----------------------------------------------------------------------===//
// EnvironmentRegionOp
//===----------------------------------------------------------------------===//

void EnvironmentRegionOp::getSuccessorRegions(
    ::mlir::RegionBranchPoint point,
    ::mlir::SmallVectorImpl<::mlir::RegionSuccessor> &regions) {
  // If the predecessor is the ExecuteRegionOp, branch into the body.
  if (point.isParent()) {
    regions.push_back(::mlir::RegionSuccessor(&getRegion()));
    return;
  }

  // Otherwise, the region branches back to the parent operation.
  regions.push_back(::mlir::RegionSuccessor(getOperation(), getResults()));
}

void EnvironmentRegionOp::inlineIntoParent(::mlir::PatternRewriter &builder,
                                           EnvironmentRegionOp op) {
  ::mlir::Block *block = &op.getRegion().front();
  auto term = ::mlir::cast<EnvironmentRegionYieldOp>(block->getTerminator());
  auto args = llvm::to_vector(term.getResults());
  builder.eraseOp(term);
  builder.inlineBlockBefore(block, op);
  builder.replaceOp(op, args);
}

void EnvironmentRegionOp::build(
    ::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
    ::mlir::Attribute environment, ::mlir::ValueRange args,
    ::mlir::TypeRange results,
    ::mlir::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>
        bodyBuilder) {
  build(odsBuilder, odsState, results, environment, args);
  ::mlir::Region *bodyRegion = odsState.regions.back().get();

  bodyRegion->push_back(new ::mlir::Block);
  ::mlir::Block &bodyBlock = bodyRegion->front();
  if (bodyBuilder) {
    ::mlir::OpBuilder::InsertionGuard guard(odsBuilder);
    odsBuilder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(odsBuilder, odsState.location);
  }
  ensureTerminator(*bodyRegion, odsBuilder, odsState.location);
}

/// Merge adjacent env regions.
struct MergeAdjacentRegions
    : public mlir::OpRewritePattern<EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EnvironmentRegionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Get next pos and check if it is also env region op, current op cannot be
    // last as it is not a terminator.
    auto opPos = op->getIterator();
    auto nextOp = mlir::dyn_cast<EnvironmentRegionOp>(*std::next(opPos));
    if (!nextOp)
      return mlir::failure();

    if (nextOp.getEnvironment() != op.getEnvironment() ||
        nextOp.getArgs() != op.getArgs())
      return mlir::failure();

    mlir::Block *body = &op.getRegion().front();
    auto term = mlir::cast<EnvironmentRegionYieldOp>(body->getTerminator());

    auto results = op.getResults();
    auto yieldArgs = term.getResults();
    assert(results.size() == yieldArgs.size());
    auto count = static_cast<unsigned>(results.size());

    // Check if any results from first op are being used in second one, we need
    // to replace them by direct values.
    for (auto i : llvm::seq(0u, count)) {
      auto res = results[i];
      for (auto &use : llvm::make_early_inc_range(res.getUses())) {
        auto *owner = use.getOwner();
        if (nextOp->isProperAncestor(owner)) {
          auto arg = yieldArgs[i];
          rewriter.modifyOpInPlace(owner, [&]() { use.set(arg); });
        }
      }
    }

    mlir::Block *nextBody = &nextOp.getRegion().front();
    auto nextTerm =
        mlir::cast<EnvironmentRegionYieldOp>(nextBody->getTerminator());
    auto nextYieldArgs = nextTerm.getResults();

    // Construct merged yield args list, some of the results may become unused,
    // but they will be cleaned up by other pattern.
    llvm::SmallVector<mlir::Value> newYieldArgs(count + nextYieldArgs.size());
    llvm::copy(nextYieldArgs, llvm::copy(yieldArgs, newYieldArgs.begin()));

    {
      // Merge region from second op into first one.
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.inlineBlockBefore(nextBody, term);
      rewriter.setInsertionPoint(term);
      EnvironmentRegionYieldOp::create(rewriter, term->getLoc(), newYieldArgs);
      rewriter.eraseOp(term);
      rewriter.eraseOp(nextTerm);
    }

    // Construct new env region op and steal new merged region into it.
    mlir::ValueRange newYieldArgsRange(newYieldArgs);
    auto newOp = EnvironmentRegionOp::create(rewriter, op->getLoc(),
                                             newYieldArgsRange.getTypes(),
                                             op.getEnvironment(), op.getArgs());
    mlir::Region &newRegion = newOp.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), newRegion, newRegion.end());

    auto newResults = newOp.getResults();

    rewriter.replaceOp(op, newResults.take_front(count));
    rewriter.replaceOp(nextOp, newResults.drop_front(count));
    return mlir::success();
  }
};

/// Remove duplicated and unused env region yield args.
struct CleanupRegionYieldArgs
    : public mlir::OpRewritePattern<EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EnvironmentRegionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Block *body = &op.getRegion().front();
    auto term = mlir::cast<EnvironmentRegionYieldOp>(body->getTerminator());

    auto results = op.getResults();
    auto yieldArgs = term.getResults();
    assert(results.size() == yieldArgs.size());
    auto count = static_cast<unsigned>(results.size());

    // Build new yield args list, and mapping between old and new results
    llvm::SmallVector<mlir::Value> newYieldArgs;
    llvm::SmallVector<int> newResultsMapping(count, -1);
    llvm::SmallDenseMap<mlir::Value, int> argsMap;
    for (auto i : llvm::seq(0u, count)) {
      auto res = results[i];

      // Unused result.
      if (res.getUses().empty())
        continue;

      auto arg = yieldArgs[i];
      auto it = argsMap.find_as(arg);
      if (it == argsMap.end()) {
        // Add new result, compute index mapping for it.
        auto ind = static_cast<int>(newYieldArgs.size());
        argsMap.insert({arg, ind});
        newYieldArgs.emplace_back(arg);
        newResultsMapping[i] = ind;
      } else {
        // Duplicated result, reuse prev result index.
        newResultsMapping[i] = it->second;
      }
    }

    // Same yield results count - nothing changed.
    if (newYieldArgs.size() == count)
      return mlir::failure();

    // Construct new env region op, only yielding values we selected.
    mlir::ValueRange newYieldArgsRange(newYieldArgs);
    auto newOp = EnvironmentRegionOp::create(rewriter, op->getLoc(),
                                             newYieldArgsRange.getTypes(),
                                             op.getEnvironment(), op.getArgs());
    mlir::Region &newRegion = newOp.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), newRegion, newRegion.end());
    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<EnvironmentRegionYieldOp>(term, newYieldArgs);
    }

    // Construct new result list, using mapping previously constructed.
    auto newResults = newOp.getResults();
    llvm::SmallVector<mlir::Value> newResultsToTeplace(count);
    for (auto i : llvm::seq(0u, count)) {
      auto mapInd = newResultsMapping[i];
      if (mapInd != -1)
        newResultsToTeplace[i] = newResults[mapInd];
    }

    rewriter.replaceOp(op, newResultsToTeplace);
    return mlir::success();
  }
};

void EnvironmentRegionOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.insert<MergeAdjacentRegions, CleanupRegionYieldArgs>(context);
}

} // namespace region
} // namespace imex

#include <imex/Dialect/Region/IR/RegionOpsDialect.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/Region/IR/RegionOpsTypes.cpp.inc>
#define GET_ATTRDEF_CLASSES
#include <imex/Dialect/Region/IR/RegionOpsAttrs.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/Region/IR/RegionOps.cpp.inc>
