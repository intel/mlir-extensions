//===-- OptimizeTranspose.cpp - OptimizeTranspose Pass  ----------*- C++-*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains OptimizeTranspose pass.
///
//===----------------------------------------------------------------------===//

#include "imex/Utils/XeArch.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "imex/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cassert>
#include <memory>
#include <utility>

namespace imex {
#define GEN_PASS_DEF_OPTIMIZETRANSPOSE
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

namespace optimizetranspose {

// Convenience interface for defining an op pattern.
struct PatternMatcherInterface {
public:
  virtual ~PatternMatcherInterface() = default;
  // Try to match the given op with some pattern and update the ops vector.
  virtual bool match(mlir::Operation *op,
                     llvm::SmallVectorImpl<mlir::Operation *> &ops) = 0;
};

// Pattern for detecting packed layout for DPAS B. We detect the following
// linear op sequence:
// clang-format off
// %0 = vector.shape_cast %in {packed}
// %1 = vector.shuffle %0, %0, %mask {packed}
// %2 = vector.shape_cast %1 {packed}
// clang-format on
struct PackedLayoutOpsMatcher : public PatternMatcherInterface {
  bool match(mlir::Operation *op,
             llvm::SmallVectorImpl<mlir::Operation *> &ops) override {
    // Check for first ShapeCastOp.
    auto shapeCastOp = llvm::dyn_cast_if_present<mlir::vector::ShapeCastOp>(op);
    if (!shapeCastOp || !shapeCastOp->hasAttr("packed"))
      return false;
    if (shapeCastOp.use_empty())
      return false;
    // ShapeCastOp should have a shuffle op as user.
    auto shuffleOp = llvm::dyn_cast_if_present<mlir::vector::ShuffleOp>(
        *shapeCastOp->user_begin());
    if (!shuffleOp || !shuffleOp->hasAttr("packed"))
      return false;
    // This shuffle op should use the ShapeCastOp as its only operand.
    for (auto user : shapeCastOp->getUsers()) {
      if (user != shuffleOp)
        return false;
    }
    // ShuffleOp must have single user which is ShapeCastOp.
    if (!shuffleOp->hasOneUse())
      return false;
    auto shapeCastOp2 = llvm::dyn_cast_if_present<mlir::vector::ShapeCastOp>(
        *shuffleOp->user_begin());
    if (!shapeCastOp2 || !shapeCastOp2->hasAttr("packed"))
      return false;
    // We found the desired pattern. update the ops vector.
    ops.insert(ops.end(), {shapeCastOp, shuffleOp, shapeCastOp2});
    return true;
  }
};

// Analysis to find LoadNd ops with DPAS B usage.
struct LoadTransposeAnalysis {
private:
  bool checkDPASBUsage(mlir::Operation *op) {
    // Op should have some users.
    if (op->use_empty())
      return false;
    // Now check all users are DPAS B usages.
    for (auto user : op->getUsers()) {
      auto dpasOp = llvm::dyn_cast_if_present<mlir::xegpu::DpasOp>(user);
      if (!dpasOp || dpasOp.getRhs().getDefiningOp() != op)
        return false;
    }
    return true;
  };
  // Analysis result.
  llvm::DenseSet<mlir::Operation *> candidates;

public:
  LoadTransposeAnalysis(mlir::Operation *op) {
    op->walk([&](mlir::xegpu::LoadNdOp loadOp) -> mlir::WalkResult {
      // Load op must have a single user.
      if (!loadOp->hasOneUse())
        return mlir::WalkResult::skip();
      // If load op already has transpose effect, we skip it.
      auto transposeAttr = loadOp.getTransposeAttr();
      if (transposeAttr &&
          transposeAttr.asArrayRef() == llvm::ArrayRef<int64_t>{1, 0})
        return mlir::WalkResult::skip();
      // Memory space of the load op must be global.
      if (loadOp.getTensorDesc().getType().getMemoryScope() !=
          mlir::xegpu::MemoryScope::Global)
        return mlir::WalkResult::skip();
      // Single user must be a transpose op.
      auto transposeOp = llvm::dyn_cast_if_present<mlir::vector::TransposeOp>(
          *loadOp->user_begin());
      if (!transposeOp)
        return mlir::WalkResult::skip();

      // If the load element type is >= 32 bits, we can directly consider it.
      auto opElementTy = loadOp.getTensorDesc().getType().getElementType();
      if (opElementTy.getIntOrFloatBitWidth() >= 32) {
        candidates.insert(loadOp);
        return mlir::WalkResult::advance();
      }

      llvm::DenseSet<mlir::Operation *> worklist;
      llvm::DenseSet<mlir::Operation *> leaves;
      // IR visitor to visit the def-use chain of the LoadOp. Traveral is
      // confined to a single block. So it must terminate. Traversal try to
      // check if the DAG created by load users have DPAS B usages only at the
      // leaves
      auto visitOp = [&](mlir::Operation *visitedOp) {
        worklist.insert(visitedOp);
        mlir::Block *parentBlock = visitedOp->getBlock();
        while (!worklist.empty()) {
          auto currOp = *worklist.begin();
          worklist.erase(currOp);
          // If the current op has no users, mark it as a leaf node.
          if (currOp->use_empty()) {
            leaves.insert(currOp);
            continue;
          }
          // Check if this has specified type of DPAS usage.
          if (checkDPASBUsage(currOp)) {
            for (auto user : currOp->getUsers())
              leaves.insert(user);
            continue;
          }
          // We are only interested in users in the same block.
          for (auto user : currOp->getUsers()) {
            if (user->getBlock() == parentBlock)
              worklist.insert(user);
          }
        }
      };
      // Traverse the def-use chain of the transposeOp.
      visitOp(transposeOp);
      // If not leaf nodes are found, return false.
      if (leaves.empty())
        return mlir::WalkResult::skip();
      // Check if all leaves of the DAG are DPAS ops.
      for (auto leaf : leaves) {
        if (!llvm::isa<mlir::xegpu::DpasOp>(leaf)) {
          return mlir::WalkResult::skip();
        }
      }
      // At this point, we have found a LoadNdOp with desired DPAS usage.
      candidates.insert(loadOp);
      return mlir::WalkResult::advance();
    });
  }
  // Check if a given LoadNdOp should be considered.
  bool contains(mlir::xegpu::LoadNdOp op) { return candidates.contains(op); }
  // Print the analysis result.
  void printAnalysisResult() {
    llvm::errs() << "LoadTransposeAnalysis Result:\n";
    for (auto op : candidates) {
      op->print(llvm::errs());
      llvm::errs() << "\n";
    }
  }
};

// Helper function to check if a value is within a range.
bool withinRange(int val, imex::Range range) {
  return val >= range.min && val <= range.max;
}
// Types of usages for the transpose. PACKED is used for DPAS B usage.
// NON_PACKED represents any other usage.
enum TransposeUsageType { PACKED = 1, NON_PACKED = 2 };

// This pattern detects a transpose op that is using the result of a load op and
// replace it with a new load op that does the load+transpose together. Pattern
// is only applied if the transpose is used in DPAS B. In addition packed layout
// conversion op sequence is also removed if it is present (alredy done by
// load+transpose op).
//
// Following:
// clang-format off
// %0 = load ...
// %1 = transpose %0 ...
// %2 = shape_cast %1 ...
// %3 = shuffle %2 ...
// %4 = shape_cast %3 ...
// ... DPAS B usage ...
// clang-format on
//
// is replaced with:
// clang-format off
// %0 = load ...
// %1 = load+transpose %0 ...
// ... DPAS B usage ...
// clang-format on
struct TransposeRewritePattern
    : public mlir::OpRewritePattern<mlir::vector::TransposeOp> {
  TransposeRewritePattern(mlir::MLIRContext *context,
                          LoadTransposeAnalysis &analysis,
                          std::shared_ptr<imex::XeuArchInterface> ptruArch)
      : OpRewritePattern<mlir::vector::TransposeOp>(context),
        analysis(analysis), uArchInterface(ptruArch) {}
  LoadTransposeAnalysis &analysis;
  std::shared_ptr<imex::XeuArchInterface> uArchInterface;

  // Check if the target HW allows doing the load+transpose together.
  bool canTranspose(mlir::xegpu::LoadNdOp loadOp,
                    TransposeUsageType transposeUsage) const {
    auto tdescTy = loadOp.getTensorDesc().getType();
    auto blockH = tdescTy.getShape()[0];
    auto blockW = tdescTy.getShape()[1];
    auto bitWidth = tdescTy.getElementType().getIntOrFloatBitWidth();
    auto transposeBitwidth = bitWidth;

    if (transposeUsage == TransposeUsageType::PACKED) {
      // DPASB usage requires 32 bit transpose.
      transposeBitwidth = 32;
      blockW = (blockW * bitWidth) / 32;
    } else if (bitWidth < 32 &&
               transposeUsage == TransposeUsageType::NON_PACKED) {
      // TODO: add support for DPAS A usage.
      return false;
    }
    auto load2DConfig = uArchInterface->get2DLoadConfig(
        loadOp, transposeBitwidth, /*vnni=*/false, /*transpose=*/true);
    // Check if the tranposed shape is supported by uArch.
    if (!withinRange(blockH, load2DConfig->blockHeight) ||
        !withinRange(blockW, load2DConfig->blockWidth))
      return false;
    // Check if the array length is supported by uArch.
    int arrayLen = tdescTy.getArrayLength();
    return llvm::any_of(load2DConfig->array_length,
                        [&](int len) { return len == arrayLen; });
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Check if this tranpose is using a load op.
    auto loadOp = llvm::dyn_cast_if_present<mlir::xegpu::LoadNdOp>(
        op.getVector().getDefiningOp());
    if (!loadOp)
      return mlir::failure();
    // Check if this load op is part of the analysis result.
    if (!analysis.contains(loadOp))
      return mlir::failure();
    // Check if the transpose has a single user and it has desired packed
    // layout conversion op sequence.
    if (!op->hasOneUse())
      return mlir::failure();
    auto opVectorType = op.getType();
    auto opElementTy = opVectorType.getElementType();
    // If the element type if < 32 bits, we need to clean up the packed layout
    // conversion op sequence.
    if (opElementTy.getIntOrFloatBitWidth() < 32) {
      // Check if the HW can support the load+transpose together.
      // TODO: add support for NON_PACKED usage for low-precsion.
      if (!canTranspose(loadOp, TransposeUsageType::PACKED))
        return mlir::failure();

      // Ceck for packed layout conversion op sequence.
      llvm::SmallVector<mlir::Operation *> packedLayoutOps;
      PackedLayoutOpsMatcher patternMatcher;
      if (!patternMatcher.match(*op->user_begin(), packedLayoutOps)) {
        return mlir::failure();
      }

      auto factor = 32 / opElementTy.getIntOrFloatBitWidth();
      // New output type has the transposed packed layout.
      auto newVectorTy =
          mlir::VectorType::get({opVectorType.getDimSize(0) / factor,
                                 opVectorType.getDimSize(1), factor},
                                opElementTy);
      // Create a new load op with transpose effect.
      auto packedAttr = mlir::UnitAttr(); // empty packed attribute.
      auto transposeAttr =
          mlir::DenseI64ArrayAttr::get(rewriter.getContext(), {1, 0});
      auto transposeBitWidthAttr = mlir::IntegerAttr::get(
          rewriter.getIntegerType(32),
          32); // need to do a 32 bit transpose to get the packed layout.
      auto newLoadOp = rewriter.create<mlir::xegpu::LoadNdOp>(
          loadOp.getLoc(), newVectorTy, loadOp.getTensorDesc(), packedAttr,
          transposeAttr, transposeBitWidthAttr, loadOp.getL1HintAttr(),
          loadOp.getL2HintAttr(), loadOp.getL3HintAttr());
      // Replace the uses of the packed layout conversion with new load.
      rewriter.replaceAllUsesWith(packedLayoutOps.back()->getResult(0),
                                  newLoadOp.getResult());
      // Remove the packed layout conversion op sequence in reverse order.
      for (auto packeLayoutOp : llvm::reverse(packedLayoutOps))
        rewriter.eraseOp(packeLayoutOp);
    }
    // If the element type is >= 32 bits, we can directly replace the
    // transpose.
    else {
      // Check if the HW can support the load+transpose together.
      if (!canTranspose(loadOp, TransposeUsageType::NON_PACKED))
        return mlir::failure();
      // New output type has the transposed shape.
      auto newVectorTy = mlir::VectorType::get(
          {opVectorType.getDimSize(0), opVectorType.getDimSize(1)},
          opElementTy);
      auto packedAttr = mlir::UnitAttr(); // empty packed attribute.
      auto transposeAttr =
          mlir::DenseI64ArrayAttr::get(rewriter.getContext(), {1, 0});
      auto newLoadOp = rewriter.create<mlir::xegpu::LoadNdOp>(
          loadOp.getLoc(), newVectorTy, loadOp.getTensorDesc(), packedAttr,
          transposeAttr, mlir::IntegerAttr(), loadOp.getL1HintAttr(),
          loadOp.getL2HintAttr(), loadOp.getL3HintAttr());
      rewriter.replaceAllUsesWith(op.getResult(), newLoadOp.getResult());
    }

    // Transpose op is dead. We can remove it.
    rewriter.eraseOp(op);
    // At this point, original load op is dead. We can remove it.
    if (loadOp->use_empty())
      rewriter.eraseOp(loadOp);
    return mlir::success();
  }
};

struct OptimizeTransposePass final
    : public imex::impl::OptimizeTransposeBase<OptimizeTransposePass> {
  OptimizeTransposePass() {
    uArchInterface = std::make_shared<imex::XePVCuArch>();
  }
  OptimizeTransposePass(const llvm::StringRef deviceName) {
    if (deviceName == "pvc") {
      uArchInterface = std::make_shared<imex::XePVCuArch>();
    }
  }
  mlir::LogicalResult
  initializeOptions(mlir::StringRef options,
                    mlir::function_ref<mlir::LogicalResult(const llvm::Twine &)>
                        errorHandler) override {
    if (failed(Pass::initializeOptions(options, errorHandler)))
      return mlir::failure();
    if (device == "pvc")
      uArchInterface = std::make_shared<imex::XePVCuArch>();
    else
      return errorHandler(llvm::Twine("Invalid device: ") + device);
    return mlir::success();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    LoadTransposeAnalysis analysis = getAnalysis<LoadTransposeAnalysis>();
    mlir::RewritePatternSet patterns(context);

    mlir::GreedyRewriteConfig config;
    config.enableRegionSimplification =
        mlir::GreedySimplifyRegionLevel::Disabled;
    config.useTopDownTraversal = true;
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingAndNewOps;
    patterns.add<TransposeRewritePattern>(context, analysis, uArchInterface);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }
  }

private:
  std::shared_ptr<imex::XeuArchInterface> uArchInterface = nullptr;
};
} // namespace optimizetranspose

std::unique_ptr<mlir::Pass>
imex::createOptimizeTransposePass(const std::string &deviceName) {
  return std::make_unique<optimizetranspose::OptimizeTransposePass>(deviceName);
}
