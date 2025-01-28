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
#include "imex/Utils/XeCommon.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
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
#include "llvm/ADT/SmallSet.h"
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

#define index_val(value)                                                       \
  rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(value))

using namespace mlir;

namespace optimizetranspose {

/// Attribute name that is used to annotate ops requiring array length
/// adjustment.
static const char *const adjustArrayLenAttrName =
    "__optimize-transpose-adjust-array-len__";

// Convenience interface for defining an op pattern.
struct PatternMatcherInterface {
public:
  virtual ~PatternMatcherInterface() = default;
  // Try to match the given op with some pattern and update the ops vector.
  virtual bool match(Operation *op,
                     llvm::SmallVectorImpl<Operation *> &ops) = 0;
};

// Pattern for detecting packed layout for DPAS B. We detect the following
// linear op sequence:
// clang-format off
// %0 = vector.shape_cast %in {packed}
// %1 = vector.shuffle %0, %0, %mask {packed}
// %2 = vector.shape_cast %1 {packed}
// clang-format on
struct PackedLayoutOpsMatcher : public PatternMatcherInterface {
  bool match(Operation *op, llvm::SmallVectorImpl<Operation *> &ops) override {
    // Check for first ShapeCastOp.
    auto shapeCastOp = llvm::dyn_cast_if_present<vector::ShapeCastOp>(op);
    if (!shapeCastOp || !shapeCastOp->hasAttr("packed"))
      return false;
    if (shapeCastOp.use_empty())
      return false;
    // ShapeCastOp should have a shuffle op as user.
    auto shuffleOp = llvm::dyn_cast_if_present<vector::ShuffleOp>(
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
    auto shapeCastOp2 = llvm::dyn_cast_if_present<vector::ShapeCastOp>(
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
  bool checkDPASBUsage(Operation *op) {
    // Op should have some users.
    if (op->use_empty())
      return false;
    // Now check all users are DPAS B usages.
    for (auto user : op->getUsers()) {
      auto dpasOp = llvm::dyn_cast_if_present<xegpu::DpasOp>(user);
      if (!dpasOp || dpasOp.getRhs().getDefiningOp() != op)
        return false;
    }
    return true;
  };

  // Helper to visit CreateNdDescOp and UpdateNdOffsetOp
  // and find all LoadNdOps that use it.
  void visitCreateNdDescOrUpdateNdOffsetOp(
      mlir::Operation *op, llvm::SmallVector<Operation *> &loadNdOpsFound) {
    llvm::SmallSet<Operation *, 8> worklist;
    worklist.insert(op);
    while (!worklist.empty()) {
      auto currOp = *worklist.begin();
      worklist.erase(currOp);
      // We found a LoadNdOp.
      if (auto loadNdOp = llvm::dyn_cast_if_present<xegpu::LoadNdOp>(currOp)) {
        loadNdOpsFound.push_back(loadNdOp);
      } else { // Process all users of the current op.
        for (auto user : currOp->getUsers()) {
          // If current user is a forOp, we need to get the block argument.
          if (auto forOp = llvm::dyn_cast_if_present<scf::ForOp>(user)) {
            auto opArgs = imex::getArgsForOperand(forOp, currOp->getResult(0));
            assert(opArgs.size() == 1 && "Duplicated tiles are not supported");
            auto blockArg = opArgs[0];
            for (auto user : blockArg.getUsers())
              worklist.insert(user);
          } else if (!llvm::isa<xegpu::UpdateNdOffsetOp>(user)) {
            worklist.insert(user);
          }
        }
      }
    }
  }

  // Helper to visit a LoadNdOp and checks if the result is consumed
  // exlusively by transposeOp(s). If so update the transposeOps vector.
  void visitLoadNdOp(xegpu::LoadNdOp loadNdOp,
                     llvm::SmallVector<Operation *> &transposeOps) {
    // If load op already has transpose effect, we skip it.
    auto transposeAttr = loadNdOp.getTransposeAttr();
    if (transposeAttr &&
        transposeAttr.asArrayRef() == llvm::ArrayRef<int64_t>{1, 0})
      return;
    // Memory space of the load op must be global.
    if (loadNdOp.getTensorDesc().getType().getMemorySpace() !=
        xegpu::MemorySpace::Global)
      return;
    // User of the load op must be either:
    // Case 1. A single transpose op.
    // Cast 2. vector.extract ops followed by a transpose ops.
    if (loadNdOp->hasOneUse()) {
      auto transposeOp = llvm::dyn_cast_if_present<vector::TransposeOp>(
          *loadNdOp->user_begin());
      if (transposeOp)
        transposeOps.push_back(transposeOp);
      return;
    }
    llvm::SmallVector<Operation *> localTransposeOps;
    for (auto user : loadNdOp->getUsers()) {
      auto extractOp = llvm::dyn_cast_if_present<vector::ExtractOp>(user);
      if (!extractOp)
        return;
      auto transposeOp = llvm::dyn_cast_if_present<vector::TransposeOp>(
          *extractOp->user_begin());
      if (!extractOp->hasOneUse() || !transposeOp)
        return;
      localTransposeOps.push_back(transposeOp);
    }
    // If all users are vector.extract followed by transpose ops, so update the
    // result.
    transposeOps.insert(transposeOps.end(), localTransposeOps.begin(),
                        localTransposeOps.end());
    return;
  }

  // Helper fuunction to visit the def-use chain of the transposeOp. Traveral is
  // confined to a single block. So it must terminate. Traversal try to
  // check if there exisit a partial DAG with source as this transposeOp and all
  // sinks are DPAS B usages.
  void visitTransposeOp(vector::TransposeOp transposeOp,
                        llvm::DenseSet<Operation *> &leaves) {
    // Check for packed layout pattern. shape_cast -> shuffle -> shape_cast.
    // Otherwise, skip.
    llvm::SmallVector<Operation *> packedLayoutOps;
    PackedLayoutOpsMatcher patternMatcher;
    if (!patternMatcher.match(*transposeOp->user_begin(), packedLayoutOps))
      return;

    llvm::DenseSet<Operation *> worklist;
    worklist.insert(transposeOp);
    Block *parentBlock = transposeOp->getBlock();
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
  }
  // Analysis result.
  llvm::DenseSet<Operation *> fusionCandidates;
  llvm::DenseSet<Operation *> arrayLenAdjustmentCandidates;

public:
  LoadTransposeAnalysis(Operation *op) {
    op->walk([&](mlir::Operation *targetOp) -> WalkResult {
      if (!llvm::isa<xegpu::CreateNdDescOp>(targetOp) &&
          !llvm::isa<xegpu::UpdateNdOffsetOp>(targetOp))
        return WalkResult::skip();

      llvm::SmallVector<Operation *> loadNdOpsFound;
      // Find all LoadNdOps that use this CreateNdDescOp.
      visitCreateNdDescOrUpdateNdOffsetOp(targetOp, loadNdOpsFound);
      // If no LoadNdOps or more than one LoadNdOps are found, we skip.
      if (loadNdOpsFound.size() != 1)
        return WalkResult::skip();

      xegpu::LoadNdOp loadNdOp =
          llvm::dyn_cast<xegpu::LoadNdOp>(loadNdOpsFound[0]);
      // Visit the LoadNdOp and check if load result is consumed by transpose.
      llvm::SmallVector<Operation *> transposeOps;
      visitLoadNdOp(loadNdOp, transposeOps);
      // If no transpose ops are found, we skip.
      if (transposeOps.empty())
        return WalkResult::skip();

      // If the load element type is >= 32 bits, we can directly consider it.
      auto opElementTy = loadNdOp.getTensorDesc().getType().getElementType();
      if (opElementTy.getIntOrFloatBitWidth() >= 32) {
        fusionCandidates.insert(loadNdOp);
        return WalkResult::advance();
      }
      // Traverse the def-use chain of the transposeOps and find leaf
      // operations.
      llvm::DenseSet<Operation *> leaves;
      for (auto transposeOp : transposeOps)
        visitTransposeOp(llvm::cast<vector::TransposeOp>(transposeOp), leaves);
      // If not leaf nodes are found, skip.
      if (leaves.empty())
        return WalkResult::skip();
      // Check if all leaves of the DAG are DPAS ops.
      for (auto leaf : leaves) {
        if (!llvm::isa<xegpu::DpasOp>(leaf)) {
          return WalkResult::skip();
        }
      }
      // At this point, we have found a LoadNdOp with desired DPAS usage.
      fusionCandidates.insert(loadNdOp);
      // Source CreateNdDescOp is considered for array length adjustment if
      // array_length > 1.
      auto createNdDescOp = llvm::dyn_cast<xegpu::CreateNdDescOp>(targetOp);
      if (createNdDescOp &&
          createNdDescOp.getTensorDesc().getType().getArrayLength() > 1)
        arrayLenAdjustmentCandidates.insert(createNdDescOp);
      return WalkResult::advance();
    });
  }
  // Check if a given LoadNdOp should be considered for fusion.
  bool isCandidateForFusion(xegpu::LoadNdOp op) {
    return fusionCandidates.contains(op);
  }
  // Check if a give CreateNdDescOp should be considered for array length
  // adjustment.
  bool isCandidateForArrayLenAdjustment(xegpu::CreateNdDescOp op) {
    return arrayLenAdjustmentCandidates.contains(op);
  }
  // Print the analysis result.
  void printAnalysisResult() {
    llvm::errs() << "LoadTransposeAnalysis Result:\n";
    llvm::errs() << "LoadNdOp Candidates:\n";
    for (auto op : fusionCandidates) {
      op->print(llvm::errs());
      llvm::errs() << "\n";
    }
    llvm::errs() << "CreateNdDescOp Candidates:\n";
    for (auto op : arrayLenAdjustmentCandidates) {
      op->print(llvm::errs());
      llvm::errs() << "\n";
    }
  }
};

// Types of usages for the transpose. PACKED is used for DPAS B usage.
// NON_PACKED represents any other usage.
enum TransposeUsageType { PACKED = 1, NON_PACKED = 2 };

// Helper function to pack the given value in vnni format.
// to 32-bit representation. e.g., vector<8x8x2xf16> to vector<8x8xf32>
static Value pack(Value value, PatternRewriter &rewriter) {
  auto type = dyn_cast<VectorType>(value.getType());
  if (!type || type.getRank() != 3)
    return value;
  auto elemTy = type.getElementType();
  if (!elemTy.isIntOrFloat())
    return value;

  auto shape = type.getShape();
  auto factor = shape[2];
  auto bits = elemTy.getIntOrFloatBitWidth();
  if (factor * bits != 32)
    return value;

  auto loc = value.getLoc();
  // first cast the value to 1D shape
  auto vecTy = VectorType::get({type.getNumElements()}, elemTy);
  value = rewriter.create<vector::ShapeCastOp>(loc, vecTy, value);

  // cast to 32-bit data, use i32 for intergers and f32 for floats.
  elemTy = elemTy.isInteger() ? (Type)rewriter.getIntegerType(32)
                              : (Type)rewriter.getF32Type();
  vecTy = VectorType::get({shape[0] * shape[1]}, elemTy);
  value = rewriter.create<vector::BitCastOp>(loc, vecTy, value);

  // cast to 2D shape
  vecTy = VectorType::get({shape[0], shape[1]}, elemTy);
  return rewriter.create<vector::ShapeCastOp>(loc, vecTy, value);
}

static void createStoreScatter(Value data, Value slm, Value base,
                               PatternRewriter &rewriter) {
  auto type = dyn_cast<VectorType>(data.getType());
  if (!type || type.getRank() > 2)
    return;

  auto loc = data.getLoc();
  auto shape = type.getShape();
  auto chunkSize = type.getRank() == 2 ? shape[0] : 1;
  auto simdLanes = type.getRank() == 2 ? shape[1] : shape[0];

  llvm::SmallVector<int64_t> staticOffsets;
  for (auto i = 0; i < simdLanes; i++) {
    staticOffsets.push_back(i * chunkSize);
  }
  auto addrTy = VectorType::get(simdLanes, base.getType());
  auto denseOffsets = DenseIntElementsAttr::get(addrTy, staticOffsets);
  Value offsets = rewriter.create<arith::ConstantOp>(loc, denseOffsets);
  base = rewriter.create<vector::BroadcastOp>(loc, addrTy, base);
  offsets = rewriter.create<arith::AddIOp>(loc, base, offsets);
  llvm::SmallVector<int64_t> tdescShape({simdLanes});
  if (chunkSize > 1)
    tdescShape.push_back(chunkSize);

  auto tdescTy = xegpu::TensorDescType::get(tdescShape, type.getElementType(),
                                            chunkSize, xegpu::MemorySpace::SLM);
  auto desc = rewriter.create<xegpu::CreateDescOp>(loc, tdescTy, slm, offsets);

  auto transposeAttr = rewriter.getUnitAttr();
  auto maskTy = VectorType::get(simdLanes, rewriter.getI1Type());
  auto mask = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(maskTy, rewriter.getBoolAttr(true)));
  rewriter.create<xegpu::StoreScatterOp>(loc, data, desc, mask, transposeAttr,
                                         nullptr /*L1*/, nullptr /*L2*/,
                                         nullptr /*L3*/);
}

static Value createBlockLoad(TypedValue<MemRefType> slm, Value base,
                             int numElems, Type slmElemTy, Type opElemTy,
                             llvm::ArrayRef<int64_t> shape,
                             PatternRewriter &rewriter) {
  auto loc = base.getLoc();
  // choose a maximum chunk size that can evenly divide numElems.
  std::vector<int> chunkSizes({64, 32, 16, 8, 4, 3, 2, 1});
  auto it = std::find_if(chunkSizes.begin(), chunkSizes.end(),
                         [&](int s) { return numElems % s == 0; });
  auto vectSize = *it;
  auto bitWidth = opElemTy.getIntOrFloatBitWidth();
  auto factor = bitWidth >= 32 ? 1 : 32 / bitWidth;
  auto numLoads = numElems / vectSize;
  auto tdescTy = xegpu::TensorDescType::get(vectSize, slmElemTy, 1, false,
                                            xegpu::MemorySpace::SLM);
  auto loadTy = VectorType::get(vectSize, slmElemTy);
  auto target1DTy = VectorType::get(vectSize * factor, opElemTy);
  auto target2DTy = VectorType::get({shape[0] / numLoads, shape[1]}, opElemTy);
  llvm::SmallVector<Value> loads;

  for (auto i = 0; i < numLoads; i++) {
    Value offset =
        rewriter.create<arith::AddIOp>(loc, base, index_val(i * vectSize));
    auto tdesc = rewriter.create<xegpu::CreateNdDescOp>(
        loc, tdescTy, slm, llvm::ArrayRef<OpFoldResult>({offset}));
    Value value = rewriter.create<xegpu::LoadNdOp>(
        loc, loadTy, tdesc, nullptr /*packed*/, nullptr /*transpose*/,
        nullptr /*transpose_bit_width*/, nullptr /*l1_hint*/,
        nullptr /*l2_hint*/, nullptr /*l3_hint*/);
    // if original data is not 32-bit, need to bitcast current 32-bit data
    //  back to original element type.
    if (bitWidth < 32)
      value = rewriter.create<vector::BitCastOp>(loc, target1DTy, value);

    // shape cast the value to 2D shape.
    value = rewriter.create<vector::ShapeCastOp>(loc, target2DTy, value);
    loads.push_back(value);
  }
  auto result = loads[0];
  for (size_t i = 1; i < loads.size(); i++) {
    result = imex::stack(result, loads[i], loc, rewriter);
  }
  return result;
}

//// ------- Array Length Adjustment Rewrite Patterns --------------------- ////

// This pattern splits CreateNdDescOp with array_length > 1 into multiple ops
// with array_length = 1. Pattern is only applicable if the analysis result
// indicates that the op is a candidate for array length adjustment.
struct CreateNdDescOpPattern
    : public OpConversionPattern<xegpu::CreateNdDescOp> {
  using OpConversionPattern<xegpu::CreateNdDescOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::CreateNdDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tdescTy = op.getTensorDesc().getType();
    if (tdescTy.getArrayLength() == 1)
      return failure();

    if (!op->hasAttr(adjustArrayLenAttrName))
      return failure();
    // TODO: Currently, we only support array length adjustment for static
    // memrefs.
    auto memrefTypedSource = cast<TypedValue<MemRefType>>(op.getSource());
    if (!memrefTypedSource || !memrefTypedSource.getType().hasStaticShape())
      return failure();

    // If the array_length is greater than 1, split the tensor desc into
    // multiple tensor descs.
    int64_t arrayLength = tdescTy.getArrayLength();
    llvm::SmallVector<Value> createNdDescOps;
    auto newTdescTy = xegpu::TensorDescType::get(
        tdescTy.getShape(), tdescTy.getElementType(), /*array_length=*/1,
        tdescTy.getBoundaryCheck(), tdescTy.getMemorySpace(),
        tdescTy.getSgMap());
    auto origOffsetY = op.getOffsets().back();
    for (int64_t i = 0; i < arrayLength; ++i) {
      auto attr = rewriter.getIndexAttr(i * tdescTy.getShape()[1]);
      auto offsetIncrementY = rewriter.create<mlir::arith::ConstantOp>(
          op->getLoc(), rewriter.getIndexType(), attr);
      // Y offset is adjusted based on col dim size.
      auto offsetY = rewriter.create<mlir::arith::AddIOp>(
          op->getLoc(), origOffsetY, offsetIncrementY);
      SmallVector<OpFoldResult> offsets =
          op.getOffsets().drop_back(); // offsets excluding Y offset.
      offsets.push_back(offsetY->getResult(0));
      auto newOp = rewriter.create<xegpu::CreateNdDescOp>(
          op.getLoc(), newTdescTy, memrefTypedSource, offsets);
      createNdDescOps.push_back(newOp);
    }

    // Create UnrealizedConversionCastOp to reconcile the types.
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, op.getType(),
                                                            createNdDescOps);

    return success();
  }
};

// If the source tile of a LoadNdOp is split into multiple tiles due to array
// length adjustment, We split the loadNdOp into multiple loadNdOps.
struct LoadNdOpPattern : public OpConversionPattern<xegpu::LoadNdOp> {
  using OpConversionPattern<xegpu::LoadNdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::LoadNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tdesc = adaptor.getTensorDesc();
    auto unrealizedCastOp =
        llvm::dyn_cast_if_present<UnrealizedConversionCastOp>(
            tdesc.getDefiningOp());
    if (!unrealizedCastOp)
      return failure();
    auto sources = tdesc.getDefiningOp()->getOperands();
    llvm::SmallVector<Value> loadNdOps;
    auto newLoadTy = VectorType::get(op.getTensorDescType().getShape(),
                                     op.getType().getElementType());
    for (auto source : sources) {
      auto loadNdOp = rewriter.create<xegpu::LoadNdOp>(
          op.getLoc(), newLoadTy, source, op.getPackedAttr(),
          op.getTransposeAttr(), op.getTransposeBitWidthAttr(),
          op.getL1HintAttr(), op.getL2HintAttr(), op.getL3HintAttr());
      loadNdOps.push_back(loadNdOp);
    }
    // Create UnrealizedConversionCastOp to reconcile the types.
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, op.getType(),
                                                            loadNdOps);
    return success();
  }
};

// Array length adjustment can result in a change of ForOp inputs, output and
// signature of the ForOp body. This pattern creates a new ForOp with the
// updated inputs, output and signature.
struct ScfForOpPattern : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> newIterArgs; // Keep track of new iter args.
    llvm::DenseMap<unsigned, std::pair<unsigned, unsigned>>
        forOpOutputTypeMapping; // Keep track old iter
                                // args indice to new iter args indice/s mapping
    for (auto [index, initArg] : llvm::enumerate(adaptor.getInitArgs())) {
      // If initArg is a UnrealizedConversionCastOp, new iter args need to be
      // updated. And we keep track of the old -> new iter arg indice mapping.
      if (auto unrealizedConversionCast =
              llvm::dyn_cast_if_present<UnrealizedConversionCastOp>(
                  initArg.getDefiningOp())) {
        auto sources = unrealizedConversionCast.getOperands();
        forOpOutputTypeMapping[index] =
            std::make_pair(newIterArgs.size(), sources.size());
        newIterArgs.insert(newIterArgs.end(), sources.begin(), sources.end());
      } else {
        forOpOutputTypeMapping[index] = std::make_pair(newIterArgs.size(), 1);
        newIterArgs.push_back(initArg);
      }
    }

    // Create a new ForOp with the expanded iter args.
    auto newForOp = rewriter.create<scf::ForOp>(
        op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(),
        ValueRange(newIterArgs));
    // We don't need the empty blocks created by rewriter.
    rewriter.eraseBlock(newForOp.getBody());

    Region *region = &op.getRegion();
    Block *block = &region->front();
    // Perform signature conversion on the body block of the original forOp.
    OneToNTypeMapping signatureConverter(block->getArgumentTypes());

    for (size_t i = 0; i < block->getNumArguments(); i++) {
      // If iter args are expanded due to UnrealizedConversionCastOp, we need
      // 1:N type mapping in the signature converter.
      if (i >= 1 && forOpOutputTypeMapping[i - 1].second > 1) {
        auto sources =
            adaptor.getInitArgs()[i - 1].getDefiningOp()->getOperands();
        auto sourceTypes = llvm::map_to_vector(
            sources, [](Value source) { return source.getType(); });
        signatureConverter.addInputs(i, sourceTypes);
      } else {
        signatureConverter.addInputs(i, block->getArgument(i).getType());
      }
    }
    // Apply the signature conversion to the block.
    rewriter.applySignatureConversion(block, signatureConverter,
                                      getTypeConverter());
    // Splice the old body region into the new for-op.
    Region &dstRegion = newForOp.getBodyRegion();
    rewriter.inlineRegionBefore(op.getRegion(), dstRegion, dstRegion.end());

    // Compute the new results for the ForOp. If the iter args are expanded, the
    // result types also need to be resolved by adding UnrealizedConversionCast
    // ops.
    llvm::SmallVector<Value> newResults;
    for (auto [index, result] : llvm::enumerate(op.getResults())) {
      auto newResultPosAndSize = forOpOutputTypeMapping[index];
      auto sources = newForOp.getResults().slice(newResultPosAndSize.first,
                                                 newResultPosAndSize.second);
      // If this result is expanded, reconcile the types using
      // UnrealizedConversionCastOp.
      if (sources.size() > 1) {
        auto unrealizedCastOp = rewriter.create<UnrealizedConversionCastOp>(
            op.getLoc(), result.getType(), sources);
        newResults.push_back(unrealizedCastOp.getResult(0));
      } else {
        newResults.push_back(sources[0]);
      }
    }
    // Create UnrealizedConversionCastOp to reconcile the types.
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, op.getResultTypes(), newResults);
    return success();
  }
};

// Adjust the inputs of the YieldOp based on the new ForOp signature.
struct ScfTYieldOpPattern : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> newOperands;
    for (auto operand : adaptor.getOperands()) {
      if (auto unrealizedConversionCast =
              llvm::dyn_cast_if_present<UnrealizedConversionCastOp>(
                  operand.getDefiningOp())) {
        auto sources = unrealizedConversionCast.getOperands();
        newOperands.insert(newOperands.end(), sources.begin(), sources.end());
      } else {
        newOperands.push_back(operand);
      }
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, newOperands);
    return success();
  }
};

// Vector ExtractOp is used to extract the individual vector register tiles from
// a loadNdOp with array_length > 1. If the array length is adjusted to 1, we no
// longer need to extract the individual tiles. This pattern replaces the
// ExtractOp with corresponding vector register tiles.
struct VectorExrtactOpPattern final
    : public OpConversionPattern<vector::ExtractOp> {
  using OpConversionPattern<vector::ExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto source = adaptor.getVector();
    auto unrealizedCastOp =
        llvm::dyn_cast_if_present<UnrealizedConversionCastOp>(
            source.getDefiningOp());
    if (!unrealizedCastOp)
      return failure();

    auto sources = source.getDefiningOp()->getOperands();
    auto extractIndex = op.getStaticPosition();
    assert(extractIndex.size() == 1 && extractIndex[0] != ShapedType::kDynamic);
    rewriter.replaceOp(op, sources[extractIndex[0]]);
    return success();
  }
};

// If the source tile of UpdateNdOffsetOp is split into multiple tiles due to
// array length adjustment, We split the UpdateNdOffsetOp into multiple
// UpdateNdOffsetOps.
struct UpdateNdOffsetOpPattern final
    : public OpConversionPattern<xegpu::UpdateNdOffsetOp> {
  using OpConversionPattern<xegpu::UpdateNdOffsetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::UpdateNdOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tdesc = adaptor.getTensorDesc();
    auto unrealizedCastOp =
        llvm::dyn_cast_if_present<UnrealizedConversionCastOp>(
            tdesc.getDefiningOp());
    if (!unrealizedCastOp)
      return failure();
    auto sources = unrealizedCastOp.getOperands();
    auto offsets = op.getMixedOffsets();
    llvm::SmallVector<Value> dynamicOffsets;
    llvm::SmallVector<int64_t> staticOffsets;
    dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
    llvm::SmallVector<Value> newOps;
    for (auto source : sources) {
      auto newOp = rewriter.create<xegpu::UpdateNdOffsetOp>(
          op.getLoc(), source.getType(), source, dynamicOffsets, staticOffsets);
      newOps.push_back(newOp);
    }
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, op.getType(),
                                                            newOps);
    return success();
  }
};

//// ------- Load + Transpose Optimization Rewrite Patterns --------------- ////

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
struct TransposeRewritePattern : public OpRewritePattern<vector::TransposeOp> {
  TransposeRewritePattern(MLIRContext *context, LoadTransposeAnalysis &analysis,
                          std::shared_ptr<imex::XeuArchInterface> ptruArch)
      : OpRewritePattern<vector::TransposeOp>(context), analysis(analysis),
        uArchInterface(ptruArch) {}
  LoadTransposeAnalysis &analysis;
  std::shared_ptr<imex::XeuArchInterface> uArchInterface;

  // Check if the target HW allows doing the load+transpose together.
  bool canTranspose(xegpu::LoadNdOp loadOp,
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
    if (!load2DConfig->blockHeight.contains(blockH) ||
        !load2DConfig->blockWidth.contains(blockW))
      return false;
    // Check if the array length is supported by uArch.
    int arrayLen = tdescTy.getArrayLength();
    return llvm::any_of(load2DConfig->array_length,
                        [&](int len) { return len == arrayLen; });
  }

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    // Check if this tranpose is using a load op.
    auto loadOp = llvm::dyn_cast_if_present<xegpu::LoadNdOp>(
        op.getVector().getDefiningOp());
    if (!loadOp || !loadOp->hasOneUse())
      return failure();

    auto opVectorType = op.getType();
    auto opElementTy = opVectorType.getElementType();

    // Only transposes that cannot be folded with the load op are considered.
    bool foldable = analysis.isCandidateForFusion(loadOp) &&
                    (canTranspose(loadOp, TransposeUsageType::PACKED) ||
                     canTranspose(loadOp, TransposeUsageType::NON_PACKED));

    if (!foldable && !loadOp.getPacked() && opElementTy.isIntOrFloat()) {
      // try to optimize the load+transpose sequence only using SLM.
      // It covers the cases of 8-bit/16-bit data types, and hardware
      // unsupported shapes of 32-bit data types, e.g., <8x32xf32>.
      auto tdescTy = op.getSourceVectorType();
      auto bitWidth = opElementTy.getIntOrFloatBitWidth();
      auto bytes = tdescTy.getNumElements() * bitWidth / 8;

      // limite the total data size <= 512 bytes, which is maximum size
      // can be handled by a single load/store lsc intrinsic.
      if (bytes > 512)
        return rewriter.notifyMatchFailure(
            op, "total data size is larger than 512 bytes.");

      // Element type for SLM, all operations to slm are done in 32-bit
      // or 64-bit granularity.
      auto elemTy = bitWidth >= 32 ? opElementTy
                    : opElementTy.isInteger()
                        ? (Type)rewriter.getIntegerType(32)
                        : (Type)rewriter.getF32Type();

      auto shape = tdescTy.getShape();
      // make sure each simd lane write a column. Ideally, it should be
      // less than 32. But IGC can split it if it into multiple instructions
      // if it is larger than 32.
      auto simdLanes = shape[1];
      // number of elements in 32-bit data.
      auto numElems = bytes / 4;
      // number of elements each simd lane to write
      int chunkSize = numElems / simdLanes;

      // the numElems has to be evenly divided by simdLanes, and the chunkSize
      // has to be in the validChunkSizes.
      auto validChunkSizes = imex::getSupportedChunkSizes(simdLanes);
      if (numElems % simdLanes != 0 ||
          !llvm::is_contained(validChunkSizes, chunkSize))
        return failure();

      auto loc = loadOp.getLoc();
      auto data = loadOp.getResult();
      if (bitWidth < 32) {
        // vnni factor, the number of elements packed into a 32-bit data
        auto factor = 32 / bitWidth;
        // add vnni transformation to load op, and the result (data) type
        // will be updated from, e.g., vector<8x16xf16> to vector<4x16x2xf16>
        int64_t vnniShape[3] = {shape[0] / factor, shape[1], factor};
        auto vnniTy = VectorType::get(vnniShape, opElementTy);
        loadOp.setPacked(true);
        loadOp.getResult().setType(vnniTy);

        // pack the result into 32-bit format, e.g., vector<4x16x2xf16> to
        // vector<4x16xf32>
        data = pack(loadOp.getResult(), rewriter);
      }

      // alloc a shared local memory for the data. Note that SLM is shared among
      // subgroups in a workgroup. The total size needed is numElems *
      // numSubgroups. However, currently dynamic allocation is not supported,
      // so we assume maximum number of subgroups is 64, considering that the
      // PVC has 8 EUs per subslice, and 8 threads per EU.
      // TODO: get the number from uArch.
      int64_t totSlmSize = 64 * numElems;
      auto slmTy = MemRefType::get({totSlmSize}, elemTy, {}, 3);
      auto slm = rewriter.create<memref::AllocOp>(loc, slmTy);

      auto sgId = rewriter.create<gpu::SubgroupIdOp>(
          loc, rewriter.getIndexType(), nullptr /* upper_bound*/);
      auto offset = rewriter.create<arith::MulIOp>(
          loc, sgId, index_val(numElems), nullptr /* overflowFlags */);

      // store data using store_scatter to SLM at the given offset.
      createStoreScatter(data, slm, offset, rewriter);

      // load numElems elements from SLM at the given offset using 1D block
      // load.
      auto result = createBlockLoad(slm, offset, numElems, elemTy, opElementTy,
                                    opVectorType.getShape(), rewriter);
      rewriter.replaceOp(op, result);
      return success();
    }

    // trying to optimize the load+transpose+dpasB sequence.

    // Check if the transpose has a single user and it has desired packed
    // layout conversion op sequence.
    if (!op->hasOneUse())
      return failure();

    // If the element type if < 32 bits, we need to clean up the packed layout
    // conversion op sequence.
    if (opElementTy.getIntOrFloatBitWidth() < 32) {
      // Check if the HW can support the load+transpose together.
      // TODO: add support for NON_PACKED usage for low-precsion.
      if (!canTranspose(loadOp, TransposeUsageType::PACKED))
        return failure();

      // Check for packed layout conversion op sequence.
      llvm::SmallVector<Operation *> packedLayoutOps;
      PackedLayoutOpsMatcher patternMatcher;
      if (!patternMatcher.match(*op->user_begin(), packedLayoutOps)) {
        return failure();
      }

      auto factor = 32 / opElementTy.getIntOrFloatBitWidth();
      // New output type has the transposed packed layout.
      auto newVectorTy = VectorType::get({opVectorType.getDimSize(0) / factor,
                                          opVectorType.getDimSize(1), factor},
                                         opElementTy);
      // Create a new load op with transpose effect.
      auto packedAttr = UnitAttr(); // empty packed attribute.
      auto transposeAttr =
          DenseI64ArrayAttr::get(rewriter.getContext(), {1, 0});
      auto transposeBitWidthAttr = IntegerAttr::get(
          rewriter.getIntegerType(32),
          32); // need to do a 32 bit transpose to get the packed layout.
      auto newLoadOp = rewriter.create<xegpu::LoadNdOp>(
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
        return failure();
      // New output type has the transposed shape.
      auto newVectorTy = VectorType::get(
          {opVectorType.getDimSize(0), opVectorType.getDimSize(1)},
          opElementTy);
      auto packedAttr = UnitAttr(); // empty packed attribute.
      auto transposeAttr =
          DenseI64ArrayAttr::get(rewriter.getContext(), {1, 0});
      auto newLoadOp = rewriter.create<xegpu::LoadNdOp>(
          loadOp.getLoc(), newVectorTy, loadOp.getTensorDesc(), packedAttr,
          transposeAttr, IntegerAttr(), loadOp.getL1HintAttr(),
          loadOp.getL2HintAttr(), loadOp.getL3HintAttr());
      rewriter.replaceAllUsesWith(op.getResult(), newLoadOp.getResult());
    }

    // Transpose op is dead. We can remove it.
    rewriter.eraseOp(op);
    // At this point, original load op is dead. We can remove it.
    if (loadOp->use_empty())
      rewriter.eraseOp(loadOp);
    return success();
  }
};

struct OptimizeTransposePass final
    : public imex::impl::OptimizeTransposeBase<OptimizeTransposePass> {
private:
  std::shared_ptr<imex::XeuArchInterface> uArchInterface = nullptr;

  bool markOpsForArrayLenAdjustment(LoadTransposeAnalysis &analysis) {
    auto *op = getOperation();
    bool changed = false;
    std::function<void(Operation *)> markOp = [&](Operation *op) {
      op->setAttr(adjustArrayLenAttrName, UnitAttr::get(op->getContext()));
      // At one point, we hit vector.extract op that do the array length
      // extraction. After this no more adjustment is needed.
      if (llvm::isa<vector::ExtractOp>(op))
        return;
      // // Users of forOps are already added.
      // if (llvm::isa<scf::ForOp>(op))
      // return;
      // Mark all its users.
      for (auto user : op->getUsers()) {
        // ForOp requires special handling.
        if (auto forOp = llvm::dyn_cast_if_present<scf::ForOp>(user)) {
          // `op` can be passed in as a block argument to the forOp. So mark all
          // users of the block arg inside forOp body.
          auto opArgs = imex::getArgsForOperand(forOp, op->getResult(0));
          assert(opArgs.size() == 1 && "Duplicated tiles are not supported");
          auto blockArg = opArgs[0];
          for (auto blockArgUser : blockArg.getUsers())
            markOp(blockArgUser);
          // Also, do the same for the corresponding result of the forOp.
          // auto result = forOp->getResult(blockArg.getArgNumber() -
          //                                forOp.getNumInductionVars());
          // for (auto resultUser : result.getUsers())
          //   markOp(resultUser);
        }
        markOp(user);
      }
    };
    op->walk([&](xegpu::CreateNdDescOp createNdDescOp) -> WalkResult {
      if (analysis.isCandidateForArrayLenAdjustment(createNdDescOp)) {
        markOp(createNdDescOp);
        changed = true;
      }
      return WalkResult::advance();
    });
    return changed;
  }

  void runArrayLengthAdjustment() {
    Operation *module = getOperation();
    TypeConverter typeConverter;
    auto addNToOneCast = [](OpBuilder &builder, Type type, ValueRange inputs,
                            Location loc) {
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      return cast.getResult(0);
    };
    typeConverter.addSourceMaterialization(addNToOneCast);
    typeConverter.addArgumentMaterialization(addNToOneCast);
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    // Only rewrite ops marked for array length adjustment.
    target
        .addDynamicallyLegalOp<xegpu::CreateNdDescOp, scf::ForOp, scf::YieldOp,
                               xegpu::LoadNdOp, xegpu::UpdateNdOffsetOp>(
            [&](Operation *op) -> bool {
              return op->getAttr(adjustArrayLenAttrName) == mlir::Attribute();
            });
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalDialect<arith::ArithDialect>();
    patterns.add<CreateNdDescOpPattern, LoadNdOpPattern, ScfForOpPattern,
                 ScfTYieldOpPattern, VectorExrtactOpPattern,
                 UpdateNdOffsetOpPattern>(&getContext());
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return signalPassFailure();

    // At this point, some of the original ops may still be using the results of
    // UnrealizedConversionCastOps. So clean them up. For Example, some post ops
    // (arith.* ops) or store ops may still be using the results of adjusted
    // ForOp results through UnrealizedConversionCastOps.
    module->walk([](Operation *op) {
      // Ignore UnrealizedConversionCastOps that were inserted by array length
      // adjustment. These will be cleaned up by DCE.
      if (llvm::isa<UnrealizedConversionCastOp>(op))
        return;
      llvm::SmallVector<Value> newOperands;
      bool operandsUpdated = false;
      for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
        if (auto unrealizedConversionCast =
                llvm::dyn_cast_if_present<UnrealizedConversionCastOp>(
                    operand.getDefiningOp())) {
          // This UnrealizedConversionCastOp is a dummy pass through added by
          // our array langth adjustment patterns i.e. It preserves the input
          // and output types.
          assert(unrealizedConversionCast.getOperandTypes() ==
                     unrealizedConversionCast.getResultTypes() &&
                 "unexpected unrealized conversion cast found as operand");
          // Get the corresponding result index for this operand.
          auto resultIndex =
              imex::getResultIndex(unrealizedConversionCast, operand);
          // Collect the operand from the corresponding result index. This
          // should be our new operand.
          newOperands.push_back(
              unrealizedConversionCast.getOperand(resultIndex));
          operandsUpdated = true;
        } else {
          newOperands.push_back(operand);
        }
      }
      // If no operands are updated, we skip.
      if (!operandsUpdated)
        return;
      // Update the operands.
      op->setOperands(newOperands);
      op->removeAttr(adjustArrayLenAttrName);
    });

    // Clean up the UnrealizedConversionCastOps that are not used.
    // TODO: Maybe we don't need this if CSE is run after this pass.
    module->walk([](mlir::UnrealizedConversionCastOp castOp) {
      if (castOp->use_empty())
        castOp.erase();
    });
  }

  void runLoadTransposeFusion(LoadTransposeAnalysis &analysis) {
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    GreedyRewriteConfig config;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
    config.useTopDownTraversal = true;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    patterns.add<TransposeRewritePattern>(context, analysis, uArchInterface);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }

public:
  void runOnOperation() override {
    LoadTransposeAnalysis analysis = getAnalysis<LoadTransposeAnalysis>();
    // analysis.printAnalysisResult();
    bool needAdjustment = markOpsForArrayLenAdjustment(analysis);
    if (needAdjustment) {
      runArrayLengthAdjustment();
      // Re-run the analysis after array length adjustment.
      getAnalysisManager().clear();
      analysis = getAnalysis<LoadTransposeAnalysis>();
      // analysis.printAnalysisResult();
    }
    runLoadTransposeFusion(analysis);
  }
  OptimizeTransposePass() {
    uArchInterface = std::make_shared<imex::XePVCuArch>();
  }
  OptimizeTransposePass(const llvm::StringRef deviceName) {
    if (deviceName == "pvc") {
      uArchInterface = std::make_shared<imex::XePVCuArch>();
    }
  }
  LogicalResult initializeOptions(
      StringRef options,
      function_ref<LogicalResult(const llvm::Twine &)> errorHandler) override {
    if (failed(Pass::initializeOptions(options, errorHandler)))
      return failure();
    if (device == "pvc")
      uArchInterface = std::make_shared<imex::XePVCuArch>();
    else
      return errorHandler(llvm::Twine("Invalid device: ") + device);
    return success();
  }
};
} // namespace optimizetranspose

std::unique_ptr<Pass>
imex::createOptimizeTransposePass(const std::string &deviceName) {
  return std::make_unique<optimizetranspose::OptimizeTransposePass>(deviceName);
}
