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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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

#define index_val(value)                                                       \
  rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(value))

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
      if (loadOp.getTensorDesc().getType().getMemorySpace() !=
          mlir::xegpu::MemorySpace::Global)
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

// Helper function to pack the given value in vnni format.
// to 32-bit representation. e.g., vector<8x8x2xf16> to vector<8x8xf32>
static mlir::Value pack(mlir::Value value, mlir::PatternRewriter &rewriter) {
  auto type = mlir::dyn_cast<mlir::VectorType>(value.getType());
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
  auto vecTy = mlir::VectorType::get({type.getNumElements()}, elemTy);
  value = rewriter.create<mlir::vector::ShapeCastOp>(loc, vecTy, value);

  // cast to 32-bit data, use i32 for intergers and f32 for floats.
  elemTy = elemTy.isInteger() ? (mlir::Type)rewriter.getIntegerType(32)
                              : (mlir::Type)rewriter.getF32Type();
  vecTy = mlir::VectorType::get({shape[0] * shape[1]}, elemTy);
  value = rewriter.create<mlir::vector::BitCastOp>(loc, vecTy, value);

  // cast to 2D shape
  vecTy = mlir::VectorType::get({shape[0], shape[1]}, elemTy);
  return rewriter.create<mlir::vector::ShapeCastOp>(loc, vecTy, value);
}

static void createStoreScatter(mlir::Value data, mlir::Value slm,
                               mlir::Value base,
                               mlir::PatternRewriter &rewriter) {
  auto type = mlir::dyn_cast<mlir::VectorType>(data.getType());
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
  auto addrTy = mlir::VectorType::get(simdLanes, base.getType());
  auto denseOffsets = mlir::DenseIntElementsAttr::get(addrTy, staticOffsets);
  mlir::Value offsets =
      rewriter.create<mlir::arith::ConstantOp>(loc, denseOffsets);
  base = rewriter.create<mlir::vector::BroadcastOp>(loc, addrTy, base);
  offsets = rewriter.create<mlir::arith::AddIOp>(loc, base, offsets);
  llvm::SmallVector<int64_t> tdescShape({simdLanes});
  if (chunkSize > 1)
    tdescShape.push_back(chunkSize);

  auto tdescTy = mlir::xegpu::TensorDescType::get(
      tdescShape, type.getElementType(), chunkSize,
      mlir::xegpu::MemorySpace::SLM);
  auto desc =
      rewriter.create<mlir::xegpu::CreateDescOp>(loc, tdescTy, slm, offsets);

  auto transposeAttr = rewriter.getUnitAttr();
  auto maskTy = mlir::VectorType::get(simdLanes, rewriter.getI1Type());
  auto mask = rewriter.create<mlir::arith::ConstantOp>(
      loc, mlir::DenseElementsAttr::get(maskTy, rewriter.getBoolAttr(true)));
  rewriter.create<mlir::xegpu::StoreScatterOp>(loc, data, desc, mask,
                                               transposeAttr, nullptr /*L1*/,
                                               nullptr /*L2*/, nullptr /*L3*/);
}

static mlir::Value createBlockLoad(mlir::TypedValue<mlir::MemRefType> slm,
                                   mlir::Value base, int numElems,
                                   mlir::Type slmElemTy, mlir::Type opElemTy,
                                   llvm::ArrayRef<int64_t> shape,
                                   mlir::PatternRewriter &rewriter) {
  auto loc = base.getLoc();
  // choose a maximum chunk size that can evenly divide numElems.
  std::vector<int> chunkSizes({64, 32, 16, 8, 4, 3, 2, 1});
  auto it = std::find_if(chunkSizes.begin(), chunkSizes.end(),
                         [&](int s) { return numElems % s == 0; });
  auto vectSize = *it;
  auto bitWidth = opElemTy.getIntOrFloatBitWidth();
  auto factor = bitWidth >= 32 ? 1 : 32 / bitWidth;
  auto numLoads = numElems / vectSize;
  auto tdescTy = mlir::xegpu::TensorDescType::get(
      vectSize, slmElemTy, 1, false, mlir::xegpu::MemorySpace::SLM);
  auto loadTy = mlir::VectorType::get(vectSize, slmElemTy);
  auto target1DTy = mlir::VectorType::get(vectSize * factor, opElemTy);
  auto target2DTy =
      mlir::VectorType::get({shape[0] / numLoads, shape[1]}, opElemTy);
  llvm::SmallVector<mlir::Value> loads;

  for (auto i = 0; i < numLoads; i++) {
    mlir::Value offset = rewriter.create<mlir::arith::AddIOp>(
        loc, base, index_val(i * vectSize));
    auto tdesc = rewriter.create<mlir::xegpu::CreateNdDescOp>(
        loc, tdescTy, slm, llvm::ArrayRef<mlir::OpFoldResult>({offset}));
    mlir::Value value = rewriter.create<mlir::xegpu::LoadNdOp>(
        loc, loadTy, tdesc, nullptr /*packed*/, nullptr /*transpose*/,
        nullptr /*transpose_bit_width*/, nullptr /*l1_hint*/,
        nullptr /*l2_hint*/, nullptr /*l3_hint*/);
    // if original data is not 32-bit, need to bitcast current 32-bit data
    //  back to original element type.
    if (bitWidth < 32)
      value = rewriter.create<mlir::vector::BitCastOp>(loc, target1DTy, value);

    // shape cast the value to 2D shape.
    value = rewriter.create<mlir::vector::ShapeCastOp>(loc, target2DTy, value);
    loads.push_back(value);
  }
  auto result = loads[0];
  for (size_t i = 1; i < loads.size(); i++) {
    result = imex::stack(result, loads[i], loc, rewriter);
  }
  return result;
}

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
    if (!loadOp || !loadOp->hasOneUse())
      return mlir::failure();

    auto opVectorType = op.getType();
    auto opElementTy = opVectorType.getElementType();

    // Only transposes that cannot be folded with the load op are considered.
    bool foldable = analysis.contains(loadOp) &&
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
                        ? (mlir::Type)rewriter.getIntegerType(32)
                        : (mlir::Type)rewriter.getF32Type();

      auto shape = tdescTy.getShape();
      // make sure each simd lane write a column. Ideally, it should be
      // less than 32. But IGC can split it if it into multiple instructions
      // if it is larger than 32.
      auto simdLanes = shape[1];
      // number of elements in 32-bit data.
      auto numElems = bytes / 4;
      // number of elements each simd lane to write
      int chunkSize = numElems / simdLanes;
      llvm::SmallVector<int> validChunkSizes = {64, 32, 16, 8, 4, 3, 2, 1};

      // the numElems has to be evenly divided by simdLanes, and the chunkSize
      // has to be in the validChunkSizes.
      if (numElems % simdLanes != 0 ||
          !llvm::is_contained(validChunkSizes, chunkSize))
        return mlir::failure();

      auto loc = loadOp.getLoc();
      auto data = loadOp.getResult();
      if (bitWidth < 32) {
        // vnni factor, the number of elements packed into a 32-bit data
        auto factor = 32 / bitWidth;
        // add vnni transformation to load op, and the result (data) type
        // will be updated from, e.g., vector<8x16xf16> to vector<4x16x2xf16>
        int64_t vnniShape[3] = {shape[0] / factor, shape[1], factor};
        auto vnniTy = mlir::VectorType::get(vnniShape, opElementTy);
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
      auto slmTy = mlir::MemRefType::get({totSlmSize}, elemTy, {}, 3);
      auto slm = rewriter.create<mlir::memref::AllocOp>(loc, slmTy);

      auto sgId = rewriter.create<mlir::gpu::SubgroupIdOp>(
          loc, rewriter.getIndexType(), nullptr /* upper_bound*/);
      auto offset = rewriter.create<mlir::arith::MulIOp>(
          loc, sgId, index_val(numElems), nullptr /* overflowFlags */);

      // store data using store_scatter to SLM at the given offset.
      createStoreScatter(data, slm, offset, rewriter);

      // load numElems elements from SLM at the given offset using 1D block
      // load.
      auto result = createBlockLoad(slm, offset, numElems, elemTy, opElementTy,
                                    opVectorType.getShape(), rewriter);
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    // trying to optimize the load+transpose+dpasB sequence.

    // Check if the transpose has a single user and it has desired packed
    // layout conversion op sequence.
    if (!op->hasOneUse())
      return mlir::failure();

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
