//===-------------- Blocking.cpp --------- Blocking Pass  -------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains lowering transformation for determing the problem size
/// that can be handled by an XeGPU operator (hardware instruction). XeTile
/// program can work one bigger problem size that cannot be handled by a
/// hardware instruction. But it needs to be decomposed into smaller pieces
/// such that each pieces can be handled by a hardware instruction.
///
//===----------------------------------------------------------------------===//
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#include <optional>
#include <tuple>

#include "imex/Dialect/XeTile/Transforms/BlockingAnalysis.h"
#include "imex/Dialect/XeTile/Transforms/Passes.h"
#include "imex/Utils/XeArch.h"

#define DEBUG_TYPE "xetile-blocking"

using namespace mlir;
using namespace llvm;
using namespace imex;
namespace imex {
#define GEN_PASS_DEF_XETILEBLOCKING
#include "imex/Dialect/XeTile/Transforms/Passes.h.inc"
} // namespace imex

namespace imex {
// Blocking is to decompose ops working on big tile or vector size
// into a set of ops working on smaller tile or vector size, that
// can be mapped to hardware instructions. The old implementation
// is using a 4D tile/vector type to represent the blocking result.
// with the outer 2 dimensions corresponding to the grid size or
// the number of instructions and their organization, while the
// inner 2 dimensions corresponding to the block size, which can
// be handled by a single instruction. The new implementation is
// to remove this 4D tile/vector type representation and generating
// a set of xetile or vector ops working the block size directly.
namespace Blocking {

template <typename SourceOp, typename AnalysisT>
class RewriteXeTileOp : public mlir::OpRewritePattern<SourceOp> {
public:
  using OpPatternRewriter = typename mlir::PatternRewriter;

  RewriteXeTileOp(mlir::MLIRContext *context, AnalysisT &analysis)
      : mlir::OpRewritePattern<SourceOp>(context), analysis(analysis) {}

protected:
  AnalysisT &analysis;
};

template <template <typename> class TraitType, typename AnalysisT>
class RewriteOpWithTrait : public mlir::OpTraitRewritePattern<TraitType> {
public:
  using OpPatternRewriter = typename mlir::PatternRewriter;

  RewriteOpWithTrait(mlir::MLIRContext *context, AnalysisT &analysis,
                     PatternBenefit benefit = 1)
      : mlir::OpTraitRewritePattern<TraitType>(context, benefit),
        analysis(analysis) {}

protected:
  AnalysisT &analysis;
};

static const char *const packAttrName = "__xetile_blocking_pack__";
static const char *const unpackAttrName = "__xetile_blocking_unpack__";
static const char *const blockAttrName = "__xetile_blocking_inner_block__";

static mlir::Value addUnpackOp(mlir::ValueRange srcs, mlir::Type destTy,
                               llvm::ArrayRef<int64_t> innerBlock,
                               mlir::Location loc,
                               mlir::PatternRewriter &rewriter) {
  auto attr = mlir::NamedAttribute(rewriter.getStringAttr(unpackAttrName),
                                   rewriter.getUnitAttr());
  auto innerBlkAttr =
      mlir::NamedAttribute(rewriter.getStringAttr(blockAttrName),
                           rewriter.getDenseI64ArrayAttr(innerBlock));
  auto castOp = rewriter.create<mlir::UnrealizedConversionCastOp>(
      loc, destTy, srcs,
      llvm::ArrayRef<mlir::NamedAttribute>({attr, innerBlkAttr}));
  return castOp.getResult(0);
}

static mlir::ValueRange addPackOp(mlir::Value src, mlir::TypeRange destTypes,
                                  llvm::ArrayRef<int64_t> innerBlock,
                                  mlir::Location loc,
                                  mlir::PatternRewriter &rewriter) {
  auto attr = mlir::NamedAttribute(rewriter.getStringAttr(packAttrName),
                                   rewriter.getUnitAttr());
  auto innerBlkAttr =
      mlir::NamedAttribute(rewriter.getStringAttr(blockAttrName),
                           rewriter.getDenseI64ArrayAttr(innerBlock));
  auto castOp = rewriter.create<mlir::UnrealizedConversionCastOp>(
      loc, destTypes, src,
      llvm::ArrayRef<mlir::NamedAttribute>({attr, innerBlkAttr}));
  return castOp.getResults();
}

static bool isPackOp(mlir::UnrealizedConversionCastOp castOp) {
  if (!castOp)
    return false;
  bool isVec = llvm::all_of(castOp->getResultTypes(), [](mlir::Type ty) {
    return mlir::isa<mlir::VectorType>(ty);
  });
  isVec &= llvm::all_of(castOp->getOperandTypes(), [](mlir::Type ty) {
    return mlir::isa<mlir::VectorType>(ty);
  });
  auto attr = castOp->getAttrOfType<mlir::UnitAttr>(packAttrName);
  return isVec && bool(attr);
}

static bool isUnpackOp(mlir::UnrealizedConversionCastOp castOp) {
  if (!castOp)
    return false;
  bool isVec = llvm::all_of(castOp->getResultTypes(), [](mlir::Type ty) {
    return mlir::isa<mlir::VectorType>(ty);
  });
  isVec &= llvm::all_of(castOp->getOperandTypes(), [](mlir::Type ty) {
    return mlir::isa<mlir::VectorType>(ty);
  });
  auto attr = castOp->getAttrOfType<mlir::UnitAttr>(unpackAttrName);
  return isVec && bool(attr);
}

static std::pair<mlir::DenseI64ArrayAttr, mlir::DenseI64ArrayAttr>
getGridAndBlockSizes(mlir::UnrealizedConversionCastOp castOp) {
  assert((isUnpackOp(castOp) || isPackOp(castOp)) &&
         "Expecting unpack or pack op.");
  auto innerBlkSizes =
      castOp->getAttrOfType<mlir::DenseI64ArrayAttr>(blockAttrName);
  llvm::ArrayRef<int64_t> shape;
  if (isUnpackOp(castOp))
    shape = mlir::dyn_cast<mlir::ShapedType>(castOp->getResult(0).getType())
                .getShape();

  if (isPackOp(castOp))
    shape = mlir::dyn_cast<mlir::ShapedType>(castOp->getOperand(0).getType())
                .getShape();

  auto grids = mlir::DenseI64ArrayAttr::get(
      castOp.getContext(),
      {shape[0] / innerBlkSizes[0], shape[1] / innerBlkSizes[1]});
  return {grids, innerBlkSizes};
}

// Check that lowerUnpackOrPack will be able to evenly combine/split the input
// grid into the output grid.
static bool isUnpackPackCompatible(mlir::UnrealizedConversionCastOp unpackOp,
                                   mlir::UnrealizedConversionCastOp packOp) {

  if (!isUnpackOp(unpackOp) || !isPackOp(packOp))
    return false;

  auto [inGrids, inBlkSizes] = Blocking::getGridAndBlockSizes(unpackOp);
  auto [outGrids, outBlkSizes] = Blocking::getGridAndBlockSizes(packOp);

  if (inBlkSizes[0] < outBlkSizes[0] && inGrids[0] % outGrids[0] != 0)
    return false;
  if (inBlkSizes[0] > outBlkSizes[0] && outGrids[0] % inGrids[0] != 0)
    return false;
  if (inBlkSizes[1] < outBlkSizes[1] && inGrids[1] % outGrids[1] != 0)
    return false;
  if (inBlkSizes[1] > outBlkSizes[1] && outGrids[1] % inGrids[1] != 0)
    return false;

  return true;
}

// Create a BinOp on lhs and rhs based on the CombiningKind.
static mlir::Value createBinOp(mlir::vector::CombiningKind kind,
                               mlir::Value lhs, mlir::Value rhs,
                               mlir::Location &loc,
                               mlir::PatternRewriter &rewriter) {
  assert(lhs.getType() == rhs.getType() && "Expecting same type.");
  auto elemTy = mlir::getElementTypeOrSelf(lhs);
  // ADD and MUL are defined for both Integers and Floats,
  // need to generate code based on element data type.
  if (kind == mlir::vector::CombiningKind::ADD) {
    if (mlir::isa<mlir::FloatType>(elemTy)) {
      return rewriter.create<mlir::arith::AddFOp>(loc, lhs, rhs);
    }
    if (mlir::isa<mlir::IntegerType>(elemTy)) {
      return rewriter.create<mlir::arith::AddIOp>(loc, lhs, rhs);
    }
  }

  if (kind == mlir::vector::CombiningKind::MUL) {
    if (mlir::isa<mlir::FloatType>(elemTy)) {
      return rewriter.create<mlir::arith::MulFOp>(loc, lhs, rhs);
    }
    if (mlir::isa<mlir::IntegerType>(elemTy)) {
      return rewriter.create<mlir::arith::MulIOp>(loc, lhs, rhs);
    }
  }

  switch (kind) {
  // the following are for ints only
  case mlir::vector::CombiningKind::MINUI:
    return rewriter.create<mlir::arith::MinUIOp>(loc, lhs, rhs);
  case mlir::vector::CombiningKind::MINSI:
    return rewriter.create<mlir::arith::MinSIOp>(loc, lhs, rhs);
  case mlir::vector::CombiningKind::MAXUI:
    return rewriter.create<mlir::arith::MaxUIOp>(loc, lhs, rhs);
  case mlir::vector::CombiningKind::MAXSI:
    return rewriter.create<mlir::arith::MaxSIOp>(loc, lhs, rhs);
  case mlir::vector::CombiningKind::AND:
    return rewriter.create<mlir::arith::AndIOp>(loc, lhs, rhs);
  case mlir::vector::CombiningKind::OR:
    return rewriter.create<mlir::arith::OrIOp>(loc, lhs, rhs);
  case mlir::vector::CombiningKind::XOR:
    return rewriter.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
  // the following are for floats only
  case mlir::vector::CombiningKind::MINNUMF:
    return rewriter.create<mlir::arith::MinNumFOp>(loc, lhs, rhs);
  case mlir::vector::CombiningKind::MAXNUMF:
    return rewriter.create<mlir::arith::MaxNumFOp>(loc, lhs, rhs);
  case mlir::vector::CombiningKind::MINIMUMF:
    return rewriter.create<mlir::arith::MinimumFOp>(loc, lhs, rhs);
  case mlir::vector::CombiningKind::MAXIMUMF:
    return rewriter.create<mlir::arith::MaximumFOp>(loc, lhs, rhs);
  default:
    llvm_unreachable("Unexpected CombiningKind.");
    return lhs;
  }
}

// helper function to lower the outer reduction,
// e.g., tile.reduce <add> %src [0]: vector<32x64xf16> to vector<1x64xf16>
// The blocking size for such reduction op is not fixed to 1x16. So the
// `sources` is a vector of values with type of vector<1x16xf16>, organized
// as grid [32, 2]. So this function will perform 31 reduction operations
// on grid[:, 0], and grid[:, 1] respectively
static llvm::SmallVector<mlir::Value>
lowerOuterReduction(mlir::ValueRange sources, llvm::ArrayRef<int64_t> grid,
                    mlir::vector::CombiningKind kind, mlir::Location loc,
                    mlir::PatternRewriter &rewriter) {
  llvm::SmallVector<mlir::Value> results;
  for (auto j = 0; j < grid[1]; j++) {
    auto val = sources[j];
    for (auto i = 1; i < grid[0]; i++) {
      val = createBinOp(kind, val, sources[i * grid[1] + j], loc, rewriter);
    }
    auto shapedTy = mlir::dyn_cast<mlir::ShapedType>(val.getType());
    // needs one reduction is block size is not 1 for the reduction dim.
    if (shapedTy && shapedTy.getDimSize(0) != 1) {
      auto shape = shapedTy.getShape().vec();
      shape[0] = 1;
      auto resTy = shapedTy.clone(shape);
      val = rewriter.create<xetile::ReductionOp>(loc, resTy, kind, val,
                                                 llvm::ArrayRef<int64_t>({0}));
    }
    results.push_back(val);
  }
  return results;
}

// expected inputs are a grid of vector<1xnxf16> values. The grid shape is
// [i, j]. i and n is power of 2 and the third dim is always 1, which should be
// set by the blocking pass. For a vector of vector<32x64xf16> with reduction
// on dim 1, it will blocked into a vector<32x4x1x16> with reduction on dim 1
// and dim 3. lowerInnerReductionWithIntraVectorShuffles performs the reduction
// with arithmetic operations on vector<16xf16>. To perform reduction on dim 1,
// simple vector arithmetic operations are issued, we will get 32 vectors of
// vector<16xf16>, each vector<16xf16> represents the partial reduction result
// of each row. To perform redcution on dim 3, it uses two vector shuffles
/// to shuffle values from two conjuction rows. For example, given
// row1 = [a0, a1, ..., a15], and  row2 = [b0, b1, ..., b15]. It will shuffle
// the vector into row1' = [a0, .., a7, b0, ..., b7],
// row2' = [a8, ..., a15, b8, ..., b15], and then perform the vector arith op
// on row1' and row2', geting the result: c = [c0, ..., c7, c8, ..., c15].
// here, c0, ..., c7 are the partial reduction results of row1 and c8, ..., c15
// are the partial results of row2.  This process will be repeated until get the
// final result, such that each element in c represents a final reduction result
// of a row.
static llvm::SmallVector<mlir::Value>
lowerInnerReductionWithIntraVectorShuffles(
    mlir::ValueRange sources, mlir::Type elemTy, llvm::ArrayRef<int64_t> grid,
    llvm::ArrayRef<int64_t> block, mlir::vector::CombiningKind kind,
    mlir::Location loc, mlir::PatternRewriter &rewriter) {

  auto isPowerOfTwo = [](auto n) { return (n & (n - 1)) == 0; };

  // make sure the dim0 of the block is 1 in blocking pass
  // different from outer reduction, this is strictly required
  // for this method.
  assert(block[0] == 1 && "dim0 of the block has to be 1.");
  assert(isPowerOfTwo(grid[0]) && isPowerOfTwo(block[1]) &&
         "sizes of dim1 of grid and block should be power of 2.");

  auto genShuffleMasks = [&](int blkSize, int vecSize) {
    llvm::SmallVector<int64_t> mask1;
    llvm::SmallVector<int64_t> mask2;
    auto s1 = 0, s2 = blkSize;
    for (auto i = 0; i < vecSize; i++) {
      if (i && i % blkSize == 0) {
        s1 += blkSize;
        s2 += blkSize;
      }

      mask1.push_back(s1);
      mask2.push_back(s2);
      s1++;
      s2++;
    }
    return std::make_pair(mask1, mask2);
  };

  // Stage 1: vector<ixjx1xnxf16> equals to a grid of ixj of vector<1xnxf16>
  // after lowering to xegpu. This stage performs j-1 reduction operations on
  // j dim of the grid, the result is a vector of vector<ixnxf16>.
  llvm::SmallVector<mlir::Value> intermediates(grid[0]);
  for (auto i = 0; i < grid[0]; i++) {
    auto val = sources[i * grid[1]];
    for (auto j = 1; j < grid[1]; j++) {
      val = createBinOp(kind, val, sources[i * grid[1] + j], loc, rewriter);
    }
    // cast the result of e.g., vector<1x16xf16> into vector<16xf16>
    auto targetTy = mlir::VectorType::get({block[1]}, elemTy);
    val = rewriter.create<mlir::vector::ShapeCastOp>(loc, targetTy, val);
    intermediates[i] = val;
  }

  // Stage 2: doing intra vector reduction with shuffle Ops.
  // Each vector in the result of stage 1 can be viewed as a row
  // each row has e.g., 32 elements:
  // v1 = [a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 ... a31]
  // v2 = [b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 ... b31]
  // ...
  // vn = [p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 ... p31]
  // To reduce it, we repeatedly shuffle halves of two consecutive vectors.
  // One can view it as: transpose halves of two partial aggregates, reduce
  // vertically, get 1 vector with reduced halves of two vectors. For example,
  // for v1 and v2, we get:
  //    nv1 = [a0, .., a15, b0, .., b15]
  //    nv2 = [a16, .., a31, b16, .., b31]
  //    nv_reduced = reductionOp(nv1,nv2)
  // such that the left half of the vector contains the partial reduction
  // of v1, and the right half contains the partial reduction of v2.
  // and the the number of vectors is reduced by half after one iteration.
  // and we reduce the block size by half, and repeat the process until
  // the block size is 1.
  // The intermediate result of this stage is an array of vectors with
  // type, e.g., vector<nxf16>, array size is `i/n`. And these vectors
  // will be merged into a single vector with type vector<ixf16>.

  // each row should not have > 1 partial aggregate at the end
  auto partialRowAggSize{block[1]};
  auto numVecsLeft{grid[0]};
  while (partialRowAggSize != 1 && numVecsLeft != 1) {
    partialRowAggSize /= 2;
    auto workList = intermediates;
    intermediates.clear();
    assert(workList.size() % 2 == 0 && "The size should be divisible by 2.");
    auto masks = genShuffleMasks(partialRowAggSize, block[1]);
    for (size_t i = 0; i < workList.size(); i += 2) {
      auto v1 = workList[i];
      auto v2 = workList[i + 1];
      auto shuffleOp1 =
          rewriter.create<mlir::vector::ShuffleOp>(loc, v1, v2, masks.first);
      auto shuffleOp2 =
          rewriter.create<mlir::vector::ShuffleOp>(loc, v1, v2, masks.second);
      auto reduce = createBinOp(kind, shuffleOp1, shuffleOp2, loc, rewriter);
      intermediates.push_back(reduce);
    }
    numVecsLeft /= 2;
  }

  if (partialRowAggSize > 1) {
    assert(intermediates.size() == 1 &&
           "We must have ONE row with non-finalized aggregates.");
    auto toFinalize = intermediates.back();
    intermediates.clear();
    uint32_t currentAggVecSize = block[1];
    do {
      currentAggVecSize /= 2;
      partialRowAggSize /= 2;
      auto [vecUpperMask, vecLowerMask] =
          genShuffleMasks(partialRowAggSize, currentAggVecSize);
      auto shuffleOp1 = rewriter.create<mlir::vector::ShuffleOp>(
          loc, toFinalize, toFinalize, vecUpperMask);
      auto shuffleOp2 = rewriter.create<mlir::vector::ShuffleOp>(
          loc, toFinalize, toFinalize, vecLowerMask);
      toFinalize = createBinOp(kind, shuffleOp1, shuffleOp2, loc, rewriter);
    } while (partialRowAggSize != 1);
    intermediates.push_back(toFinalize);
  }
  return intermediates;
}

// a unified function lowering Unpack and Pack ops.
static llvm::SmallVector<mlir::Value>
lowerUnpackOrPack(mlir::ValueRange inputs, mlir::DenseI64ArrayAttr inBlkSizes,
                  mlir::DenseI64ArrayAttr outBlkSizes,
                  mlir::DenseI64ArrayAttr inGrids,
                  mlir::DenseI64ArrayAttr outGrids, mlir::Location loc,
                  mlir::OpBuilder &builder) {
  // handle based on the dim0, and save results into intermediates
  llvm::SmallVector<mlir::Value> intermediates(outGrids[0] * inGrids[1]);
  if (inBlkSizes[0] == outBlkSizes[0]) { // do nothing
    intermediates = inputs;
  } else if (inBlkSizes[0] < outBlkSizes[0]) { // stack on dim 0
    // `nums` small vectors will be stacked into one big vector
    auto nums = inGrids[0] / outGrids[0];
    llvm::SmallVector<mlir::Value> valSet;
    for (auto j = 0; j < inGrids[1]; j++) {
      for (auto i = 0; i < inGrids[0]; i++) {
        auto idx = i * inGrids[1] + j;
        valSet.push_back(inputs[idx]);
        if (valSet.size() == static_cast<size_t>(nums)) {
          auto newOp = packVectorsWith(valSet, stack, loc, builder);
          intermediates[i / nums * inGrids[1] + j] = newOp;
          valSet.clear();
        }
      }
    }
  } else {
    // do extract on dim0 using vector::ExtractStridedSliceOp
    // intermediates.resize(outGrids[0] * inGrids[1]);
    llvm::SmallVector<int64_t> blkSizes({outBlkSizes[0], inBlkSizes[1]});

    // each vector will be horizonally cut into `nums` subvectors
    auto nums = outGrids[0] / inGrids[0];
    llvm::SmallVector<int64_t> strides({1, 1});
    for (auto i = 0; i < inGrids[0]; i++) {
      for (auto j = 0; j < inGrids[1]; j++) {
        auto startPos = i * nums * inGrids[1] + j;
        auto v = inputs[i * inGrids[1] + j];
        for (auto k = 0; k < nums; k++) {
          llvm::SmallVector<int64_t> offsets({k * blkSizes[0], 0});
          auto newOp = builder.create<mlir::vector::ExtractStridedSliceOp>(
              loc, v, offsets, blkSizes, strides);
          auto idx = startPos + k * inGrids[1];
          intermediates[idx] = newOp;
        }
      }
    }
  }

  // handle intermediates based on the dim1, and save results into newOps
  llvm::SmallVector<mlir::Value> newOps;
  llvm::SmallVector<int64_t> interGrids = {outGrids[0], inGrids[1]};
  if (inBlkSizes[1] == outBlkSizes[1]) {
    // do nothing since they have the same size
    newOps = intermediates;
  } else if (inBlkSizes[1] < outBlkSizes[1]) {
    // doing concat since blkSZ of input vector is smaller
    // `nums` of small vectors will be concated into a big one
    size_t nums = inGrids[1] / outGrids[1];
    llvm::SmallVector<mlir::Value> valSet;
    for (auto i = 0; i < interGrids[0]; i++) {
      for (auto j = 0; j < interGrids[1]; j++) {
        valSet.push_back(intermediates[i * interGrids[1] + j]);
        if (valSet.size() == nums) {
          auto newOp = packVectorsWith(valSet, concat, loc, builder);
          newOps.push_back(newOp);
          valSet.clear();
        }
      }
    }
  } else { // doing extract on dim 1
    llvm::SmallVector<int64_t> blkSizes({outBlkSizes[0], outBlkSizes[1]});
    llvm::SmallVector<int64_t> strides({1, 1});
    auto nums = outGrids[1] / interGrids[1];
    for (auto i = 0; i < interGrids[0]; i++) {
      for (auto j = 0; j < interGrids[1]; j++) {
        auto v = intermediates[i * interGrids[1] + j];
        for (int64_t k = 0; k < nums; k++) {
          llvm::SmallVector<int64_t> offsets({0, k * blkSizes[1]});
          auto newOp = builder.create<mlir::vector::ExtractStridedSliceOp>(
              loc, v, offsets, blkSizes, strides);
          newOps.push_back(newOp);
        }
      }
    }
  }
  return newOps;
}

static llvm::SmallVector<mlir::Type>
convertTypes(mlir::ShapedType type, llvm::ArrayRef<int64_t> blockSize) {
  auto newTy = type.clone(blockSize, type.getElementType());
  auto size = std::accumulate(blockSize.begin(), blockSize.end(), 1,
                              std::multiplies<int64_t>());
  return llvm::SmallVector<mlir::Type>(type.getNumElements() / size, newTy);
}

// clang-format off
// rewrite a arith.constant op on big sizes into multiple arith.constant ops on
// smaller sizes, which is determined by the blocking analysis. For example,
// %0 = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7]]>: vector<2x4xf16>,
// will be replaced by:
// %0_0 = arith.constant dense<[[0, 1], [4, 5]]>: vector<2x2xf16>
// %0_1 = arith.constant dense<[[2, 3], [6, 7]]>: vector<2x2xf16>
// assuming the blocking size is [2, 2].
// clang-format on

class RewriteArithConstantOp
    : public RewriteXeTileOp<mlir::arith::ConstantOp, BlockingAnalysis> {
public:
  using RewriteXeTileOp<mlir::arith::ConstantOp,
                        BlockingAnalysis>::RewriteXeTileOp;
  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    bool allUsersAreUnrealizedCastOp =
        llvm::all_of(op->getUsers(), [](auto user) {
          auto castOp = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(user);
          auto packAttr = user->getAttr(packAttrName);
          return castOp && packAttr;
        });
    // currently only handles the case where the constant op is used by
    // an unrealized cast op.
    if (allUsersAreUnrealizedCastOp) {
      auto value = llvm::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
      if (!value || value.getType().getRank() != 2)
        return mlir::failure();

      auto blockSize = analysis.getDefBlockSize(op.getResult());
      if (!blockSize)
        return mlir::failure();

      auto shape = value.getType().getShape();
      auto elemTy = value.getType().getElementType();
      auto newTy = mlir::VectorType::get(blockSize.asArrayRef(), elemTy);
      auto values = value.getValues<mlir::Attribute>();

      llvm::SmallVector<mlir::Value> newOps;
      for (auto i = 0; i < shape[0]; i += blockSize[0]) {
        for (auto j = 0; j < shape[1]; j += blockSize[1]) {
          llvm::SmallVector<mlir::Attribute> subValues;
          for (auto x = 0; x < blockSize[0]; x++) {
            for (auto y = 0; y < blockSize[1]; y++) {
              subValues.push_back(values[(i + x) * shape[1] + j + y]);
            }
          }
          auto subValue = mlir::DenseElementsAttr::get(newTy, subValues);
          auto newOp = rewriter.create<mlir::arith::ConstantOp>(loc, subValue);
          newOps.push_back(newOp);
        }
      }
      auto castOp = addUnpackOp(newOps, value.getType(), blockSize.asArrayRef(),
                                loc, rewriter);
      rewriter.replaceOp(op, castOp);
      return mlir::success();
    }
    return mlir::failure();
  }
};

// rewrite a init_tile op on big size into multiple init_tile ops on smaller
// size, which is based on blocking analysis.
class RewriteInitTileOp
    : public RewriteXeTileOp<xetile::InitTileOp, BlockingAnalysis> {
public:
  using RewriteXeTileOp<xetile::InitTileOp, BlockingAnalysis>::RewriteXeTileOp;

  mlir::LogicalResult
  matchAndRewrite(xetile::InitTileOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = op.getContext();
    auto tileTy = op.getType();
    auto shape = tileTy.getShape();

    auto blockSize = analysis.getDefBlockSize(op.getTile());
    // skip it if there is no valid blockSize available, or the
    // tile is already with the target size.
    if (!blockSize || shape == blockSize.asArrayRef())
      return mlir::failure();

    llvm::SmallVector<mlir::Value> newOps;
    // handle scattered tiles.
    if (tileTy.getScatterAttr() == mlir::BoolAttr::get(ctx, true)) {
      auto indices = op.getIndices();
      assert(indices && "indices is missing.");
      auto indicesTy = indices.getType();

      auto convertedTileTypes = convertTypes(tileTy, blockSize.asArrayRef());
      auto newIndicesTypes = convertTypes(indicesTy, blockSize.asArrayRef());

      auto subIndices = addPackOp(indices, newIndicesTypes,
                                  blockSize.asArrayRef(), loc, rewriter);

      for (auto [t, i] : llvm::zip(convertedTileTypes, subIndices)) {
        llvm::SmallVector<mlir::Value> operands({op.getSource(), i});
        auto newOp = rewriter.create<xetile::InitTileOp>(
            loc, mlir::TypeRange({t}), operands, op->getAttrs());
        newOps.push_back(newOp);
      }
    } else { // handle blocked tiles
      auto newTileTy = tileTy.clone(blockSize.asArrayRef());
      // TODO: add array_length support.
      auto width = blockSize[1];
      llvm::SmallVector<int64_t, 2> grids(
          {shape[0] / blockSize[0], shape[1] / width});
      auto mixedOffsets = op.getMixedOffsets();

      auto addi = [&](mlir::OpFoldResult a, int64_t b) -> mlir::Value {
        if (mlir::isa<mlir::Attribute>(a)) {
          auto attr = a.get<mlir::Attribute>();
          auto sum =
              rewriter.getIndexAttr(cast<IntegerAttr>(attr).getInt() + b);
          return rewriter.create<mlir::arith::ConstantOp>(loc, sum);
        } else {
          auto aV = a.get<mlir::Value>();
          auto bV = rewriter.create<mlir::arith::ConstantOp>(
              loc, rewriter.getIndexAttr(b));
          return rewriter.create<mlir::arith::AddIOp>(loc, aV, bV);
        }
      };

      // For n-D memrefs where n > 2, we need to handle the last two
      // dimensions, and keep the first n-2 dimensions as is.
      int64_t x = mixedOffsets.size() - 2;
      int64_t y = mixedOffsets.size() - 1;
      mlir::OpFoldResult oldX = mixedOffsets[x];
      mlir::OpFoldResult oldY = mixedOffsets[y];

      for (int64_t i = 0; i < grids[0]; i++) {
        for (int64_t j = 0; j < grids[1]; j++) {
          auto subOffX = blockSize[0] * i;
          auto subOffY = width * j;
          mixedOffsets[x] = addi(oldX, subOffX);
          mixedOffsets[y] = addi(oldY, subOffY);
          llvm::SmallVector<mlir::Value> offsets;
          llvm::SmallVector<int64_t> constOffsets;
          mlir::dispatchIndexOpFoldResults(mixedOffsets, offsets, constOffsets);
          auto constOffsetsAttr = rewriter.getDenseI64ArrayAttr(constOffsets);
          auto newOp = rewriter.create<xetile::InitTileOp>(
              loc, newTileTy, op.getSource(), offsets, op.getSizes(),
              op.getStrides(), constOffsetsAttr, op.getConstSizesAttr(),
              op.getConstStridesAttr(), nullptr);
          newOps.push_back(newOp);
        }
      }
    }
    auto castOp =
        addUnpackOp(newOps, tileTy, blockSize.asArrayRef(), loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return mlir::success();
  }
};

// rewrite a prefetch_tile op on big tile size into multiple prefetch_tile ops
// on smaller tile size, which is based on blocking analysis.
class RewritePrefetchTileOp
    : public RewriteXeTileOp<xetile::PrefetchTileOp, BlockingAnalysis> {
public:
  using RewriteXeTileOp<xetile::PrefetchTileOp,
                        BlockingAnalysis>::RewriteXeTileOp;

  mlir::LogicalResult
  matchAndRewrite(xetile::PrefetchTileOp op,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tile = op.getTile();
    auto tileTy = tile.getType();
    auto shape = tileTy.getShape();
    auto blockSize = analysis.getUseBlockSize(tile, op->getOpOperand(0));
    // define op is not updated yet.
    if (!blockSize || shape == blockSize.asArrayRef())
      return failure();
    auto convertedTileTypes = convertTypes(tileTy, blockSize.asArrayRef());
    auto convertedTiles = addPackOp(tile, convertedTileTypes,
                                    blockSize.asArrayRef(), loc, rewriter);

    for (auto t : convertedTiles) {
      rewriter.create<xetile::PrefetchTileOp>(loc, mlir::TypeRange(), t,
                                              op->getAttrs());
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// rewrite a load_tile op on big tile size into multiple load_tile ops
// on smaller tile size, which is based on blocking analysis.
class RewriteLoadTileOp
    : public RewriteXeTileOp<xetile::LoadTileOp, BlockingAnalysis> {
public:
  using RewriteXeTileOp<xetile::LoadTileOp, BlockingAnalysis>::RewriteXeTileOp;

  mlir::LogicalResult
  matchAndRewrite(xetile::LoadTileOp op,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tile = op.getSource();
    auto tileTy = tile.getType();
    auto shape = tileTy.getShape();
    auto blockSize = analysis.getUseBlockSize(tile, op->getOpOperand(0));

    if (!blockSize || shape == blockSize.asArrayRef())
      return failure();

    auto convertedTileTypes = convertTypes(tileTy, blockSize.asArrayRef());
    auto convertedTiles = addPackOp(tile, convertedTileTypes,
                                    blockSize.asArrayRef(), loc, rewriter);

    auto vecTy = ::mlir::VectorType::get(blockSize.asArrayRef(),
                                         tileTy.getElementType());

    llvm::SmallVector<mlir::Value> newOps;
    for (auto t : convertedTiles) {
      auto newOp =
          rewriter.create<xetile::LoadTileOp>(loc, vecTy, t, op->getAttrs());
      newOps.push_back(newOp);
    }

    auto castOp = addUnpackOp(newOps, op.getType(), blockSize.asArrayRef(), loc,
                              rewriter);

    rewriter.replaceOp(op, castOp);
    return mlir::success();
  }
};

// rewrite a store_tile op on big tile size into multiple store_tile ops
// on smaller tile size, which is based on blocking analysis.
class RewriteStoreTileOp
    : public RewriteXeTileOp<xetile::StoreTileOp, BlockingAnalysis> {
public:
  using RewriteXeTileOp<xetile::StoreTileOp, BlockingAnalysis>::RewriteXeTileOp;

  mlir::LogicalResult
  matchAndRewrite(xetile::StoreTileOp op,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto value = op.getValue();
    auto valTy = value.getType();
    auto shape = valTy.getShape();
    auto tile = op.getTile();
    auto tileTy = tile.getType();
    auto blockSize = analysis.getUseBlockSize(value, op->getOpOperand(0));

    if (!blockSize || shape == blockSize.asArrayRef())
      return failure();

    auto convertedValTypes = convertTypes(valTy, blockSize.asArrayRef());
    auto convertedTileTypes = convertTypes(tileTy, blockSize.asArrayRef());
    auto convertedValues = addPackOp(value, convertedValTypes,
                                     blockSize.asArrayRef(), loc, rewriter);
    auto convertedTiles = addPackOp(tile, convertedTileTypes,
                                    blockSize.asArrayRef(), loc, rewriter);

    for (auto [v, t] : llvm::zip(convertedValues, convertedTiles)) {
      rewriter.create<xetile::StoreTileOp>(loc, v, t, op.getL1HintAttr(),
                                           op.getL2HintAttr(),
                                           op.getL3HintAttr());
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// rewrite a LoadGatherOp on big tile size into multiple LoadGatherOps
// on smaller tile size.
class RewriteLoadGatherOp
    : public RewriteXeTileOp<xetile::LoadGatherOp, BlockingAnalysis> {
public:
  using RewriteXeTileOp<xetile::LoadGatherOp,
                        BlockingAnalysis>::RewriteXeTileOp;

  mlir::LogicalResult
  matchAndRewrite(xetile::LoadGatherOp op,
                  PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto mask = op.getMask();
    auto tile = op.getTile();
    auto type = tile.getType();
    auto elemTy = type.getElementType();

    auto blockSize = analysis.getUseBlockSize(tile, op->getOpOperand(0));
    if (!blockSize || type.getShape() == blockSize.asArrayRef())
      return mlir::failure();

    auto convertedTileTypes = convertTypes(type, blockSize.asArrayRef());
    auto convertedMaskTypes =
        convertTypes(mask.getType(), blockSize.asArrayRef());

    auto tiles = addPackOp(tile, convertedTileTypes, blockSize.asArrayRef(),
                           loc, rewriter);
    auto masks = addPackOp(mask, convertedMaskTypes, blockSize.asArrayRef(),
                           loc, rewriter);
    auto newValueTy = mlir::VectorType::get(blockSize.asArrayRef(), elemTy);
    llvm::SmallVector<mlir::Value> newOps;
    for (auto [t, m] : llvm::zip(tiles, masks)) {
      auto newOp = rewriter.create<xetile::LoadGatherOp>(
          loc, newValueTy, t, m, op.getPaddingAttr(), op.getL1HintAttr(),
          op.getL2HintAttr(), op.getL3HintAttr());
      newOps.push_back(newOp);
    }

    auto castOp = addUnpackOp(newOps, op.getType(), blockSize.asArrayRef(), loc,
                              rewriter);
    rewriter.replaceOp(op, castOp);
    return mlir::success();
  }
};

// rewrite a StoreScatterOp on big tile size into multiple StoreScatterOps
// on smaller tile size.
class RewriteStoreScatterOp
    : public RewriteXeTileOp<xetile::StoreScatterOp, BlockingAnalysis> {
public:
  using RewriteXeTileOp<xetile::StoreScatterOp,
                        BlockingAnalysis>::RewriteXeTileOp;

  mlir::LogicalResult
  matchAndRewrite(xetile::StoreScatterOp op,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto value = op.getValue();
    auto tile = op.getTile();
    auto mask = op.getMask();
    auto tileTy = tile.getType();

    auto blockSize = analysis.getUseBlockSize(value, op->getOpOperand(0));

    if (!blockSize || tileTy.getShape() == blockSize.asArrayRef() ||
        blockSize != analysis.getUseBlockSize(tile, op->getOpOperand(1)))
      return mlir::failure();

    auto convertedValTypes =
        convertTypes(value.getType(), blockSize.asArrayRef());
    auto convertedTileTypes =
        convertTypes(tile.getType(), blockSize.asArrayRef());
    auto convertedMaskTypes =
        convertTypes(mask.getType(), blockSize.asArrayRef());

    auto values = addPackOp(value, convertedValTypes, blockSize.asArrayRef(),
                            loc, rewriter);
    auto tiles = addPackOp(tile, convertedTileTypes, blockSize.asArrayRef(),
                           loc, rewriter);
    auto masks = addPackOp(mask, convertedMaskTypes, blockSize.asArrayRef(),
                           loc, rewriter);

    for (auto [v, t, m] : llvm::zip(values, tiles, masks)) {
      (void)rewriter.create<xetile::StoreScatterOp>(
          loc, v, t, m, op.getL1HintAttr(), op.getL2HintAttr(),
          op.getL3HintAttr());
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// rewrite a update_tile_offset op on big tile size into multiple
// update_tile_offset ops on smaller tile size.
class RewriteUpdateTileOffsetOp
    : public RewriteXeTileOp<xetile::UpdateTileOffsetOp, BlockingAnalysis> {
public:
  using RewriteXeTileOp<xetile::UpdateTileOffsetOp,
                        BlockingAnalysis>::RewriteXeTileOp;

  mlir::LogicalResult
  matchAndRewrite(xetile::UpdateTileOffsetOp op,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tile = op.getTile();
    auto tileTy = tile.getType();
    auto shape = tileTy.getShape();
    auto ctx = op.getContext();

    auto blockSize = analysis.getDefBlockSize(tile);
    if (!blockSize || shape == blockSize.asArrayRef())
      return mlir::failure();

    auto convertedTileTypes = convertTypes(tileTy, blockSize.asArrayRef());
    auto convertedTiles = addPackOp(tile, convertedTileTypes,
                                    blockSize.asArrayRef(), loc, rewriter);

    llvm::SmallVector<mlir::Value> newOps;

    // handle scattered tiles.
    if (tileTy.getScatterAttr() == mlir::BoolAttr::get(ctx, true)) {
      auto indices = op.getIndices();
      assert(indices && "indices is missing.");
      auto indicesTy = indices.getType();

      auto convertedIndicesTypes =
          convertTypes(indicesTy, blockSize.asArrayRef());
      auto convertedIndices = addPackOp(indices, convertedIndicesTypes,
                                        blockSize.asArrayRef(), loc, rewriter);

      for (auto [t, i] : llvm::zip(convertedTiles, convertedIndices)) {
        auto newOp = rewriter.create<xetile::UpdateTileOffsetOp>(
            loc, t, op.getOffsetX(), op.getOffsetY(), i);
        newOps.push_back(newOp);
      }
    } else { // handle blocked tiles
      for (auto t : convertedTiles) {
        auto newOp = rewriter.create<xetile::UpdateTileOffsetOp>(
            loc, t, op.getOffsetX(), op.getOffsetY(), nullptr);
        newOps.push_back(newOp);
      }
    }

    auto castOp = addUnpackOp(newOps, op.getType(), blockSize.asArrayRef(), loc,
                              rewriter);
    rewriter.replaceOp(op, castOp);
    return mlir::success();
  }
};

// rewrite a tile_mma op on big tile size into multiple
// tile_mma ops on smaller tile size.
class RewriteTileMMAOp
    : public RewriteXeTileOp<xetile::TileMMAOp, BlockingAnalysis> {
public:
  using RewriteXeTileOp<xetile::TileMMAOp, BlockingAnalysis>::RewriteXeTileOp;

  mlir::LogicalResult
  matchAndRewrite(xetile::TileMMAOp op,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultTy = op.getResult().getType();

    auto a = op.getA();
    auto b = op.getB();
    auto c = op.getC();

    assert(a && b && "a operand or b operand is (are) missing.\n");

    auto getBlockingSize = [&](mlir::Value val, int pos) -> Block {
      if (!val)
        return Block();
      return analysis.getUseBlockSize(val, op->getOpOperand(pos));
    };

    auto aShape = a.getType().getShape();
    auto bShape = b.getType().getShape();

    auto aBlockSize = getBlockingSize(op.getA(), 0);
    auto bBlockSize = getBlockingSize(op.getB(), 1);
    auto cBlockSize = getBlockingSize(op.getC(), 2);

    llvm::SmallVector<mlir::Value> aVals, bVals, cVals;
    auto pack = [&](mlir::TypedValue<mlir::VectorType> val,
                    llvm::ArrayRef<int64_t> blockSize) {
      auto type = val.getType();
      if (type.getShape() == blockSize)
        return llvm::SmallVector<mlir::Value>({val});
      auto convertedTypes = convertTypes(type, blockSize);
      auto values = addPackOp(val, convertedTypes, blockSize, loc, rewriter);
      return llvm::to_vector(values);
    };

    if (aBlockSize)
      aVals = pack(a, aBlockSize.asArrayRef());

    if (bBlockSize)
      bVals = pack(b, bBlockSize.asArrayRef());

    if (c && cBlockSize)
      cVals = pack(c, cBlockSize.asArrayRef());

    // Vals are empty due to invalid blocking size, or with size 1 due to
    // the original shape is the same with the blocking size. The op will
    // be skipped if every operand got an invalid blocking size or the
    // original shape is the same with the blocking size.
    if (aVals.size() <= 1 && bVals.size() <= 1 && cVals.size() <= 1)
      return mlir::failure();

    uint64_t M = aShape[0] / aBlockSize[0];
    uint64_t K = aShape[1] / aBlockSize[1];
    uint64_t N = bShape[1] / bBlockSize[1];

    auto vecTy = ::mlir::VectorType::get({aBlockSize[0], bBlockSize[1]},
                                         resultTy.getElementType());
    mlir::SmallVector<mlir::Value> newOps;

    for (uint64_t i = 0; i < M; i++) {
      for (uint64_t j = 0; j < N; j++) {
        mlir::Value tmpC;
        if (c)
          tmpC = cVals[i * N + j]; // init with acc
        for (uint64_t k = 0; k < K; k++) {
          auto aVec = aVals[i * K + k];
          auto bVec = bVals[k * N + j];
          llvm::SmallVector<mlir::Value> operands({aVec, bVec});
          if (tmpC)
            operands.push_back(tmpC);
          tmpC = rewriter.create<xetile::TileMMAOp>(loc, vecTy, operands,
                                                    op->getAttrs());
        }
        newOps.push_back(tmpC);
      }
    }
    auto castOp =
        addUnpackOp(newOps, resultTy, Block().asArrayRef(), loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return mlir::success();
  }
};

// rewrite a tile_reduction op on big tile size into multiple
// tile_reduction ops on smaller tile size.
// Currently the outer reduction op is lowered into a set of
// binary ops across the reduction dimension (see details in
// comments for lowerOuterReduction). And the inner reduction
// is lowered into a set of binary ops and shuffle ops (See
// details in comments for lowerInnerReductionWithIntraVectorShuffles).
//
// TODO: Update the blocking and lowering strategy to generate
// binary ops across the blocks, and keep the reduction op on
// each block. e.g., we can choose 8x16 as block size for
// xetile.reduce<add> %src [0]: vector<32x64xf16> -> vector<1x64xf16>
// so we will have a grid (shape = [4, 4]) of 8x16 blocks, and then
// we can generate 4 binary ops working on vector<8x16> for
// grid[:, i] (i = 0, 1, 2, 3). It will result in 4 vector values
// of type vector<8x16>, and then generate reduction op on each vector.
class RewriteTileReductionOp
    : public RewriteXeTileOp<xetile::ReductionOp, BlockingAnalysis> {
public:
  using RewriteXeTileOp<xetile::ReductionOp, BlockingAnalysis>::RewriteXeTileOp;

  mlir::LogicalResult
  matchAndRewrite(xetile::ReductionOp op,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = op.getSource();
    auto srcTy = src.getType();
    auto shape = srcTy.getShape();
    auto dims = op.getReductionDims();
    // only support 2D vector, and reduction on one dimension.
    if (srcTy.getRank() != 2 || dims.size() != 1)
      return rewriter.notifyMatchFailure(op, "unsupported reduction op");

    auto blkSize = analysis.getUseBlockSize(src, op->getOpOperand(0));
    if (!blkSize)
      return rewriter.notifyMatchFailure(op, "Invalid blocking size");

    auto convertedSrcTypes = convertTypes(srcTy, blkSize.asArrayRef());
    auto convertedSrcs =
        addPackOp(src, convertedSrcTypes, blkSize.asArrayRef(), loc, rewriter);

    int64_t grid[2] = {shape[0] / blkSize[0], shape[1] / blkSize[1]};

    llvm::SmallVector<mlir::Value> newOps;
    if (dims[0] == 0) {
      newOps =
          lowerOuterReduction(convertedSrcs, grid, op.getKind(), loc, rewriter);
    } else if (dims[0] == 1) {
      auto elemTy = srcTy.getElementType();
      auto intermediates = lowerInnerReductionWithIntraVectorShuffles(
          convertedSrcs, elemTy, grid, blkSize.asArrayRef(), op.getKind(), loc,
          rewriter);

      for (auto v : intermediates) {
        auto resultTy = mlir::VectorType::get({1, 1}, elemTy);
        for (auto i = 0; i < blkSize[1]; i++) {
          auto pos = rewriter.create<mlir::arith::ConstantOp>(
              loc, rewriter.getI32IntegerAttr(i));
          auto extractOp =
              rewriter.create<mlir::vector::ExtractElementOp>(loc, v, pos);
          auto splatOp = rewriter.create<mlir::vector::SplatOp>(
              op.getLoc(), resultTy, extractOp);
          newOps.push_back(splatOp);
        }
      }
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported reduction dim");
    }

    blkSize[dims[0]] = 1;
    auto castOp =
        addUnpackOp(newOps, op.getType(), blkSize.asArrayRef(), loc, rewriter);
    rewriter.replaceOp(op, castOp);

    return mlir::success();
  }
};

// rewrite a tile_broadcast op on big tile size into multiple
// tile_broadcast ops on smaller tile size.
class RewriteTileBroadcastOp
    : public RewriteXeTileOp<xetile::BroadcastOp, BlockingAnalysis> {
public:
  using RewriteXeTileOp<xetile::BroadcastOp, BlockingAnalysis>::RewriteXeTileOp;

  mlir::LogicalResult
  matchAndRewrite(xetile::BroadcastOp op,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = op.getSource();
    auto srcTy = src.getType();
    auto elemTy = srcTy.getElementType();
    auto dims = op.getBroadcastDim();
    if (srcTy.getRank() != 2 || dims.size() != 1)
      return rewriter.notifyMatchFailure(op, "unsupported broadcast op");

    auto srcBlkSize = analysis.getUseBlockSize(src, op->getOpOperand(0));
    auto resBlkSize = analysis.getDefBlockSize(op.getResult());

    if (!srcBlkSize || !resBlkSize)
      return rewriter.notifyMatchFailure(op, "Invalid blocking size");

    if (srcTy.getShape() == srcBlkSize.asArrayRef())
      return rewriter.notifyMatchFailure(op, "No need to block");

    auto convertedSrcTypes = convertTypes(srcTy, srcBlkSize.asArrayRef());
    auto convertedSrcs = addPackOp(src, convertedSrcTypes,
                                   srcBlkSize.asArrayRef(), loc, rewriter);

    auto resTy = op.getResult().getType();
    int64_t resultGrid[2] = {resTy.getShape()[0] / resBlkSize[0],
                             resTy.getShape()[1] / resBlkSize[1]};

    llvm::SmallVector<mlir::Value> newOps;
    if (dims[0] == 0) {
      // clang-format off
      // broadcast along the first dim, we simply need to replicate the source.
      // For example, for
      //    xetile.broadcast %src [0]: vector<1x64xf16> -> vector<32x64xf16>
      // After blocking (assuming block size = [1, 16]) and lowering to xegpu,
      // its input values (source) will be a vector of values with type <1x16xf16>
      // and size = 4, which can be viewed as:
      // | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> |
      // so we need to replicate it 32 times (resultGrid[0]) to get final results:
      //  0: | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> |
      //  ......
      // 31: | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> |
      // clang-format on
      for (auto i = 0; i < resultGrid[0]; i++)
        newOps.append(convertedSrcs.begin(), convertedSrcs.end());
    } else if (dims[0] == 1) {
      // clang-format off
      // broadcast along the second dim, we use both splatOp and replicates.
      // For example: xetile.broadcast %src [1]: vector<32x1xf16> ->
      // vector<32x64xf16>. After blocking (assuming block size = [1, 16]) and
      // lowering to xegpu, the input value (source) will be a vector of values
      // with type <1x1xf16> and size = 32, which can be viewed as:
      //    0: | vector<1x1xf16> |
      //           ...
      //   31: | vector<1x1xf16> |
      // first, splatOp is used to broadcast the value of vector<1x1xf16> to
      // vector<1x16xf16>
      //    0: | vector<1x16xf16> |
      //           ...
      //   31: | vector<1x16xf16> |
      // and then we replicate the splatOp 4 times (resultGrid[1]) to get the
      // final results:
      //    0: | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> |
      //           ...
      //   31: | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> | vector<1x16xf16> |
      // clang-format on
      auto dstTy = mlir::VectorType::get(resBlkSize.asArrayRef(), elemTy);
      for (auto src : convertedSrcs) {
        auto ty = mlir::dyn_cast<mlir::VectorType>(src.getType());
        assert(ty && ty.getNumElements() == 1 &&
               "Expecting a <1x1xelemty> vector type.");
        auto ext = rewriter.create<mlir::vector::ExtractOp>(
            loc, src, llvm::ArrayRef<int64_t>({0, 0}));
        auto splatOp = rewriter.create<mlir::vector::SplatOp>(loc, dstTy, ext);
        newOps.append(resultGrid[1], splatOp);
      }
    } else {
      return mlir::failure();
    }
    auto castOp =
        addUnpackOp(newOps, resTy, resBlkSize.asArrayRef(), loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return mlir::success();
  }
};

// rewrite a tile_transpose op on big tile size into multiple
// tile_transpose ops on smaller tile size.
class RewriteTileTransposeOp
    : public RewriteXeTileOp<xetile::TransposeOp, BlockingAnalysis> {
  using RewriteXeTileOp<xetile::TransposeOp, BlockingAnalysis>::RewriteXeTileOp;

  mlir::LogicalResult
  matchAndRewrite(xetile::TransposeOp op,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getVector();
    auto inputTy = input.getType();
    auto inShape = inputTy.getShape();
    auto result = op.getResult();
    auto resultTy = result.getType();
    auto resultShape = resultTy.getShape();

    auto permutation = op.getPermutation();
    if (permutation != mlir::ArrayRef<int64_t>({1, 0}))
      return rewriter.notifyMatchFailure(op, "Unsupported permutation");

    auto inBlockSize = analysis.getUseBlockSize(input, op->getOpOperand(0));
    auto outBlockSize = analysis.getDefBlockSize(result);
    if (!inBlockSize || !outBlockSize || inShape == inBlockSize.asArrayRef() ||
        resultShape == outBlockSize.asArrayRef())
      return mlir::failure();
    auto elemTy = inputTy.getElementType();

    auto newDstTy = mlir::VectorType::get(outBlockSize.asArrayRef(), elemTy);

    auto convertedInputTypes = convertTypes(inputTy, inBlockSize.asArrayRef());
    auto convertedResultTypes =
        convertTypes(resultTy, outBlockSize.asArrayRef());

    auto convertedInputs = addPackOp(input, convertedInputTypes,
                                     inBlockSize.asArrayRef(), loc, rewriter);

    int64_t grids[2] = {resultShape[0] / outBlockSize[0],
                        resultShape[1] / outBlockSize[1]};
    llvm::SmallVector<mlir::Value> newOps;
    for (auto i : llvm::seq<int64_t>(0, grids[0])) {
      for (auto j : llvm::seq<int64_t>(0, grids[1])) {
        int64_t idx = i + grids[0] * j;
        mlir::Value arg = convertedInputs[idx];
        mlir::Value res = rewriter.create<xetile::TransposeOp>(
            loc, newDstTy, arg, permutation);
        newOps.push_back(res);
      }
    }
    auto castOp =
        addUnpackOp(newOps, resultTy, outBlockSize.asArrayRef(), loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return mlir::success();
  }
};

// rewrite a vectorizable op (e.g., addf) on big vector size into multiple
// same ops on smaller vector size.
// TODO: replace it with upstream unroll pattern in vector dialect.
class RewriteVectorizableOp
    : public RewriteOpWithTrait<mlir::OpTrait::Vectorizable, BlockingAnalysis> {
public:
  using RewriteOpWithTrait::RewriteOpWithTrait;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  OpPatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "op must have 1 result");

    auto res = op->getResult(0);
    auto resType = mlir::dyn_cast<mlir::VectorType>(res.getType());
    if (!resType || resType.getRank() != 2)
      return mlir::failure();

    auto resShape = resType.getShape();
    auto blockSize =
        analysis.getUseBlockSize(op->getOperand(0), op->getOpOperand(0));

    if (!blockSize || resShape == blockSize.asArrayRef())
      return mlir::failure();

    auto elemTy = resType.getElementType();
    auto newTy = mlir::VectorType::get(blockSize.asArrayRef(), elemTy);

    Location loc = op->getLoc();
    llvm::SmallVector<mlir::ValueRange> newOperands;
    for (auto opr : op->getOperands()) {
      auto oprTy = mlir::dyn_cast<mlir::VectorType>(opr.getType());
      if (!oprTy || oprTy.getRank() != 2)
        newOperands.emplace_back(opr);
      auto convertedTypes = convertTypes(oprTy, blockSize.asArrayRef());
      auto convertedValues =
          addPackOp(opr, convertedTypes, blockSize.asArrayRef(), loc, rewriter);
      newOperands.push_back(convertedValues);
    }

    mlir::OpBuilder::InsertionGuard g(rewriter);

    int64_t grids[2] = {resShape[0] / blockSize[0], resShape[1] / blockSize[1]};
    llvm::SmallVector<mlir::Value> newOps;
    for (int64_t i = 0; i < grids[0]; i++) {
      for (int64_t j = 0; j < grids[1]; j++) {
        int64_t idx = i * grids[1] + j;
        llvm::SmallVector<mlir::Value> operands;
        for (auto valRange : newOperands) {
          if (valRange.size() == 1)
            operands.push_back(valRange[0]);
          if (idx < (int64_t)valRange.size())
            operands.push_back(valRange[idx]);
        }
        if (operands.size() != op->getNumOperands())
          return mlir::failure();
        mlir::OperationState opState(loc, op->getName(), operands,
                                     mlir::TypeRange(newTy), op->getAttrs(),
                                     op->getSuccessors());
        auto newOp = rewriter.create(opState);
        newOps.push_back(newOp->getResult(0));
      }
    }

    auto castOp =
        addUnpackOp(newOps, resType, blockSize.asArrayRef(), loc, rewriter);

    rewriter.replaceOp(op, castOp);
    return mlir::success();
  }
};

// Update the SCF forOp when it has arguments being blocked and and needs
// to be replaced with a set of new arguments with smaller size
// TODO: Can we replace this pattern to match with RegionBranchOpInterface?
// It may improve the generality of the pattern.
class RewriteSCFForOp
    : public RewriteXeTileOp<mlir::scf::ForOp, BlockingAnalysis> {
public:
  using RewriteXeTileOp<mlir::scf::ForOp, BlockingAnalysis>::RewriteXeTileOp;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto initArgs = op.getInitArgs();
    auto regionArgs = op.getRegionIterArgs();
    auto results = op.getResults();
    llvm::SmallVector<Block> blockSZs;

    // We use region args as anchor. PackOps will be inserted if ncessary
    // for each init args, and UnpackOps will be inserted for each argument
    // and result.
    for (auto arg : regionArgs) {
      auto argBlock = analysis.getDefBlockSize(arg);
      blockSZs.push_back(argBlock);
    }

    // preprocess the init args by adding pack ops if necessary,
    // and build the SignatureConversion for region arguments.
    auto origArgCount = op.getNumRegionIterArgs();
    mlir::TypeConverter::SignatureConversion argConversion(origArgCount);
    llvm::SmallVector<mlir::Value> convertedInitArgs;
    for (auto [i, v] : llvm::enumerate(initArgs)) {
      auto blockSZ = blockSZs[i];
      auto type = mlir::dyn_cast<mlir::ShapedType>(v.getType());
      if (!blockSZ || !type || type.getShape() == blockSZ.asArrayRef()) {
        argConversion.addInputs(i, v.getType());
        convertedInitArgs.push_back(v);
      } else {
        auto newTypes = convertTypes(type, blockSZ.asArrayRef());
        argConversion.addInputs(i, newTypes);
        auto values =
            addPackOp(v, newTypes, blockSZ.asArrayRef(), loc, rewriter);
        convertedInitArgs.append(values.begin(), values.end());
      }
    }

    // no change is needed if convertedInitArgs is the same as current ones.
    if (llvm::equal(convertedInitArgs, initArgs))
      return mlir::failure();

    auto newOp = rewriter.create<mlir::scf::ForOp>(
        loc, op.getLowerBound(), op.getUpperBound(), op.getStep(),
        convertedInitArgs);
    mlir::Block *newBlock = newOp.getBody();
    // remove the terminator of the new block
    if (newBlock->mightHaveTerminator())
      rewriter.eraseOp(newBlock->getTerminator());

    llvm::SmallVector<mlir::Value> castArgs;
    if (auto inductionVals = newOp.getLoopInductionVars())
      castArgs = inductionVals.value();

    auto savedIP = rewriter.saveInsertionPoint();
    PatternRewriter::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(newBlock);
    // create unpackOp for converted region arguments if necessary.
    auto convertedArgs = newOp.getRegionIterArgs();
    for (unsigned i = 0; i < origArgCount; i++) {
      auto inputMap = argConversion.getInputMapping(i);
      if (!inputMap || inputMap->size == 1) {
        castArgs.push_back(convertedArgs[inputMap->inputNo]);
      } else {
        auto arg = addUnpackOp(
            convertedArgs.slice(inputMap->inputNo, inputMap->size),
            regionArgs[i].getType(), blockSZs[i].asArrayRef(), loc, rewriter);
        castArgs.push_back(arg);
      }
    }
    rewriter.restoreInsertionPoint(savedIP);
    rewriter.mergeBlocks(op.getBody(), newBlock, castArgs);

    llvm::SmallVector<mlir::Value> castResults;
    auto convertedResults = newOp.getResults();
    for (unsigned i = 0; i < origArgCount; i++) {
      auto inputMap = argConversion.getInputMapping(i);
      if (!inputMap || inputMap->size == 1) {
        castResults.push_back(convertedResults[inputMap->inputNo]);
      } else {
        auto res = addUnpackOp(
            convertedResults.slice(inputMap->inputNo, inputMap->size),
            results[i].getType(), blockSZs[i].asArrayRef(), loc, rewriter);
        castResults.push_back(res);
      }
    }

    rewriter.replaceOp(op, castResults);
    return mlir::success();
  }
};

// Update the SCF Yield op when its operands have being blocked and and needs
// to be replaced with a set of new values with smaller size
class RewriteSCFYieldOp
    : public RewriteXeTileOp<mlir::scf::YieldOp, BlockingAnalysis> {
public:
  using RewriteXeTileOp<mlir::scf::YieldOp, BlockingAnalysis>::RewriteXeTileOp;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    llvm::SmallVector<mlir::Value> convertedResults;
    for (auto res : op.getResults()) {
      auto blockSZ = analysis.getDefBlockSize(res);
      auto type = mlir::dyn_cast<mlir::ShapedType>(res.getType());
      if (blockSZ && type && type.getShape() != blockSZ.asArrayRef()) {
        auto newTypes = convertTypes(type, blockSZ.asArrayRef());
        auto values =
            addPackOp(res, newTypes, blockSZ.asArrayRef(), loc, rewriter);
        convertedResults.append(values.begin(), values.end());
      } else {
        convertedResults.push_back(res);
      }
    }
    if (llvm::equal(convertedResults, op.getResults()))
      return mlir::failure();
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, convertedResults);
    return mlir::success();
  }
};

// Rewrite a create_mask op on big vector size into multiple create_mask ops
// on smaller vector size.
class RewriteCreateMaskOp
    : public RewriteXeTileOp<mlir::vector::CreateMaskOp, BlockingAnalysis> {
public:
  using RewriteXeTileOp<mlir::vector::CreateMaskOp,
                        BlockingAnalysis>::RewriteXeTileOp;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::CreateMaskOp op,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto res = op.getResult();
    auto resTy = res.getType();
    auto shape = resTy.getShape();
    auto blockSize = analysis.getDefBlockSize(res);
    if (!blockSize || shape == blockSize.asArrayRef())
      return mlir::failure();

    auto operands = op.getOperands();

    auto sub = [&](mlir::Value a, int64_t b) -> mlir::Value {
      auto ofr = mlir::getAsOpFoldResult(a);
      if (auto cst = mlir::getConstantIntValue(ofr)) {
        auto val = std::max<int64_t>(*cst - b, 0);
        auto attr = rewriter.getIndexAttr(val);
        return rewriter.create<mlir::arith::ConstantOp>(loc, attr);
      } else {
        auto bV = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(b));
        return rewriter.create<mlir::arith::SubIOp>(loc, a, bV);
      }
    };

    auto elemTy = resTy.getElementType();
    auto newTy = mlir::VectorType::get(blockSize.asArrayRef(), elemTy);
    llvm::SmallVector<mlir::Value> newOps;
    mlir::Value x = operands[0];
    for (int64_t i = 0; i < shape[0]; i += blockSize[0]) {
      mlir::Value y = operands[1];
      for (int64_t j = 0; j < shape[1]; j += blockSize[1]) {
        auto newOp = rewriter.create<mlir::vector::CreateMaskOp>(
            loc, newTy, mlir::ValueRange({x, y}));
        newOps.push_back(newOp);
        y = sub(y, blockSize[1]);
      }
      x = sub(x, blockSize[0]);
    }
    auto castOp =
        addUnpackOp(newOps, resTy, blockSize.asArrayRef(), loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return mlir::success();
  }
};

// Rewrite a splat op on big vector size into multiple splat ops
// on smaller vector size.
class RewriteSplatOp
    : public RewriteXeTileOp<mlir::vector::SplatOp, BlockingAnalysis> {
public:
  using RewriteXeTileOp<mlir::vector::SplatOp,
                        BlockingAnalysis>::RewriteXeTileOp;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::SplatOp op,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto res = op.getAggregate();
    auto resTy = res.getType();
    auto shape = resTy.getShape();
    auto blockSize = analysis.getDefBlockSize(res);
    if (!blockSize || resTy.getRank() != 2 || shape == blockSize.asArrayRef())
      return mlir::failure();

    auto newTy =
        mlir::VectorType::get(blockSize.asArrayRef(), resTy.getElementType());
    auto newOp = rewriter.create<mlir::vector::SplatOp>(
        loc, newTy, op->getOperands(), op->getAttrs());
    auto numOps = resTy.getNumElements() / newTy.getNumElements();
    llvm::SmallVector<mlir::Value> newOps(numOps, newOp);
    auto castOp =
        addUnpackOp(newOps, resTy, blockSize.asArrayRef(), loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return mlir::success();
  }
};
} // namespace Blocking

void populateXeTileBlockingPatterns(mlir::RewritePatternSet &patterns,
                                    BlockingAnalysis &analysis) {
  patterns.insert<
      Blocking::RewriteArithConstantOp, Blocking::RewriteInitTileOp,
      Blocking::RewritePrefetchTileOp, Blocking::RewriteLoadTileOp,
      Blocking::RewriteStoreTileOp, Blocking::RewriteLoadGatherOp,
      Blocking::RewriteStoreScatterOp, Blocking::RewriteUpdateTileOffsetOp,
      Blocking::RewriteTileMMAOp, Blocking::RewriteTileReductionOp,
      Blocking::RewriteTileBroadcastOp, Blocking::RewriteTileTransposeOp,
      Blocking::RewriteVectorizableOp, Blocking::RewriteSCFForOp,
      Blocking::RewriteSCFYieldOp, Blocking::RewriteCreateMaskOp,
      Blocking::RewriteCreateMaskOp>(patterns.getContext(), analysis);
}

// Lowers XeTile to blocked layout with high-dim vector
class XeTileBlockingPass : public impl::XeTileBlockingBase<XeTileBlockingPass> {
public:
  XeTileBlockingPass() {
    uArchInterface = std::make_shared<imex::XePVCuArch>();
  }

  XeTileBlockingPass(const std::string &deviceName) {
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
    auto mod = this->getOperation();
    // skip functions with XeTile.TileType inputs and outputs
    if (!isSupportedModule(mod)) {
      mod.emitOpError(
          "Currently FunctionType with xetile.TileType is not supported.");
      return signalPassFailure();
    }

    if (!uArchInterface) {
      mod.emitOpError("Can not get GPU Arch Definition for given Arch param");
      return signalPassFailure();
    }

    BlockingAnalysis analysis(uArchInterface);
    if (mlir::failed(analysis.run(mod)))
      return signalPassFailure();

    LLVM_DEBUG(analysis.printAnalysisResult());

    mlir::MLIRContext &context = getContext();

    mlir::GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    // ops inside regions, e.g., body of scf.for, needs to be processed
    // before the op (e.g., scf.for) containing the region; otherwise
    // the blocking analysis result for region args will be destroyed
    // after scf.for is updated, leading to their users cannot be updated
    // correctly.
    mod.walk([&](mlir::Region *region) {
      config.scope = region;
      mlir::RewritePatternSet patterns(&context);
      populateXeTileBlockingPatterns(patterns, analysis);
      llvm::SmallVector<mlir::Operation *> ops;
      region->walk([&](mlir::Operation *op) {
        if (op->getParentRegion() == region)
          ops.push_back(op);
      });
      if (mlir::failed(
              mlir::applyOpPatternsAndFold(ops, std::move(patterns), config)))
        return signalPassFailure();
    });

    // run CSE and Canonicalizer again to remove identical packOps,
    // and fold pack/unpack ops with the same block size.
    mlir::PassManager pm(&context);
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    if (mlir::failed(pm.run(mod)))
      return signalPassFailure();

    // Resolve unfoldable Unpack and Pack Ops
    mod.walk([&](mlir::UnrealizedConversionCastOp castOp) {
      // remove dead castOp.
      if (castOp->use_empty()) {
        castOp.erase();
        return mlir::WalkResult::advance();
      }

      mlir::OpBuilder builder(castOp);
      auto context = castOp.getContext();

      // handle unpack op
      if (Blocking::isUnpackOp(castOp)) {
        auto user = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(
            *castOp->user_begin());

        auto [inGrids, inBlkSizes] = Blocking::getGridAndBlockSizes(castOp);
        mlir::DenseI64ArrayAttr outBlkSizes, outGrids;

        // if unpack op is used by a pack op,
        if (castOp->hasOneUse() && Blocking::isPackOp(user) &&
            Blocking::isUnpackPackCompatible(castOp, user)) {
          std::tie(outGrids, outBlkSizes) =
              Blocking::getGridAndBlockSizes(user);
        } else {
          auto outTy =
              mlir::dyn_cast<mlir::ShapedType>(castOp->getResult(0).getType());
          outGrids = mlir::DenseI64ArrayAttr::get(context, {1, 1});
          outBlkSizes = mlir::DenseI64ArrayAttr::get(context, outTy.getShape());
        }

        builder.setInsertionPointAfter(castOp);
        auto newOps = Blocking::lowerUnpackOrPack(
            castOp.getInputs(), inBlkSizes, outBlkSizes, inGrids, outGrids,
            castOp.getLoc(), builder);
        if (newOps.size() == 1) {
          castOp->getResult(0).replaceAllUsesWith(newOps[0]);
        } else {
          for (auto [n, c] : llvm::zip_equal(newOps, user->getResults()))
            c.replaceAllUsesWith(n);
        }
        return mlir::WalkResult::advance();
      }

      // handle pack op
      if (Blocking::isPackOp(castOp)) {
        auto opr = castOp->getOperand(0)
                       .getDefiningOp<mlir::UnrealizedConversionCastOp>();
        // should be handled as a pair of unpack and pack op in above.
        if (Blocking::isUnpackOp(opr) && opr->hasOneUse() &&
            Blocking::isUnpackPackCompatible(opr, castOp))
          return mlir::WalkResult::advance();

        auto inTy =
            mlir::dyn_cast<mlir::ShapedType>(castOp->getOperand(0).getType());
        auto inGrids = mlir::DenseI64ArrayAttr::get(context, {1, 1});
        auto inBlkSizes =
            mlir::DenseI64ArrayAttr::get(context, inTy.getShape());
        auto [outGrids, outBlkSizes] = Blocking::getGridAndBlockSizes(castOp);
        auto newOps = Blocking::lowerUnpackOrPack(
            castOp.getInputs(), inBlkSizes, outBlkSizes, inGrids, outGrids,
            castOp.getLoc(), builder);
        for (auto [n, c] : llvm::zip_equal(newOps, castOp->getResults()))
          c.replaceAllUsesWith(n);
      }
      return mlir::WalkResult::advance();
    });

    // remove dead castOp.
    mod.walk([&](mlir::UnrealizedConversionCastOp castOp) {
      if (castOp->use_empty())
        castOp.erase();
    });
  }

private:
  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;
};

/// Create a pass
std::unique_ptr<::mlir::Pass>
createXeTileBlockingPass(const std::string &deviceName) {
  return std::make_unique<XeTileBlockingPass>(deviceName);
}
} // namespace imex
