//===- ArithOpConversion.cpp - XeTileToXeGPU conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the ArithOpConversionPattern, used in XeTileToXeGPU
/// conversion, converting the Arith Ops.
///
//===----------------------------------------------------------------------===//

#include "ArithOpConversion.h"

namespace imex {

using VectorTypedValue = mlir::TypedValue<mlir::VectorType>;
using funcTy = VectorTypedValue(mlir::Value, mlir::Value, mlir::Location,
                                mlir::PatternRewriter &);

// see its description in XeTileOpConversion.cpp
extern VectorTypedValue concat(mlir::Value v1, mlir::Value v2,
                               mlir::Location loc,
                               mlir::PatternRewriter &rewriter);

// see its description in XeTileOpConversion.cpp
extern mlir::Value mergeVectorsWrapper(mlir::ValueRange ins,
                                       std::function<funcTy> transFunc,
                                       mlir::Location loc,
                                       XeOneToNPatternRewriter &rewriter);

static mlir::Value createBinOp(mlir::vector::CombiningKind kind,
                               mlir::Value lhs, mlir::Value rhs,
                               mlir::Type elemTy, mlir::Location &loc,
                               XeOneToNPatternRewriter &rewriter) {

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

llvm::SmallVector<mlir::Value>
lowerOuterReduction(mlir::ValueRange sources, llvm::ArrayRef<int64_t> shape,
                    mlir::vector::CombiningKind kind, mlir::Location loc,
                    mlir::Type elemTy, XeOneToNPatternRewriter &rewriter) {
  assert(shape.size() == 4 && "shape should be 4D.");
  llvm::SmallVector<mlir::Value> intermediates;
  for (auto j = 0; j < shape[1]; j++) {
    auto combiningVal = sources[j];
    for (auto i = 1; i < shape[0]; i++) {
      combiningVal = createBinOp(kind, combiningVal, sources[i * shape[1] + j],
                                 elemTy, loc, rewriter);
    }
    {
      // TODO: After blocking If the first dimension of the small block is not
      // 1, the combiningVal is now in shape as, e.g., vector<4x16xf16> instead
      // of vector<1x16xf16> then more reductions are needed in dim0, to make it
      // as vector<1x16xf16>. Currently, this is not implemented, since we are
      // now restricted blocking pass to set it as 1 now. It may cannot achieve
      // peak performance in some cases.
      assert(shape[2] == 1 &&
             "more reductions is needed in dim0, but not supported.");
    }
    intermediates.push_back(combiningVal);
  }
  return intermediates;
}

// expected input is type of vector<ixjx1xnxf16>, where i and n is power of 2
// and the third dim is always 1, which should be set by the blocking pass.
// For a vector of vector<32x64xf16> with reduction on dim 1, it will blocked
// into a vector<32x4x1x16> with reduction on dim 1 and dim 3.
// lowerInnerReductionWithIntraVectorShuffles performs the reduction with
// arithmetic operations on vector<16xf16>. To perform reduction on dim 1,
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
llvm::SmallVector<mlir::Value> lowerInnerReductionWithIntraVectorShuffles(
    mlir::ValueRange sources, llvm::ArrayRef<int64_t> shape,
    mlir::vector::CombiningKind kind, mlir::Location loc, mlir::Type elemTy,
    XeOneToNPatternRewriter &rewriter) {

  assert(shape.size() == 4 && "shape should be 4D.");

  auto isPowerOfTwo = [](auto n) { return (n & (n - 1)) == 0; };

  // make sure the dim0 of the block is 1 in blocking pass
  // different from outer reduction, this is strictly required
  // for this method.
  assert(shape[2] == 1 && "dim0 of the block has to be 1.");
  assert(isPowerOfTwo(shape[0]) && isPowerOfTwo(shape[3]) &&
         "sizes of dim1 and dim4 should be power of 2.");

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
  llvm::SmallVector<mlir::Value> intermediates(shape[0]);
  for (auto i = 0; i < shape[0]; i++) {
    auto combiningVal = sources[i * shape[1]];
    for (auto j = 1; j < shape[1]; j++) {
      combiningVal = createBinOp(kind, combiningVal, sources[i * shape[1] + j],
                                 elemTy, loc, rewriter);
    }
    // cast the result of e.g., vector<1x16xf16> into vector<16xf16>
    auto targetTy = mlir::VectorType::get({shape[3]}, elemTy);
    combiningVal =
        rewriter.create<mlir::vector::ShapeCastOp>(loc, targetTy, combiningVal);
    intermediates[i] = combiningVal;
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
  auto partialRowAggSize{shape[3]};
  auto numVecsLeft{shape[0]};
  while (partialRowAggSize != 1 && numVecsLeft != 1) {
    partialRowAggSize /= 2;
    auto workList = intermediates;
    intermediates.clear();
    assert(workList.size() % 2 == 0 && "The size should be divisible by 2.");
    auto masks = genShuffleMasks(partialRowAggSize, shape[3]);
    for (size_t i = 0; i < workList.size(); i += 2) {
      auto v1 = workList[i];
      auto v2 = workList[i + 1];
      auto shuffleOp1 =
          rewriter.create<mlir::vector::ShuffleOp>(loc, v1, v2, masks.first);
      auto shuffleOp2 =
          rewriter.create<mlir::vector::ShuffleOp>(loc, v1, v2, masks.second);
      auto reductionVal =
          createBinOp(kind, shuffleOp1, shuffleOp2, elemTy, loc, rewriter);
      intermediates.push_back(reductionVal);
    }
    numVecsLeft /= 2;
  }

  if (partialRowAggSize > 1) {
    assert(intermediates.size() == 1 &&
           "We must have ONE row with non-finalized aggregates.");
    auto toFinalize = intermediates.back();
    intermediates.clear();
    uint32_t currentAggVecSize = shape[3];
    do {
      currentAggVecSize /= 2;
      partialRowAggSize /= 2;
      auto [vecUpperMask, vecLowerMask] =
          genShuffleMasks(partialRowAggSize, currentAggVecSize);
      auto shuffleOp1 = rewriter.create<mlir::vector::ShuffleOp>(
          loc, toFinalize, toFinalize, vecUpperMask);
      auto shuffleOp2 = rewriter.create<mlir::vector::ShuffleOp>(
          loc, toFinalize, toFinalize, vecLowerMask);
      toFinalize =
          createBinOp(kind, shuffleOp1, shuffleOp2, elemTy, loc, rewriter);
    } while (partialRowAggSize != 1);
    intermediates.push_back(toFinalize);
  }
  return intermediates;
}

// TODO: Debug the IGC crash on this path. Currently, the upstream lows
// vector.reduction <add> into a spirv.CL.mul operation. But the generated
// code caused a crash in IGC.
llvm::SmallVector<mlir::Value> lowerInnerReductionWithVectorReduction(
    mlir::ValueRange sources, llvm::ArrayRef<int64_t> shape,
    mlir::vector::CombiningKind kind, mlir::Location loc, mlir::Type elemTy,
    XeOneToNPatternRewriter &rewriter) {

  assert(shape.size() == 4 && "shape should be 4D.");
  // vector<ixjx1xnxf16> equals to a grid of ixj of vector<1xnxf16>
  // this stage will use vector.shapecast to cast vector<1xnxf16> into 1D and
  // use vector.reduction firstly to perform the reduction over each vector,
  // and then use arith opertors to perform the reduction over the
  // aforementioned results for a row.
  llvm::SmallVector<mlir::Value> results;
  for (auto i = 0; i < shape[0]; i++) {
    llvm::SmallVector<mlir::Value> reductions;
    // perform reduction over each vector in a row
    for (auto j = 0; j < shape[1]; j++) {
      auto targetTy = mlir::VectorType::get({shape[2] * shape[3]}, elemTy);
      auto cast = rewriter.create<mlir::vector::ShapeCastOp>(
          loc, targetTy, sources[i * shape[1] + j]);
      auto value = rewriter.create<mlir::vector::ReductionOp>(loc, kind, cast);
      reductions.push_back(value);
    }
    auto reductionVal = reductions[0];
    // perform reduction over the results of each vector in a row
    for (auto j = 1; j < shape[1]; j++) {
      reductionVal =
          createBinOp(kind, reductionVal, reductions[j], elemTy, loc, rewriter);
    }
    results.push_back(reductionVal);
  }
  return results;
}

class SgVectorMultiDimReductionOpPattern
    : public XeOneToNConversion<mlir::vector::MultiDimReductionOp> {
  using XeOneToNConversion<
      mlir::vector::MultiDimReductionOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::MultiDimReductionOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
    auto srcTy = op.getSource().getType();
    auto elemTy = srcTy.getElementType();
    auto dims = op.getReductionDims();
    // its input should be a 4D vector, and has 2 reduction dims,
    // otherwise run the blocking pass first.
    if (dims.size() != 2 || srcTy.getRank() != 4)
      return mlir::failure();

    auto loc = op.getLoc();
    auto shape = srcTy.getShape();
    auto sources = adaptor.getSource();

    rewriter.setInsertionPoint(op);
    // doing reduction on outer dimension
    if (dims[0] == 0 &&
        dims[1] == 2) {
      auto intermediates = lowerOuterReduction(sources, shape, op.getKind(),
                                               loc, elemTy, rewriter);
      {
        // TODO: need a better way to represent the result (align with
        // unpack/pack logic). currently we just shuffle them and cast it to the
        // type/shape in xetile program.
        auto reducedVal =
            mergeVectorsWrapper(intermediates, concat, loc, rewriter);
        auto targetTy = mlir::VectorType::get({shape[1], shape[3]}, elemTy);
        auto newOp = rewriter.create<mlir::vector::ShapeCastOp>(loc, targetTy,
                                                                reducedVal);
        rewriter.replaceOp(op, newOp);
      }
      return mlir::success();
    }

    // doing reduction on inner dimension
    if (dims[0] == 1 &&
        dims[1] == 3) {
      auto intermediates = lowerInnerReductionWithIntraVectorShuffles(
          sources, shape, op.getKind(), loc, elemTy, rewriter);

      { // TODO: need a better way to represent the result (align with
        // unpack/pack logic).
        // currently we just shuffle them and cast it to the type/shape in
        // xetile program.
        auto reductionVal =
            mergeVectorsWrapper(intermediates, concat, loc, rewriter);
        auto targetTy = mlir::VectorType::get({shape[0], shape[2]}, elemTy);
        auto newOp = rewriter.create<mlir::vector::ShapeCastOp>(loc, targetTy,
                                                                reductionVal);
        rewriter.replaceOp(op, newOp);
      }
      return mlir::success();
    }

    // something is wrong
    return op.emitError("unsupported reduction operation.");
  }
};

class SgArithConstantOpPattern
    : public XeOneToNConversion<mlir::arith::ConstantOp> {
  using XeOneToNConversion<mlir::arith::ConstantOp>::XeOneToNConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor adaptor,
                  XeOneToNPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto value = llvm::dyn_cast<mlir::DenseElementsAttr>(op.getValue());

    // We only interesting 4D vectors
    if (!value || value.getType().getRank() != 4)
      return mlir::failure();

    llvm::SmallVector<mlir::Attribute> elems(
        value.value_begin<mlir::Attribute>(),
        value.value_end<mlir::Attribute>());

    auto shape = value.getType().getShape();
    auto elemTy = value.getElementType();
    auto vecTy = mlir::VectorType::get({shape[2], shape[3]}, elemTy);

    // slice a block of (shape[2], shape[3]) from elems.
    auto slice = [&](int i, int j) {
      llvm::SmallVector<mlir::Attribute> block;
      auto width = shape[1] * shape[3];
      i = i * shape[2];
      j = j * shape[3];
      for (int64_t r = 0; r < shape[2]; r++)
        for (int64_t c = 0; c < shape[3]; c++)
          block.push_back(elems[(i + r) * width + j + c]);
      return block;
    };

    rewriter.setInsertionPoint(op);
    llvm::SmallVector<mlir::Value> newOps;
    for (auto i = 0; i < shape[0]; i++) {
      for (auto j = 0; j < shape[1]; j++) {
        auto values = slice(i, j);
        auto attr = mlir::DenseElementsAttr::get(vecTy, values);
        auto newOp = rewriter.create<mlir::arith::ConstantOp>(loc, attr);
        newOps.push_back(newOp);
      }
    }

    rewriter.replaceOp(op, newOps);
    return mlir::success();
  }
};

bool isLegalArithOp(mlir::Operation *op) {
  if (llvm::isa<mlir::arith::ConstantOp>(op)) {
    auto constOp = llvm::cast<mlir::arith::ConstantOp>(op);
    auto resultTy = constOp.getResult().getType();
    if (mlir::isa<mlir::VectorType>(resultTy) &&
        mlir::cast<mlir::VectorType>(resultTy).getRank() == 4)
      return false;
  }
  return true;
}

void populateArithOpConversionPatterns(imex::XeOneToNTypeConverter &converter,
                                       mlir::RewritePatternSet &patterns,
                                       TileUsageAnalysis &analysis) {
  patterns.add<SgArithConstantOpPattern, SgVectorMultiDimReductionOpPattern>(
      patterns.getContext(), converter, analysis);
}

} // namespace imex
