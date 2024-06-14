//===- CastOp.cpp - NDArray dialect  --------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the CastOp of the NDArray dialect.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>

/// Ported from mlir::tensor::CastOp
bool imex::ndarray::CastOp::areCastCompatible(mlir::TypeRange inputs,
                                              mlir::TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  mlir::Type a = inputs.front(), b = outputs.front();
  auto aT = llvm::dyn_cast<imex::ndarray::NDArrayType>(a);
  auto bT = llvm::dyn_cast<imex::ndarray::NDArrayType>(b);
  if (!aT || !bT)
    return false;

  if (aT.getElementType() != bT.getElementType())
    return false;

  return mlir::succeeded(
      mlir::verifyCompatibleShape(aT.getTensorType(), bT.getTensorType()));
}

/// Ported from mlir::tensor
bool imex::ndarray::canFoldIntoConsumerOp(imex::ndarray::CastOp castOp) {
  if (!castOp)
    return false;

  // Can fold if the source of cast has at least as much static information as
  // its results.
  return mlir::tensor::preservesStaticInformation(
      castOp.getType().getTensorType(),
      castOp.getSource().getType().getTensorType());
}
bool imex::ndarray::canFoldIntoConsumerOp(mlir::tensor::CastOp castOp) {
  if (!castOp)
    return false;

  // Can fold if the source of cast has at least as much static information as
  // its results.
  return mlir::tensor::preservesStaticInformation(
      castOp.getType(),
      castOp.getSource().getType());
}

/// Ported from mlir::tensor
mlir::LogicalResult imex::ndarray::foldArrayCast(mlir::Operation *op) {
  bool folded = false;
  for (mlir::OpOperand &operand : op->getOpOperands()) {
    auto castOp = operand.get().getDefiningOp<imex::ndarray::CastOp>();
    if (castOp && imex::ndarray::canFoldIntoConsumerOp(castOp)) {
      operand.set(castOp.getOperand());
      folded = true;
    }
  }
  return mlir::success(folded);
}

/// Compute a TensorType that has the joined shape knowledge of the two
/// given TensorTypes. The element types need to match.
/// Ported from mlir::tensor
static mlir::TensorType joinShapes(mlir::TensorType one, mlir::TensorType two) {
  assert(one.getElementType() == two.getElementType());

  if (!one.hasRank())
    return two;
  if (!two.hasRank())
    return one;

  int64_t rank = one.getRank();
  if (rank != two.getRank())
    return {};

  mlir::SmallVector<int64_t, 4> join;
  join.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    if (one.isDynamicDim(i)) {
      join.push_back(two.getDimSize(i));
      continue;
    }
    if (two.isDynamicDim(i)) {
      join.push_back(one.getDimSize(i));
      continue;
    }
    if (one.getDimSize(i) != two.getDimSize(i))
      return {};
    join.push_back(one.getDimSize(i));
  }
  return mlir::RankedTensorType::get(join, one.getElementType());
}

/// Replaces chains of two ndarray.cast operations by a single ndarray.cast
/// operation if doing so does not remove runtime constraints.
/// Ported from mlir::tensor::CastOp
struct ChainedNDArrayCast
    : public mlir::OpRewritePattern<imex::ndarray::CastOp> {
  using mlir::OpRewritePattern<imex::ndarray::CastOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ndarray::CastOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto arCastOperand = op.getOperand().getDefiningOp<imex::ndarray::CastOp>();

    if (!arCastOperand)
      return mlir::failure();

    // infer cast shapes with array types
    auto sourceType = llvm::cast<imex::ndarray::NDArrayType>(
                          arCastOperand.getOperand().getType())
                          .getTensorType();
    auto intermediateType =
        llvm::cast<imex::ndarray::NDArrayType>(arCastOperand.getType())
            .getTensorType();
    auto resultType =
        llvm::cast<imex::ndarray::NDArrayType>(op.getType()).getTensorType();

    // We can remove the intermediate cast if joining all three produces the
    // same result as just joining the source and result shapes.in
    auto tmpJoin = joinShapes(sourceType, intermediateType);
    if (!tmpJoin)
      return mlir::failure();
    auto firstJoin = joinShapes(tmpJoin, resultType);
    if (!firstJoin)
      return mlir::failure();

    // The newJoin always exists if the above join exists, it might just contain
    // less information. If so, we cannot drop the intermediate cast, as doing
    // so would remove runtime checks.
    auto newJoin = joinShapes(sourceType, resultType);
    if (firstJoin != newJoin)
      return mlir::failure();

    auto sourcePTType =
        mlir::dyn_cast<imex::ndarray::NDArrayType>(op.getSource().getType());
    auto resultPTType = sourcePTType.cloneWith(resultType.getShape(),
                                               resultType.getElementType());
    ;
    rewriter.replaceOpWithNewOp<imex::ndarray::CastOp>(
        op, resultPTType, arCastOperand.getOperand());
    return mlir::success();
  }
};

void imex::ndarray::CastOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<ChainedNDArrayCast>(context);
}

/// Pattern to rewrite a CastElemTypeOp replacing dynamically shaped inputs
/// by statically shaped inputs if they are defined by an appropriate CastOp.
class CastElemTypeOpInputCanonicalizer final
    : public mlir::OpRewritePattern<::imex::ndarray::CastElemTypeOp> {
public:
  using mlir::OpRewritePattern<
      ::imex::ndarray::CastElemTypeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::CastElemTypeOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    if (!llvm::isa<::imex::ndarray::NDArrayType>(op.getResult().getType())) {
      return mlir::failure();
    };

    auto src = op.getInput();
    auto srcNDTyp = mlir::dyn_cast<::imex::ndarray::NDArrayType>(src.getType());
    auto defOp = src.getDefiningOp<::imex::ndarray::CastOp>();
    if (!srcNDTyp || srcNDTyp.hasStaticShape() || !defOp) {
      return mlir::failure();
    }
    auto defOpSrc = defOp.getSource();
    auto defSrcNDTyp =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(defOpSrc.getType());
    if (!defSrcNDTyp || !defSrcNDTyp.hasStaticShape()) {
      return mlir::failure();
    }
    rewriter.modifyOpInPlace(op, [&]() { op->setOperand(0, defOpSrc); });
    return ::mlir::success();
  }
};

/// Pattern to rewrite a CastElemTypeOp replacing dynamically shaped result type
/// by statically shaped result type if input is statically shaped.
class CastElemTypeOpResultCanonicalizer final
    : public mlir::OpRewritePattern<::imex::ndarray::CastElemTypeOp> {
public:
  using mlir::OpRewritePattern<
      ::imex::ndarray::CastElemTypeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(::imex::ndarray::CastElemTypeOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    auto src = op.getInput();
    auto srcNDTyp = mlir::dyn_cast<::imex::ndarray::NDArrayType>(src.getType());
    auto resNDTyp =
        mlir::dyn_cast<::imex::ndarray::NDArrayType>(op.getResult().getType());
    if (!(srcNDTyp && resNDTyp && srcNDTyp.hasStaticShape() &&
          !resNDTyp.hasStaticShape())) {
      return mlir::failure();
    }

    auto resShape = srcNDTyp.getShape();
    auto resTyp = resNDTyp.cloneWith(resShape, resNDTyp.getElementType());
    auto newOp = rewriter.create<::imex::ndarray::CastElemTypeOp>(op->getLoc(),
                                                                  resTyp, src);
    rewriter.replaceOpWithNewOp<::imex::ndarray::CastOp>(op, resNDTyp, newOp);

    return ::mlir::success();
  }
};

void imex::ndarray::CastElemTypeOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results
      .add<CastElemTypeOpResultCanonicalizer, CastElemTypeOpInputCanonicalizer>(
          context);
}
