//===- Blocking.h ----------- Blocking Pass ---------*- C++ -*------------===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains lowering transformation for determing the problem size
/// that can be handled by an XeGPU operator (hardware instruction). XeTile
/// program can work one bigger problem size that cannot be handled by a
/// hardware instruction. But it needs to be decomposed into smaller pieces
/// such that each pieces can be handled by a hardware instruction.
/////===----------------------------------------------------------------------===//
#ifndef _XeTileTranformBase_H_INCLUDED_
#define _XeTileTranformBase_H_INCLUDED_

#include <llvm/IR/ValueMap.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/OneToNTypeConversion.h>

#include "imex/Dialect/XeTile/IR/XeTileOps.h"
#include "imex/Utils/DebugUtils.h"
#include "imex/Utils/PassWrapper.h"
#include "imex/Utils/XeCommon.h"

namespace imex {

template <typename SourceOp, typename AnalysisT>
class XeTileConversion : public imex::XeConversionPattern<AnalysisT> {
public:
  using OpAdaptor = typename SourceOp::Adaptor;
  using OpPatternRewriter = typename mlir::PatternRewriter;

  XeTileConversion(mlir::MLIRContext *context, XeTypeConverter &typeConverter,
                   AnalysisT &analysis, mlir::PatternBenefit benefit = 1)
      : XeConversionPattern<AnalysisT>(typeConverter, analysis,
                                       SourceOp::getOperationName(), benefit,
                                       context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override final {
    auto sourceOp = llvm::cast<SourceOp>(op);
    OpAdaptor adaptor(op->getOperands(), sourceOp);
    return matchAndRewrite(sourceOp, adaptor, rewriter);
  }

  virtual mlir::LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const {
    llvm_unreachable("must override matchAndRewrite or a rewrite method");
  }

  xetile::TileUnpackOp addUnpackOp(mlir::Value src,
                                   OpPatternRewriter &rewriter) const {
    auto srcTy = llvm::dyn_cast_if_present<mlir::VectorType>(src.getType());
    assert(srcTy && srcTy.getRank() == 4);
    auto shape = srcTy.getShape();
    auto grids = shape.take_front(2);
    auto innerBlocks = shape.take_back(2);
    llvm::SmallVector<int64_t> unpackShape(
        {grids[0] * innerBlocks[0], grids[1] * innerBlocks[1]});

    auto unpackTy = mlir::VectorType::get(unpackShape, srcTy.getElementType());
    return rewriter.create<xetile::TileUnpackOp>(
        src.getLoc(), unpackTy, src,
        mlir::DenseI64ArrayAttr::get(src.getContext(), innerBlocks));
  }

  mlir::Value addPackOp(mlir::Value src, llvm::ArrayRef<int64_t> targetBlkSizes,
                        OpPatternRewriter &rewriter) const {
    auto srcTy = mlir::dyn_cast<mlir::VectorType>(src.getType());
    assert(srcTy && targetBlkSizes.size() == 2);
    auto shape = srcTy.getShape();
    llvm::SmallVector<int64_t> packShape(
        {shape[0] / targetBlkSizes[0], shape[1] / targetBlkSizes[1],
         targetBlkSizes[0], targetBlkSizes[1]});

    auto packTy = mlir::VectorType::get(packShape, srcTy.getElementType());
    auto packOp = rewriter.create<xetile::TilePackOp>(
        src.getLoc(), packTy, src,
        mlir::DenseI64ArrayAttr::get(src.getContext(), targetBlkSizes));
    return packOp;
  }

  mlir::Value addUnpackAndPackOps(mlir::Value src,
                                  llvm::ArrayRef<int64_t> targetBlocks,
                                  OpPatternRewriter &rewriter) const {
    auto unpack = addUnpackOp(src, rewriter);
    return addPackOp(unpack, targetBlocks, rewriter);
  }
};

template <template <typename> class TraitType, typename AnalysisT>
class XeTileTraitConversion : public imex::XeConversionPattern<AnalysisT> {
public:
  XeTileTraitConversion(mlir::MLIRContext *context,
                        XeTypeConverter &typeConverter, AnalysisT &analysis,
                        mlir::PatternBenefit benefit = 1)
      : XeConversionPattern<AnalysisT>(
            typeConverter, analysis, mlir::Pattern::MatchTraitOpTypeTag(),
            mlir::TypeID::get<TraitType>(), benefit, context) {}
};
} // namespace imex

#endif
