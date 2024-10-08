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
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/Support/Debug.h>

#include <algorithm>
#include <optional>

#include "imex/Dialect/XeTile/Transforms/Blocking.h"
#include "imex/Dialect/XeTile/Transforms/Passes.h"
#include "imex/Utils/DebugUtils.h"
#include "imex/Utils/XeArch.h"

using namespace mlir;
using namespace imex;
namespace imex {
#define GEN_PASS_DEF_XETILEBLOCKING
#include "imex/Dialect/XeTile/Transforms/Passes.h.inc"
} // namespace imex

namespace imex {

extern void
populateXeTileBlockAligningPatterns(imex::XeTypeConverter &converter,
                                    mlir::RewritePatternSet &patterns,
                                    PropagateAnalysis &analysis);

enum OpType { Prefetch, Load, Store, Elementwise, Transpose };

// Find the maximum divisible number between minHeight/Width and maxHeight/Width
// and use that as the inner block sizes.
int findMaxInnerBlockSize(int num, int maxNum, int minNum) {
  for (int i = maxNum; i >= minNum; i--) {
    if (num % i == 0) {
      return i;
    }
  }
  return -1;
}

llvm::SmallVector<int64_t, 2>
getInnerBlockHeightWidth(int maxHeight, int maxWidth, int minHeight,
                         int minWidth, int height, int width) {

  llvm::SmallVector<int64_t, 2> innerBlockSizes;

  if (height < minHeight || width < minWidth) {
    return {};
  }

  if (height == maxHeight || height < maxHeight) {
    innerBlockSizes.push_back(height);
  } else if (height > maxHeight) {
    auto innerBlockHeight =
        imex::findMaxInnerBlockSize(height, maxHeight, minHeight);
    if (innerBlockHeight != -1)
      innerBlockSizes.push_back(innerBlockHeight);
    else {
      llvm::dbgs() << "Invalid Block Height Shape \n";
      return {};
    }
  }

  if (width == maxWidth || width < maxWidth) {
    innerBlockSizes.push_back(width);
  } else if (width > maxWidth) {
    auto innerBlockWidth =
        imex::findMaxInnerBlockSize(width, maxWidth, minWidth);
    if (innerBlockWidth != -1)
      innerBlockSizes.push_back(innerBlockWidth);
    else {
      llvm::dbgs() << "Invalid Block Width Shape \n";
      return {};
    }
  }
  return innerBlockSizes;
}

// TODO: placeholder, replace it with uArch interface
template <OpType op>
llvm::SmallVector<int64_t, 2>
getInnerBlockSizes(mlir::Operation *operation, mlir::Type elemTy, int height,
                   int width, std::shared_ptr<XeuArchInterface> uArchInterface,
                   bool vnni = false, bool transpose = false) {
  assert(elemTy.isIntOrFloat());
  int elementSize = elemTy.getIntOrFloatBitWidth();
  if (op == OpType::Load && elementSize > 16 && vnni) {
    llvm::dbgs() << "load with VNNI for \"" << elemTy
                 << "\" is not supported.\n";
    return {};
  }

  int maxHeight, maxWidth, minHeight, minWidth;

  if (op == OpType::Load) {

    mlir::FailureOr<LoadStore2DConfig> params = uArchInterface->get2DLoadConfig(
        operation, elementSize, vnni, transpose);
    if (mlir::succeeded(params)) {
      maxHeight = params->blockHeight.max;
      minHeight = params->blockHeight.min;
      maxWidth = params->blockWidth.max;
      minWidth = params->blockWidth.min;
    } else {
      llvm::dbgs() << "Invalid Config Params \n";
      return {};
    }

    return imex::getInnerBlockHeightWidth(maxHeight, maxWidth, minHeight,
                                          minWidth, height, width);
  }

  if (op == OpType::Prefetch) {

    mlir::FailureOr<LoadStore2DConfig> params =
        uArchInterface->get2DPrefetchConfig(operation, elementSize);
    if (mlir::succeeded(params)) {
      maxHeight = params->blockHeight.max;
      minHeight = params->blockHeight.min;
      maxWidth = params->blockWidth.max;
      minWidth = params->blockWidth.min;
    } else {
      llvm::dbgs() << "Invalid Config Params \n";
      return {};
    }

    return imex::getInnerBlockHeightWidth(maxHeight, maxWidth, minHeight,
                                          minWidth, height, width);
  }

  if (op == OpType::Store) {

    mlir::FailureOr<LoadStore2DConfig> params =
        uArchInterface->get2DStoreConfig(elementSize);
    if (mlir::succeeded(params)) {
      maxHeight = params->blockHeight.max;
      minHeight = params->blockHeight.min;
      maxWidth = params->blockWidth.max;
      minWidth = params->blockWidth.min;
    } else {
      llvm::dbgs() << "Invalid Config Params \n";
      return {};
    }

    return imex::getInnerBlockHeightWidth(maxHeight, maxWidth, minHeight,
                                          minWidth, height, width);
  }

  if (op == OpType::Elementwise) {
    int64_t subgroupSize = uArchInterface->getOneGRFSizeBits() / elementSize;

    maxHeight = 1;
    minHeight = 1;
    maxWidth = subgroupSize;
    minWidth = 1;

    return imex::getInnerBlockHeightWidth(maxHeight, maxWidth, minHeight,
                                          minWidth, height, width);
  }

  if (op == OpType::Transpose) {
    // TODO: get from uArch?
    maxHeight = 16;
    minHeight = 1;
    maxWidth = 8;
    minWidth = 1;

    return imex::getInnerBlockHeightWidth(maxHeight, maxWidth, minHeight,
                                          minWidth, height, width);
  }

  llvm_unreachable("Unsupported.");
  return {};
}

// works similar to getUsers. If the user is a SCF::ForOp,
// it will return the users of corresponding scf::ForOp argument.
// TODO: make it to be general to handle composite ops, e.g,
// SCF::ForOp, SCF::WhileOp, etc.
static llvm::SmallVector<mlir::Operation *> getEffectiveUsers(mlir::Value val) {
  llvm::SmallVector<mlir::Operation *> users;
  for (auto user : val.getUsers()) {
    if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(user)) {
      auto arg = getArgForOperand(forOp, val);
      users.append(arg.user_begin(), arg.user_end());
    } else {
      users.push_back(user);
    }
  }
  return users;
}

static llvm::SmallVector<unsigned int>
getMMASize(mlir::Type elemTy, const int APrecision, const int BPrecision,
           const int CPrecision, const int DPrecision,
           std::shared_ptr<XeuArchInterface> uArchInterface) {
  assert(elemTy.isIntOrFloat() && "unsupported element type.");
  auto dpasParams = uArchInterface->getDPASConfig(APrecision, BPrecision,
                                                  CPrecision, DPrecision);
  return llvm::SmallVector<unsigned int>(
      {dpasParams.m, dpasParams.k, dpasParams.n});
}

// It blocks/extends a 2D constant dense vector into a
// 4D vector with the last 2 dim corresponding to block size.
// example: arith.constant dense<0.0>: vector<32x32xf16>
//      --> arith.constant dense<0.0>: vector<4x2x8x16xf16>
// [8, 16] is the block size.
struct ArithConstantOpPattern
    : public XeTileConversion<mlir::arith::ConstantOp, TileUsageAnalysis> {

  using XeTileConversion<mlir::arith::ConstantOp,
                         TileUsageAnalysis>::XeTileConversion;

  ArithConstantOpPattern(mlir::MLIRContext *context,
                         imex::XeTypeConverter &converter,
                         TileUsageAnalysis &analysis,
                         std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter, analysis) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto value = llvm::dyn_cast<mlir::DenseElementsAttr>(op.getValue());

    // TODO: it maybe unstable to determine whether doing blocking or not
    //  for a constant op simply based on its 2D shape.
    if (!value || value.getType().getRank() != 2)
      return mlir::failure();

    auto shape = value.getType().getShape();
    auto blkSZ =
        getInnerBlockSizes<Load>(op.getOperation(), value.getElementType(),
                                 shape[0], shape[1], this->uArchInterface);
    if (blkSZ.empty())
      return rewriter.notifyMatchFailure(op, "Invalid inner block sizes");

    auto newTy = mlir::VectorType::get(
        {shape[0] / blkSZ[0], shape[1] / blkSZ[1], blkSZ[0], blkSZ[1]},
        value.getElementType());

    // TODO: it is logically incorrect to reshape a dense value to a vector
    // type.
    //  it doesnot show the impact of pack effect. It works on some specific
    //  cases in which all elements has the same value, but not general.
    value = value.reshape(newTy);
    auto newOp = rewriter.create<mlir::arith::ConstantOp>(loc, value);
    auto unpack = addUnpackOp(newOp, rewriter);

    rewriter.replaceOp(op, unpack);
    return mlir::success();
  }
};

// Blocks a vector.create_mask op such that, ideally, it matches its consuming
// select op (which is an elementwise op). In this way, during the lowering to
// XeGPU, there will be a one-to-one correspondence and an
// unrealized_conversion_cast will not be needed.
struct VectorCreateMaskOpPattern
    : public XeTileConversion<mlir::vector::CreateMaskOp, TileUsageAnalysis> {

  using XeTileConversion<mlir::vector::CreateMaskOp,
                         TileUsageAnalysis>::XeTileConversion;

  VectorCreateMaskOpPattern(mlir::MLIRContext *context,
                            imex::XeTypeConverter &converter,
                            TileUsageAnalysis &analysis,
                            std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter, analysis) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::CreateMaskOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto res = op.getResult();
    auto resType = res.getType();
    if (resType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "type is not 2D vector");

    // Only two cases are supported for now:
    // 1.The first operand is a constant equal to the first dimension of the
    // output shape (i.e., all rows are enabled). In other words, masking
    // columns within a row is supported.
    // 2.The second operand is a constant equal to the second dimension of the
    // output shape (i.e., all columns are enabled).
    auto shape = resType.getShape();
    APInt cstOp0, cstOp1;
    if (!(matchPattern(op->getOperand(0), m_ConstantInt(&cstOp0)) &&
          cstOp0.getSExtValue() == shape[0]) &&
        !(matchPattern(op->getOperand(1), m_ConstantInt(&cstOp1)) &&
          cstOp1.getSExtValue() == shape[1])) {
      op->emitOpError() << "Unsupported operands";
      return mlir::failure();
    }

    auto blocks = getInnerBlockSizes<Elementwise>(
        op, resType.getElementType(), shape[0], shape[1], this->uArchInterface);

    if (blocks.empty()) {
      op->emitOpError() << "Invalid inner block sizes";
      return mlir::failure();
    }

    // TODO: support blocking the outer dimension.
    if (blocks[0] != 1) {
      op->emitOpError() << "Unsupported inner block sizes";
      return mlir::failure();
    }

    auto newTy = mlir::VectorType::get(
        {shape[0] / blocks[0], shape[1] / blocks[1], blocks[0], blocks[1]},
        resType.getElementType());
    Location loc = op->getLoc();
    rewriter.startOpModification(op);
    // Due to the simplifications mentioned above, for now, the index operands
    // are not adjusted. In fact, only the first index operand (masked rows) or
    // the second index operand (masked columns) will be used during the
    // lowering to XeGPU.
    op->setOperands({op->getOperand(0), op->getOperand(1), op->getOperand(0),
                     op->getOperand(1)});
    res.setType(newTy);
    rewriter.finalizeOpModification(op);
    rewriter.setInsertionPointAfter(op);
    auto unpack = rewriter.create<xetile::TileUnpackOp>(
        loc, resType, res, mlir::DenseI64ArrayAttr::get(getContext(), blocks));
    rewriter.replaceAllUsesExcept(res, unpack.getResult(), unpack);
    return mlir::success();
  }
};

// Pattern for generic elemetwise ops. Blocks op according to
// getInnerBlocks<Elementwise>. Pack/Unpack ops are inserted on the ops
// boundaries if needed.
struct VectorizableOpPattern
    : public XeTileTraitConversion<mlir::OpTrait::Vectorizable,
                                   TileUsageAnalysis> {

  using XeTileTraitConversion::XeTileTraitConversion;

  VectorizableOpPattern(mlir::MLIRContext *context,
                        imex::XeTypeConverter &converter,
                        TileUsageAnalysis &analysis,
                        std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileTraitConversion(context, converter, analysis) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "op must have 1 result");

    auto res = op->getResult(0);
    auto resType = mlir::dyn_cast<mlir::VectorType>(res.getType());
    if (!resType || resType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "type is not 2D vector");

    auto shape = resType.getShape();
    auto blocks = getInnerBlockSizes<Elementwise>(
        op, resType.getElementType(), shape[0], shape[1], this->uArchInterface);

    if (blocks.empty()) {
      op->emitOpError() << "Invalid inner block sizes ";
      return mlir::failure();
    }

    auto newTy = mlir::VectorType::get(
        {shape[0] / blocks[0], shape[1] / blocks[1], blocks[0], blocks[1]},
        resType.getElementType());

    Location loc = op->getLoc();
    rewriter.startOpModification(op);
    for (auto &&[i, arg] : llvm::enumerate(op->getOperands())) {
      auto srcTy = mlir::dyn_cast<mlir::VectorType>(arg.getType());
      if (!srcTy || srcTy.getRank() != 2)
        continue;

      auto unpackShape = srcTy.getShape();
      int64_t packShape[] = {unpackShape[0] / blocks[0],
                             unpackShape[1] / blocks[1], blocks[0], blocks[1]};

      auto packTy = mlir::VectorType::get(packShape, srcTy.getElementType());
      mlir::Value packOp = rewriter.create<xetile::TilePackOp>(
          loc, packTy, arg, mlir::DenseI64ArrayAttr::get(getContext(), blocks));

      op->setOperand(i, packOp);
    }

    res.setType(newTy);
    rewriter.finalizeOpModification(op);

    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(op);
    auto unpack = rewriter.create<xetile::TileUnpackOp>(
        loc, resType, res, mlir::DenseI64ArrayAttr::get(getContext(), blocks));

    rewriter.replaceAllUsesExcept(res, unpack.getResult(), unpack);
    return mlir::success();
  }
};

template <typename OpTy>
struct TransposeOpPattern : public XeTileConversion<OpTy, TileUsageAnalysis> {

  using XeTileConversion<OpTy, TileUsageAnalysis>::XeTileConversion;
  using OpAdaptor = typename OpTy::Adaptor;

  TransposeOpPattern(mlir::MLIRContext *context,
                     imex::XeTypeConverter &converter,
                     TileUsageAnalysis &analysis,
                     std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion<OpTy, TileUsageAnalysis>(context, converter,
                                                  analysis) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  mlir::LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  mlir::PatternRewriter &rewriter) const override {
    auto res = op.getResult();
    auto resType = mlir::cast<mlir::VectorType>(res.getType());
    if (resType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "type is not 2D vector");

    auto permutation = op.getPermutation();
    if (permutation != mlir::ArrayRef<int64_t>({1, 0}))
      return rewriter.notifyMatchFailure(op, "Unsupported permutation");

    auto shape = resType.getShape();
    auto blocks = getInnerBlockSizes<Transpose>(
        op, resType.getElementType(), shape[0], shape[1], this->uArchInterface);

    if (blocks.size() != 2)
      return rewriter.notifyMatchFailure(op, "Invalid inner block sizes");

    int64_t inBlocks[2] = {blocks[1], blocks[0]};

    auto newSrcTy = mlir::VectorType::get(
        {shape[1] / blocks[1], shape[0] / blocks[0], blocks[1], blocks[0]},
        resType.getElementType());

    auto newDstTy = mlir::VectorType::get(
        {shape[0] / blocks[0], shape[1] / blocks[1], blocks[0], blocks[1]},
        resType.getElementType());

    mlir::Value arg = adaptor.getVector();

    Location loc = op->getLoc();
    mlir::Value pack = rewriter.create<xetile::TilePackOp>(
        loc, newSrcTy, arg,
        mlir::DenseI64ArrayAttr::get(op.getContext(), inBlocks));

    int64_t newPermutation[4] = {1, 0, 3, 2};
    mlir::Value transpose =
        rewriter.create<OpTy>(loc, newDstTy, pack, newPermutation);

    mlir::Value unpack = rewriter.create<xetile::TileUnpackOp>(
        loc, resType, transpose,
        mlir::DenseI64ArrayAttr::get(op.getContext(), blocks));

    rewriter.replaceOp(op, unpack);

    return mlir::success();
  }
};

struct VectorMultiDimReductionOpPattern
    : public XeTileConversion<mlir::vector::MultiDimReductionOp,
                              TileUsageAnalysis> {
  using XeTileConversion<mlir::vector::MultiDimReductionOp,
                         TileUsageAnalysis>::XeTileConversion;

  VectorMultiDimReductionOpPattern(mlir::MLIRContext *context,
                                   imex::XeTypeConverter &converter,
                                   TileUsageAnalysis &analysis,
                                   std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter, analysis) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::MultiDimReductionOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto reductionDims = op.getReductionDims();
    auto srcTy = op.getSource().getType();
    auto shape = srcTy.getShape();

    if (srcTy.getRank() != 2 || reductionDims.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "source type is not 2D vector or reduction dims are not 1");

    auto blkSizes = getInnerBlockSizes<Elementwise>(
        op, srcTy.getElementType(), shape[0], shape[1], this->uArchInterface);

    if (blkSizes.empty())
      return rewriter.notifyMatchFailure(op, "Invalid inner block sizes");

    // reduction on one dim becomes reduction on two dims after blocking.
    // For example:
    // multi_reduction<add>, %e, %a[1]: vector<16x32xf16> to vector<16xf16>
    // will be transformed to
    // multi_reduction<add>, %e, %a[1, 3]: vector<16x2x1x16xf16> to
    // vector<16x1xf16>
    auto dim = reductionDims[0];
    auto newReductionDims = rewriter.getDenseI64ArrayAttr({dim, dim + 2});

    auto newDestShape =
        (dim == 0)
            ? llvm::SmallVector<int64_t>({shape[1] / blkSizes[1], blkSizes[1]})
            : llvm::SmallVector<int64_t>({shape[0] / blkSizes[0], blkSizes[0]});
    auto newDestType =
        mlir::VectorType::get(newDestShape, srcTy.getElementType());

    auto newSource =
        addPackOp(adaptor.getSource(), {blkSizes[0], blkSizes[1]}, rewriter);
    auto newAcc = rewriter.create<mlir::vector::ShapeCastOp>(loc, newDestType,
                                                             adaptor.getAcc());
    auto newOp = rewriter.create<mlir::vector::MultiDimReductionOp>(
        loc, newDestType, op.getKindAttr(), newSource, newAcc,
        newReductionDims);
    auto castOp = rewriter.create<mlir::vector::ShapeCastOp>(
        loc, op.getDest().getType(), newOp);
    rewriter.replaceOp(op, castOp.getResult());

    return mlir::success();
  }
};

struct TileReductionOpPattern
    : public XeTileConversion<xetile::ReductionOp, TileUsageAnalysis> {

  using XeTileConversion<xetile::ReductionOp,
                         TileUsageAnalysis>::XeTileConversion;

  TileReductionOpPattern(mlir::MLIRContext *context,
                         imex::XeTypeConverter &converter,
                         TileUsageAnalysis &analysis,
                         std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter, analysis) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  mlir::LogicalResult
  matchAndRewrite(xetile::ReductionOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto srcTy = op.getSource().getType();
    auto elemTy = srcTy.getElementType();
    auto shape = srcTy.getShape();
    auto reductionDims = op.getReductionDims();

    if (srcTy.getRank() != 2 || reductionDims.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "source type is not 2D vector or reduction dims are not 1");

    auto blkSizes = getInnerBlockSizes<Elementwise>(
        op, elemTy, shape[0], shape[1], this->uArchInterface);

    if (blkSizes.empty())
      return rewriter.notifyMatchFailure(op, "Invalid inner block sizes");

    // reduction on one dim becomes reduction on two dims after blocking.
    // For example:
    // reduce<add>, %e [1]: vector<16x32xf16> to vector<16xf16>
    // will be transformed to
    // reduce<add>, %e [1, 3]: vector<16x2x1x16xf16> to
    // vector<16x1xf16>
    auto dim = reductionDims[0];
    auto newReductionDims =
        mlir::DenseI64ArrayAttr::get(op.getContext(), {dim, dim + 2});
    llvm::SmallVector<int64_t> newDestShape({shape[0] / blkSizes[0],
                                             shape[1] / blkSizes[1],
                                             blkSizes[0], blkSizes[1]});
    for (auto dim : newReductionDims.asArrayRef()) {
      newDestShape[dim] = 1;
    }

    auto newDestType =
        mlir::VectorType::get(newDestShape, srcTy.getElementType());

    auto newSource =
        addPackOp(adaptor.getSource(), {blkSizes[0], blkSizes[1]}, rewriter);
    auto newDest = rewriter.create<xetile::ReductionOp>(
        loc, newDestType, op.getKind(), newSource, newReductionDims);
    auto unpack = addUnpackOp(newDest.getResult(), rewriter);
    rewriter.replaceOp(op, unpack);
    return mlir::success();
  }
};

struct TileBroadcastOpPattern
    : public XeTileConversion<xetile::BroadcastOp, TileUsageAnalysis> {

  using XeTileConversion<xetile::BroadcastOp,
                         TileUsageAnalysis>::XeTileConversion;

  TileBroadcastOpPattern(mlir::MLIRContext *context,
                         imex::XeTypeConverter &converter,
                         TileUsageAnalysis &analysis,
                         std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter, analysis) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  mlir::LogicalResult
  matchAndRewrite(xetile::BroadcastOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto srcTy = op.getSource().getType();
    auto elemTy = srcTy.getElementType();
    auto shape = srcTy.getShape();
    auto broadcastDims = op.getBroadcastDim();

    if (srcTy.getRank() != 2 || broadcastDims.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "source type is not 2D vector or rank of broadcastDims is not 1");

    auto inBlkSizes = getInnerBlockSizes<Elementwise>(
        op, elemTy, shape[0], shape[1], this->uArchInterface);

    auto outShape = op.getResult().getType().getShape();
    auto outBlkSizes = getInnerBlockSizes<Elementwise>(
        op, elemTy, outShape[0], outShape[1], this->uArchInterface);

    if (inBlkSizes.empty() || outBlkSizes.empty())
      return rewriter.notifyMatchFailure(op, "Invalid inner block sizes");

    llvm::SmallVector<int64_t> newDestShape({outShape[0] / outBlkSizes[0],
                                             outShape[1] / outBlkSizes[1],
                                             outBlkSizes[0], outBlkSizes[1]});

    auto newSource = addPackOp(adaptor.getSource(),
                               {inBlkSizes[0], inBlkSizes[1]}, rewriter);

    auto newDestType =
        mlir::VectorType::get(newDestShape, srcTy.getElementType());

    // broadcast on one dim becomes broadcast on two dims after blocking.
    // For example:
    // broadcast %a [0]: vector<1x32xf16> to vector<16x32xf16>
    // will be transformed to
    // broadcast %a [0, 2]: vector<1x2x1x16xf16> to vector<16x2x1x16xf16>
    auto dim = broadcastDims[0];
    auto newBroadcastDims =
        mlir::DenseI64ArrayAttr::get(op.getContext(), {dim, dim + 2});
    auto newDest = rewriter.create<xetile::BroadcastOp>(
        loc, newDestType, newSource, newBroadcastDims);
    auto unpack = addUnpackOp(newDest.getResult(), rewriter);
    rewriter.replaceOp(op, unpack);
    return mlir::success();
  }
};

// It rewrites the SCF forOp, it mainly updates the arguments of its
// region block. unpack ops are added for VectorType operands if needed.
struct SCFForOpPattern
    : public XeTileConversion<mlir::scf::ForOp, TileUsageAnalysis> {

  using XeTileConversion<mlir::scf::ForOp, TileUsageAnalysis>::XeTileConversion;

  SCFForOpPattern(mlir::MLIRContext *context, imex::XeTypeConverter &converter,
                  TileUsageAnalysis &analysis,
                  std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter, analysis) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  ::mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    // we don't need to update the forOp if it has no region
    // iter args, or the region iter args type are not changed.
    bool changed = false;
    for (unsigned i = 0; i < op.getNumRegionIterArgs(); i++) {
      auto initArg = adaptor.getInitArgs()[i];
      auto regionArg = op.getRegionIterArg(i);
      changed |= (initArg.getType() != regionArg.getType()) ||
                 bool(initArg.getDefiningOp<xetile::TileUnpackOp>());
    }
    if (!changed)
      return mlir::failure();

    // preprocess the init args and remove the unpackOp by
    // adding pack op with the same innerblock size, so that
    // they can be folded (removed).
    llvm::SmallVector<mlir::Value> newInitArgs;
    for (auto arg : adaptor.getInitArgs()) {
      if (auto defOp = arg.getDefiningOp<xetile::TileUnpackOp>()) {
        auto packOp = addPackOp(arg, defOp.getInnerBlocks(), rewriter);
        newInitArgs.push_back(packOp);
      } else {
        newInitArgs.push_back(arg);
      }
    }

    auto newOp = rewriter.create<mlir::scf::ForOp>(
        op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), newInitArgs);
    mlir::Block *newBlock = newOp.getBody();
    // remove the terminator of the new block
    if (newBlock->mightHaveTerminator())
      rewriter.eraseOp(newBlock->getTerminator());

    auto savedIP = rewriter.saveInsertionPoint();
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(newBlock);

    // contruct the inputs for the new scf::for block.
    // An unpackOp is inserted for the corresponding init arg
    // of the new block if its init value is updated with a pack op.
    llvm::SmallVector<mlir::Value> newArguments;
    for (auto [i, arg] : llvm::enumerate(newBlock->getArguments())) {
      if (i && newInitArgs[i - 1].getDefiningOp<xetile::TilePackOp>()) {
        auto unpack = addUnpackOp(arg, rewriter);
        newArguments.push_back(unpack);
      } else {
        newArguments.push_back(arg);
      }
    }

    rewriter.restoreInsertionPoint(savedIP);

    mlir::Block *block = op.getBody();
    rewriter.mergeBlocks(block, newBlock, newArguments);

    llvm::SmallVector<mlir::Value> newValues;
    for (auto [i, result] : llvm::enumerate(newOp->getResults())) {
      // if corresponding init arg is updated with a pack op
      // an unpack op is needed for the result to make it
      // transparent to its users.
      if (newInitArgs[i].getDefiningOp<xetile::TilePackOp>()) {
        auto unpack = addUnpackOp(result, rewriter);
        newValues.push_back(unpack);
      } else {
        newValues.push_back(result);
      }
    }
    rewriter.replaceOp(op, newValues);
    return mlir::success();
  }
};

// It serves to insert pack ops for approriate vales if needed.
// for example, tile_mma result is vector<32x32xf16> (after unpack),
// but its corresponding argument in forOp is with type vector<1x2x32x16xf16>
// This op pattern will insert a pack op to make it consistent with the
// corresponding argument type.
struct SCFYieldOpPattern
    : public XeTileConversion<mlir::scf::YieldOp, TileUsageAnalysis> {

  using XeTileConversion<mlir::scf::YieldOp,
                         TileUsageAnalysis>::XeTileConversion;

  SCFYieldOpPattern(mlir::MLIRContext *context,
                    imex::XeTypeConverter &converter,
                    TileUsageAnalysis &analysis,
                    std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter, analysis) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  ::mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto forOp = op->getParentOfType<mlir::scf::ForOp>();
    if (!forOp)
      return mlir::failure();

    bool changed = false;
    for (auto [i, v] : llvm::enumerate(adaptor.getResults())) {
      if (auto defOp = v.getDefiningOp<xetile::TileUnpackOp>()) {
        // get InnerBlock size from the corresponding output type
        auto ty =
            mlir::dyn_cast<mlir::VectorType>(forOp->getResult(i).getType());
        if (ty && ty.getRank() == 4) {
          auto innerBlock = ty.getShape().take_back(2);
          auto packOp = addPackOp(v, innerBlock, rewriter);
          rewriter.startOpModification(op);
          op->setOperand(i, packOp);
          rewriter.finalizeOpModification(op);
          changed = true;
        }
      }
    }
    return mlir::success(changed);
  }
};

// It updates init_tile by attaching innerBlock attribute to the result
// tile. The block size is choosed based on how the tile is used, including
// prefetch, load, store. Since hardware support different sizes for them.
struct InitTileOpPattern
    : public XeTileConversion<xetile::InitTileOp, TileUsageAnalysis> {

  using XeTileConversion<xetile::InitTileOp,
                         TileUsageAnalysis>::XeTileConversion;

  InitTileOpPattern(mlir::MLIRContext *context,
                    imex::XeTypeConverter &converter,
                    TileUsageAnalysis &analysis,
                    std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter, analysis) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::InitTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto tileTy = op.getType();
    if (tileTy.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "Skipped InitTileOp because the result tile is not rank 2.\n");

    auto innerBlocks = tileTy.getInnerBlocks();
    auto memorySpace = op.getSourceMemorySpaceAsInt();

    // skip it if innerBlocks has been set by user or compiler.
    if (innerBlocks)
      return mlir::failure();

    auto elemTy = tileTy.getElementType();
    int elementSize = elemTy.getIntOrFloatBitWidth();

    if (memorySpace == 3) {                    // for shared memory
      const unsigned int lscConstraints = 512; // 512 bytes constraint by lsc
      const unsigned int subgroupSize = 16;
      auto shape = tileTy.getShape();
      int64_t innerBlockSizes[2];
      // prefer to use gather loads with 16 simd lanes
      innerBlockSizes[0] = shape[0] % subgroupSize == 0 ? 16 : 1;
      innerBlockSizes[1] =
          (lscConstraints * 8) / (elementSize * innerBlockSizes[0]);
      innerBlockSizes[1] =
          std::min<int64_t>(innerBlockSizes[1], tileTy.getShape()[1]);
      innerBlocks = mlir::DenseI64ArrayAttr::get(getContext(), innerBlockSizes);
    } else { // for global memory
      if (isForPrefetch(op)) {
        innerBlocks = mlir::DenseI64ArrayAttr::get(
            getContext(), getInnerBlockSizes<Prefetch>(
                              op.getOperation(), elemTy, tileTy.getShape()[0],
                              tileTy.getShape()[1], this->uArchInterface));
      } else if (isForLoad(op)) {

        // Set transpose and vnni
        bool vnni = false;
        bool transpose = false;

        auto order = tileTy.getOrder();
        if (order[0] == 0 && order[1] == 1)
          transpose = true;

        for (auto user : getEffectiveUsers(op)) {
          if (auto loadTileOp = llvm::dyn_cast<xetile::LoadTileOp>(user)) {
            if (isForDPASB(loadTileOp) && elementSize < 32) {
              vnni = true;
              break;
            }
          }
        }

        if (vnni && transpose && elementSize < 32) {
          int factor = 32 / elementSize;
          vnni = false;
          llvm::SmallVector<int64_t, 2> innerBlock = getInnerBlockSizes<Load>(
              op.getOperation(), mlir::FloatType::getF32(getContext()),
              tileTy.getShape()[1], (tileTy.getShape()[0]) / factor,
              this->uArchInterface, vnni, transpose);
          std::swap(innerBlock[0], innerBlock[1]);
          innerBlock[0] *= factor;
          innerBlocks = mlir::DenseI64ArrayAttr::get(getContext(), innerBlock);

        } else if (transpose && elementSize < 32) {
          return rewriter.notifyMatchFailure(op, "Invalid transpose.");
        } else {
          innerBlocks = mlir::DenseI64ArrayAttr::get(
              getContext(),
              getInnerBlockSizes<Load>(
                  op.getOperation(), elemTy, tileTy.getShape()[0],
                  tileTy.getShape()[1], this->uArchInterface, vnni, transpose));
        }
      } else if (isForStore(op)) {
        innerBlocks = mlir::DenseI64ArrayAttr::get(
            getContext(), getInnerBlockSizes<Store>(
                              op.getOperation(), elemTy, tileTy.getShape()[0],
                              tileTy.getShape()[1], this->uArchInterface));
      } else {
        return rewriter.notifyMatchFailure(
            op,
            "The tile is used for multiple purpose. The init-duplicate pass "
            "should be run first to resolve this issue.");
      }
    }

    if (innerBlocks.empty()) {
      op->emitOpError() << "Invalid inner block sizes ";
      return mlir::failure();
    }

    auto attr = imex::xetile::XeTileAttr::get(
        op.getContext(), tileTy.getSgMap(), tileTy.getWgMap(),
        tileTy.getOrder(), innerBlocks, tileTy.getMemorySpace());

    auto newTileTy =
        imex::xetile::TileType::get(tileTy.getShape(), elemTy, attr);

    rewriter.startOpModification(op);
    op.getTile().setType(newTileTy);
    rewriter.finalizeOpModification(op);

    return mlir::success();
  }
};

// It updates load_tile to reveal effects of innerblock attribute by
// representing value as 4D vector. An unpack op is added at the end
// to make this change to be transparent to its users.
struct LoadTileOpPattern
    : public XeTileConversion<xetile::LoadTileOp, TileUsageAnalysis> {

  using XeTileConversion<xetile::LoadTileOp,
                         TileUsageAnalysis>::XeTileConversion;

  LoadTileOpPattern(mlir::MLIRContext *context,
                    imex::XeTypeConverter &converter,
                    TileUsageAnalysis &analysis,
                    std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter, analysis) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::LoadTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto tileTy = op.getSource().getType();
    auto shape = tileTy.getShape();
    auto innerBlocks = tileTy.getInnerBlocks();
    auto rank = op.getValue().getType().getRank();

    if (!innerBlocks || rank == 4)
      return rewriter.notifyMatchFailure(
          op, "Input is not updated or the op has been updated.\n");

    auto vecTy = ::mlir::VectorType::get({shape[0] / innerBlocks[0],
                                          shape[1] / innerBlocks[1],
                                          innerBlocks[0], innerBlocks[1]},
                                         tileTy.getElementType());

    rewriter.startOpModification(op);
    op.getValue().setType(vecTy);
    rewriter.finalizeOpModification(op);
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(op);
    auto unpack = addUnpackOp(op, rewriter);
    rewriter.replaceAllUsesExcept(op, unpack, unpack);

    return mlir::success();
  }
};

// It updates store_tile to reveal effects of innerblock attribute.
// It uses pack op to align the shape of its vector value to the tile shape.
struct StoreTileOpPattern
    : public XeTileConversion<xetile::StoreTileOp, TileUsageAnalysis> {

  using XeTileConversion<xetile::StoreTileOp,
                         TileUsageAnalysis>::XeTileConversion;

  StoreTileOpPattern(mlir::MLIRContext *context,
                     imex::XeTypeConverter &converter,
                     TileUsageAnalysis &analysis,
                     std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter, analysis) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::StoreTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto tileTy = llvm::dyn_cast<xetile::TileType>(adaptor.getTile().getType());
    auto innerBlocks = tileTy.getInnerBlocks();
    auto value = adaptor.getValue();
    auto valTy = mlir::dyn_cast<mlir::VectorType>(value.getType());

    // its inputs has not been updated yet.
    if (innerBlocks && valTy.getRank() == 2) {
      value = addPackOp(value, innerBlocks.asArrayRef(), rewriter);
      rewriter.replaceOpWithNewOp<xetile::StoreTileOp>(op, value,
                                                       adaptor.getTile());
      return mlir::success();
    }
    return mlir::failure();
  }
};

// It updates tile_mma to reveal effects of innerblock attribute.
// Values will be reprented as 4D vectors. An unpack op is applied
// to its result to make the change transparent to its users.
struct TileMMAOpPattern
    : public XeTileConversion<xetile::TileMMAOp, TileUsageAnalysis> {

  using XeTileConversion<xetile::TileMMAOp,
                         TileUsageAnalysis>::XeTileConversion;

  TileMMAOpPattern(mlir::MLIRContext *context, imex::XeTypeConverter &converter,
                   TileUsageAnalysis &analysis,
                   std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter, analysis) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::TileMMAOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto resultTy = op.getResult().getType();
    if (resultTy.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "The result of tile_mma must be 2D vector.\n");

    auto a = adaptor.getA();
    auto b = adaptor.getB();
    auto c = adaptor.getC();

    assert(a && b && "a operand or b operand is (are) missing.\n");

    // expecting a, b, c are either 2D vectors defined by UnpackOp or 4D vectors
    // when its operands has been processed. Otherwise, inputs has not been
    // fully processed and it is not the right time to update the tile_mma op.
    if ((!a.getDefiningOp<xetile::TileUnpackOp>() &&
         mlir::cast<VectorType>(a.getType()).getRank() != 4) ||
        (!b.getDefiningOp<xetile::TileUnpackOp>() &&
         mlir::cast<VectorType>(b.getType()).getRank() != 4) ||
        (c && !c.getDefiningOp<xetile::TileUnpackOp>() &&
         mlir::cast<VectorType>(c.getType()).getRank() != 4))
      return rewriter.notifyMatchFailure(
          op, "expecting a, b, and c are either 2D vectors defined by "
              "unpackOps or 4D vectors.\n");

    unsigned int CPrecision = resultTy.getElementType().getIntOrFloatBitWidth();
    if (c)
      CPrecision = op.getC().getType().getElementType().getIntOrFloatBitWidth();

    auto mmaSize = getMMASize(
        op.getElementType(),
        op.getAType().getElementType().getIntOrFloatBitWidth(),
        op.getBType().getElementType().getIntOrFloatBitWidth(), CPrecision,
        op.getResult().getType().getElementType().getIntOrFloatBitWidth(),
        this->uArchInterface);

    // packing a, b, c accordingly to with the expected innerblock size.
    // if they are defined by unpackOps, packOps are added, if they are
    // 4D vectors, pairs of unpack and pack ops are added.
    auto packing = [&](mlir::Value val, llvm::ArrayRef<int64_t> shape,
                       OpPatternRewriter &rewriter) -> mlir::Value {
      if (val.getDefiningOp<xetile::TileUnpackOp>())
        return addPackOp(val, shape, rewriter);
      return addUnpackAndPackOps(val, shape, rewriter);
    };
    a = packing(a, {mmaSize[0], mmaSize[1]}, rewriter);
    b = packing(b, {mmaSize[1], mmaSize[2]}, rewriter);
    if (c)
      c = packing(c, {mmaSize[0], mmaSize[2]}, rewriter);

    assert(mlir::dyn_cast<VectorType>(a.getType()).getRank() == 4 &&
           mlir::dyn_cast<VectorType>(b.getType()).getRank() == 4 &&
           (!c || mlir::dyn_cast<VectorType>(c.getType()).getRank() == 4) &&
           "a, b and c (if has) should be transformed into 4D vectors.\n");

    auto shape = resultTy.getShape();
    auto vecTy = ::mlir::VectorType::get(
        {shape[0] / mmaSize[0], shape[1] / mmaSize[2], mmaSize[0], mmaSize[2]},
        resultTy.getElementType());

    mlir::Value newOp = rewriter.create<imex::xetile::TileMMAOp>(
        op.getLoc(), vecTy, a, b, c, nullptr, nullptr, nullptr);
    newOp = addUnpackOp(newOp, rewriter);
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

// It updates update_tile_offset to reveal effects of innerblock attribute
// by updating the type of it result.
struct UpdateTileOffsetOpPattern
    : public XeTileConversion<xetile::UpdateTileOffsetOp, TileUsageAnalysis> {

  using XeTileConversion<xetile::UpdateTileOffsetOp,
                         TileUsageAnalysis>::XeTileConversion;

  UpdateTileOffsetOpPattern(mlir::MLIRContext *context,
                            imex::XeTypeConverter &converter,
                            TileUsageAnalysis &analysis,
                            std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter, analysis) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::UpdateTileOffsetOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    if (adaptor.getTile().getType() == op.getResult().getType())
      return mlir::failure();

    rewriter.replaceOpWithNewOp<xetile::UpdateTileOffsetOp>(
        op, adaptor.getTile().getType(), adaptor.getTile(),
        adaptor.getOffsetX(), adaptor.getOffsetY());
    return mlir::success();
  }
};

void populateXeTileBlockingPatterns(
    imex::XeTypeConverter &converter, mlir::RewritePatternSet &patterns,
    TileUsageAnalysis &analysis, std::shared_ptr<XeuArchInterface> ptruArch) {
  patterns.insert<ArithConstantOpPattern, VectorCreateMaskOpPattern,
                  VectorizableOpPattern, SCFForOpPattern, SCFYieldOpPattern,
                  InitTileOpPattern, LoadTileOpPattern, StoreTileOpPattern,
                  TileMMAOpPattern, UpdateTileOffsetOpPattern,
                  VectorMultiDimReductionOpPattern, TileReductionOpPattern,
                  TileBroadcastOpPattern>(patterns.getContext(), converter,
                                          analysis, ptruArch);
  patterns.insert<TransposeOpPattern<mlir::vector::TransposeOp>,
                  TransposeOpPattern<xetile::TransposeOp>>(
      patterns.getContext(), converter, analysis, ptruArch);
}

// Lowers XeTile to blocked layout with high-dim vector
class XeTileBlockingPass : public impl::XeTileBlockingBase<XeTileBlockingPass> {

public:
  XeTileBlockingPass() = default;

  XeTileBlockingPass(const std::string &deviceName) {
    if (this->device.getNumOccurrences() == 0) {
      this->device = deviceName;

      if (deviceName == "pvc") {
        uArchInterface = std::make_shared<XePVCuArch>();
      }
    }
  }

  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
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

    XeTypeConverter typeConverter(context);
    mlir::RewritePatternSet patterns(&context);

    // Use TopDown traversal order, and only look at existing ops
    // to simpliy the code logic and speedup the pass
    mlir::GreedyRewriteConfig config;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
    config.useTopDownTraversal = true;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    { // initialize the inner block size per op.
      patterns.clear();
      auto &analysis = getAnalysis<TileUsageAnalysis>();
      populateXeTileBlockingPatterns(typeConverter, patterns, analysis,
                                     uArchInterface);
      if (failed(
              applyPatternsAndFoldGreedily(mod, std::move(patterns), config))) {
        return signalPassFailure();
      }
    }
    { // aligning the inner block size among tile_mma and load_tile.
      patterns.clear();
      auto &analysis = getAnalysis<PropagateAnalysis>();
      populateXeTileBlockAligningPatterns(typeConverter, patterns, analysis);
      if (failed(
              applyPatternsAndFoldGreedily(mod, std::move(patterns), config))) {
        return signalPassFailure();
      }
    }
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
