//===-------------- Blocking.cpp --------- Blocking Pass  -------*- C++ -*-===//
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
///
//===----------------------------------------------------------------------===//

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/Support/Debug.h>

#include <optional>

#include "imex/Dialect/XeTile/Transforms/Blocking.h"
#include "imex/Dialect/XeTile/Transforms/Passes.h"
#include "imex/Utils/DebugUtils.h"
#include "imex/Utils/XeArch.h"

#include "PassDetail.h"

using namespace mlir;
using namespace imex;
namespace imex {
#define GEN_PASS_DECL_XETILEBLOCKING
#define GEN_PASS_DEF_XETILEBLOCKING
#include "imex/Dialect/XeTile/Transforms/Passes.h.inc"
} // namespace imex

namespace imex {

enum OpType { Prefetch, Load, Store, Elementwise };

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
    llvm::dbgs() << "Invalid Block Size \n";
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

  imex::LoadStore2DConfig configParams;
  int maxHeight, maxWidth, minHeight, minWidth;

  // TODO : Separate it in future.
  if (op == OpType::Load || op == OpType::Prefetch) {

    mlir::FailureOr<LoadStore2DConfig> params = uArchInterface->get2DLoadConfig(
        operation, elementSize, vnni, transpose);
    if (mlir::succeeded(params)) {
      configParams = *params;
      maxHeight = configParams.blockHeight.max;
      minHeight = configParams.blockHeight.min;
      maxWidth = configParams.blockWidth.max;
      minWidth = configParams.blockWidth.min;
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
      configParams = *params;
      maxHeight = configParams.blockHeight.max;
      minHeight = configParams.blockHeight.min;
      maxWidth = configParams.blockWidth.max;
      minWidth = configParams.blockWidth.min;
    } else {
      llvm::dbgs() << "Invalid Config Params \n";
      return {};
    }

    return imex::getInnerBlockHeightWidth(maxHeight, maxWidth, minHeight,
                                          minWidth, height, width);
  }

  if (op == OpType::Elementwise) {
    // TODO: get from uArch?
    int64_t subgroupSize = 16;

    return {1, subgroupSize};
  }
  llvm_unreachable("Unsupported.");
  return {};
}

// it blocks a constant dense value if it is used by XeTile operators,
// e.g, tile_mma and store_tile. It currently extends a 2D vector into
// 4D vector with the last 2 dim corresponding to block size.
// example: arith.constant dense<0.0>: vector<32x32xf16>
//      --> arith.constant dense<0.0>: vector<4x2x8x16xf16>
// [8, 16] is the block size.
struct ArithConstantOpPattern
    : public XeTileConversion<mlir::arith::ConstantOp> {

  using XeTileConversion<mlir::arith::ConstantOp>::XeTileConversion;

  ArithConstantOpPattern(mlir::MLIRContext *context,
                         imex::XeTypeConverter &converter,
                         std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto value = llvm::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
    if (!value || value.getType().getRank() != 2)
      return mlir::failure();

    auto shape = value.getType().getShape();
    auto blkSZ =
        getInnerBlockSizes<Load>(op.getOperation(), value.getElementType(),
                                 shape[0], shape[1], this->uArchInterface);
    if (blkSZ.empty()) {
      op->emitOpError() << "Invalid inner block sizes ";
      return mlir::failure();
    }

    auto newTy = mlir::VectorType::get(
        {shape[0] / blkSZ[0], shape[1] / blkSZ[1], blkSZ[0], blkSZ[1]},
        value.getElementType());

    value = value.reshape(newTy);
    auto newOp = rewriter.create<mlir::arith::ConstantOp>(loc, value);
    auto unpack = addUnpackOp(newOp, rewriter);

    // TODO: we may can replace it with standard replaceOp method when
    // we have full support for other non-xetile operators.
    rewriter.replaceOpWithIf(
        op, unpack->getResults(), [&](mlir::OpOperand &op) {
          auto *owner = op.getOwner();

          // the direct user is an xetile operator
          if (llvm::isa<xetile::XeTileDialect>(owner->getDialect()))
            return true;

          // the direct user is an scf::ForOp, but the corresponding argument
          // is used by an xetile operator
          if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(owner)) {
            auto arg = forOp.getTiedLoopRegionIterArg(&op);

            auto haveXeTileUsers = std::any_of(
                arg.user_begin(), arg.user_end(), [&](mlir::Operation *op) {
                  return llvm::isa<xetile::XeTileDialect>(op->getDialect());
                });

            if (auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(
                    forOp.getRegion().front().getTerminator())) {
              auto idx = forOp.getTiedLoopResult(&op).getResultNumber();
              auto definingOp = yieldOp.getResults()[idx].getDefiningOp();
              if (definingOp)
                haveXeTileUsers |=
                    llvm::isa<xetile::XeTileDialect>(definingOp->getDialect());
            }
            return haveXeTileUsers;
          }
          return false;
        });
    return mlir::success();
  }
};

// Pattern for generic elemetwise ops. Blocks op according to
// getInnerBlocks<Elementwise>. Pack/Unpack ops are inserted on the ops
// boundaries if needed.
struct VectorizableOpPattern
    : public XeTileTraitConversion<mlir::OpTrait::Vectorizable> {

  using XeTileTraitConversion::XeTileTraitConversion;

  VectorizableOpPattern(mlir::MLIRContext *context,
                        imex::XeTypeConverter &converter,
                        std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileTraitConversion(context, converter) {
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

// It rewrites the SCF forOp, it mainly updates the arguments of its
// region block. unpack ops are added for VectorType operands if needed.
struct SCFForOpPattern : public XeTileConversion<mlir::scf::ForOp> {

  using XeTileConversion<mlir::scf::ForOp>::XeTileConversion;

  SCFForOpPattern(mlir::MLIRContext *context, imex::XeTypeConverter &converter,
                  std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  ::mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    // preprocess the init args and add pack op if it is
    // defined by an unpack op.
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

    mlir::Block *block = op.getBody();
    mlir::Block *newBlock = newOp.getBody();
    rewriter.mergeBlocks(block, newBlock, newBlock->getArguments());

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
struct SCFYieldOpPattern : public XeTileConversion<mlir::scf::YieldOp> {

  using XeTileConversion<mlir::scf::YieldOp>::XeTileConversion;

  SCFYieldOpPattern(mlir::MLIRContext *context,
                    imex::XeTypeConverter &converter,
                    std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  ::mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto forOp = llvm::dyn_cast_if_present<mlir::scf::ForOp>(op->getParentOp());
    if (!forOp)
      return mlir::failure();

    llvm::SmallVector<mlir::Value> results;
    for (auto [i, value] : llvm::enumerate(adaptor.getResults())) {
      auto valTy = value.getType().dyn_cast<mlir::VectorType>();
      auto tgtTy =
          forOp.getRegionIterArg(i).getType().dyn_cast<mlir::VectorType>();
      if (valTy && tgtTy && valTy.getRank() == 2 && tgtTy.getRank() == 4) {
        auto innerBlock = tgtTy.getShape().take_back(2);
        auto pack = addPackOp(value, innerBlock, rewriter);
        results.push_back(pack);
      } else {
        results.push_back(value);
      }
    }

    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, results);

    return mlir::success();
  }
};

// It updates init_tile by attaching innerBlock attribute to the result
// tile. The block size is choosed based on how the tile is used, including
// prefetch, load, store. Since hardware support different sizes for them.
struct InitTileOpPattern : public XeTileConversion<xetile::InitTileOp> {

  using XeTileConversion<xetile::InitTileOp>::XeTileConversion;

  InitTileOpPattern(mlir::MLIRContext *context,
                    imex::XeTypeConverter &converter,
                    std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter) {
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

    // skip it if innerBlocks has been set by user or compiler.
    if (innerBlocks)
      return mlir::failure();

    auto elemTy = tileTy.getElementType();
    int elementSize = elemTy.getIntOrFloatBitWidth();
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

      for (auto user : op->getUsers()) {
        if (llvm::dyn_cast<xetile::LoadTileOp>(user)) {
          auto loadTileOp = llvm::dyn_cast<xetile::LoadTileOp>(user);
          if (isForDPASB(loadTileOp) && elementSize < 32) {
            vnni = true;
          }
          break;
        }
      }
      if (vnni && transpose && elementSize < 32) {

        int factor = 32 / elementSize;
        vnni = false;
        innerBlocks = mlir::DenseI64ArrayAttr::get(
            getContext(),
            getInnerBlockSizes<Load>(
                op.getOperation(), mlir::FloatType::getF32(getContext()),
                tileTy.getShape()[0], (tileTy.getShape()[1]) * factor,
                this->uArchInterface, vnni, transpose));
      } else if (transpose && elementSize < 32) {
        return rewriter.notifyMatchFailure(op, "Invalid transpose.");
      } else {
        innerBlocks = mlir::DenseI64ArrayAttr::get(
            getContext(),
            getInnerBlockSizes<Load>(op.getOperation(), elemTy,
                                     tileTy.getShape()[0], tileTy.getShape()[1],
                                     this->uArchInterface, vnni, transpose));
      }
    } else if (isForStore(op)) {
      innerBlocks = mlir::DenseI64ArrayAttr::get(
          getContext(), getInnerBlockSizes<Store>(
                            op.getOperation(), elemTy, tileTy.getShape()[0],
                            tileTy.getShape()[1], this->uArchInterface));
    } else {
      return rewriter.notifyMatchFailure(
          op, "The tile is used for multiple purpose. The init-duplicate pass "
              "should be run first to resolve this issue.");
    }

    if (innerBlocks.empty()) {
      op->emitOpError() << "Invalid inner block sizes ";
      return mlir::failure();
    }

    auto attr = imex::xetile::XeTileAttr::get(
        op.getContext(), tileTy.getSgMap(), tileTy.getWgMap(),
        tileTy.getOrder(), innerBlocks, tileTy.getWgData());

    auto newTileTy =
        imex::xetile::TileType::get(tileTy.getShape(), elemTy, attr);

    rewriter.replaceOpWithNewOp<xetile::InitTileOp>(
        op, newTileTy, op.getSource(), op.getOffsets(),
        op.getStaticOffsetsAttr(), op.getDynamicShape(),
        op.getDynamicStrides());

    return mlir::success();
  }
};

// It updates load_tile to reveal effects of innerblock attribute by
// representing value as 4D vector. An unpack op is added at the end
// to make this change to be transparent to its users.
struct LoadTileOpPattern : public XeTileConversion<xetile::LoadTileOp> {

  using XeTileConversion<xetile::LoadTileOp>::XeTileConversion;

  LoadTileOpPattern(mlir::MLIRContext *context,
                    imex::XeTypeConverter &converter,
                    std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::LoadTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tileTy = op.getSource().getType();
    auto shape = tileTy.getShape();
    auto innerBlocks = tileTy.getInnerBlocks();

    if (!innerBlocks)
      return rewriter.notifyMatchFailure(
          op, "The InnerBlock attr is required by missing from the tile.\n");

    auto vecTy = ::mlir::VectorType::get({shape[0] / innerBlocks[0],
                                          shape[1] / innerBlocks[1],
                                          innerBlocks[0], innerBlocks[1]},
                                         tileTy.getElementType());

    auto newOp = rewriter.create<imex::xetile::LoadTileOp>(
        loc, vecTy, adaptor.getSource(), op.getPaddingAttr());

    auto unpack = addUnpackOp(newOp, rewriter);

    rewriter.replaceOp(op, unpack);

    return mlir::success();
  }
};

// It updates store_tile to reveal effects of innerblock attribute.
// It uses pack op to align the shape of its vector value to the tile shape.
struct StoreTileOpPattern : public XeTileConversion<xetile::StoreTileOp> {

  using XeTileConversion<xetile::StoreTileOp>::XeTileConversion;

  StoreTileOpPattern(mlir::MLIRContext *context,
                     imex::XeTypeConverter &converter,
                     std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::StoreTileOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto tileTy = llvm::dyn_cast<xetile::TileType>(adaptor.getTile().getType());
    auto innerBlocks = tileTy.getInnerBlocks().asArrayRef();
    auto value = addPackOp(adaptor.getValue(), innerBlocks, rewriter);
    rewriter.replaceOpWithNewOp<xetile::StoreTileOp>(op, value,
                                                     adaptor.getTile());
    return mlir::success();
  }
};

llvm::SmallVector<unsigned int>
getMMASize(mlir::Type elemTy, const int APrecision, const int BPrecision,
           const int CPrecision, const int DPrecision,
           std::shared_ptr<XeuArchInterface> uArchInterface) {
  assert(elemTy.isIntOrFloat());
  auto bits = elemTy.getIntOrFloatBitWidth();
  imex::DPASConfig dpasParams;
  llvm::SmallVector<unsigned int> result;
  switch (bits) {
  case 16:
    dpasParams = uArchInterface->getDPASConfig(APrecision, BPrecision,
                                               CPrecision, DPrecision);
    result = llvm::SmallVector<unsigned int>(
        {dpasParams.m, dpasParams.k, dpasParams.n});
    break;
  default:
    result = llvm::SmallVector<unsigned int>({8, 8, 8});
    break;
  }
  return result;
}

// It updates tile_mma to reveal effects of innerblock attribute.
// Values will be reprented as 4D vectors. An unpack op is applied
// to its result to make the change transparent to its users.
struct TileMMAOpPattern : public XeTileConversion<xetile::TileMMAOp> {

  using XeTileConversion<xetile::TileMMAOp>::XeTileConversion;

  TileMMAOpPattern(mlir::MLIRContext *context, imex::XeTypeConverter &converter,
                   std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::TileMMAOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto a = adaptor.getA();
    auto b = adaptor.getB();
    auto c = adaptor.getC();

    unsigned int CPrecision = 0;
    if (c) {
      auto ty = c.getType().dyn_cast<mlir::VectorType>();
      auto accTy = c ? ty.getElementType() : nullptr;
      if (accTy)
        CPrecision = op.getBType().getElementType().getIntOrFloatBitWidth();
    }
    assert(a && b && "a operand or b operand is (are) missing.\n");

    auto mmaSize = getMMASize(
        op.getElementType(),
        op.getAType().getElementType().getIntOrFloatBitWidth(),
        op.getBType().getElementType().getIntOrFloatBitWidth(), CPrecision,
        op.getResult().getType().getElementType().getIntOrFloatBitWidth(),
        this->uArchInterface);

    a = addPackOp(a, {mmaSize[0], mmaSize[1]}, rewriter);
    b = addPackOp(b, {mmaSize[1], mmaSize[2]}, rewriter);

    if (c) {
      auto ty = c.getType().dyn_cast<mlir::VectorType>();
      if (ty.getRank() == 2)
        c = addPackOp(c, {mmaSize[0], mmaSize[2]}, rewriter);
      if (ty.getRank() == 4)
        c = addUnpackAndPackOps(c, {mmaSize[0], mmaSize[2]}, rewriter);
    }

    auto resultTy = op.getResult().getType();
    auto shape = resultTy.getShape();
    auto vecTy = ::mlir::VectorType::get(
        {shape[0] / mmaSize[0], shape[1] / mmaSize[2], mmaSize[0], mmaSize[2]},
        resultTy.getElementType());

    mlir::Value newOp =
        rewriter.create<imex::xetile::TileMMAOp>(loc, vecTy, a, b, c);
    newOp = addUnpackOp(newOp, rewriter);
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

// It updates update_tile_offset to reveal effects of innerblock attribute
// by updating the type of it result.
struct UpdateTileOffsetOpPattern
    : public XeTileConversion<xetile::UpdateTileOffsetOp> {

  using XeTileConversion<xetile::UpdateTileOffsetOp>::XeTileConversion;

  UpdateTileOffsetOpPattern(mlir::MLIRContext *context,
                            imex::XeTypeConverter &converter,
                            std::shared_ptr<XeuArchInterface> ptruArch)
      : XeTileConversion(context, converter) {
    this->uArchInterface = ptruArch;
  }

  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;

  ::mlir::LogicalResult
  matchAndRewrite(xetile::UpdateTileOffsetOp op, OpAdaptor adaptor,
                  OpPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xetile::UpdateTileOffsetOp>(
        op, adaptor.getTile().getType(), adaptor.getTile(),
        adaptor.getOffsetX(), adaptor.getOffsetY());
    return mlir::success();
  }
};

void populateXeTileBlockingPatterns(
    imex::XeTypeConverter &converter, mlir::RewritePatternSet &patterns,
    std::shared_ptr<XeuArchInterface> ptruArch) {
  patterns
      .insert<ArithConstantOpPattern, VectorizableOpPattern, SCFForOpPattern,
              SCFYieldOpPattern, InitTileOpPattern, LoadTileOpPattern,
              StoreTileOpPattern, TileMMAOpPattern, UpdateTileOffsetOpPattern>(
          patterns.getContext(), converter, ptruArch);
}

// Lowers XeTile to blocked layout with high-dim vector
class XeTileBlockingPass
    : public imex::impl::XeTileBlockingBase<imex::XeTileBlockingPass> {

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

    auto &usageAnalysis = getAnalysis<TileUsageAnalysis>();

    mlir::RewritePatternSet patterns(&context);
    XeTypeConverter typeConverter(context, &usageAnalysis);

    populateXeTileBlockingPatterns(typeConverter, patterns, uArchInterface);

    // Use TopDown traversal order, and only look at existing ops
    // to simpliy the code logic and speedup the pass
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = 2;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    if (failed(
            applyPatternsAndFoldGreedily(mod, std::move(patterns), config))) {
      return signalPassFailure();
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
