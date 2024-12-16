//===- XeTileToXeGPU.cpp - XeTileToXeGPU conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the XeTileToXeGPU conversion, converting the XeTile
/// dialect to the XeGPU dialect.
///
//===----------------------------------------------------------------------===//

#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/Passes.h>

#include "imex/Conversion/XeTileToXeGPU/XeTileToXeGPU.h"
#include "imex/Utils/XeArch.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace imex {
#define GEN_PASS_DEF_CONVERTXETILETOXEGPU
#include "imex/Conversion/Passes.h.inc"
} // namespace imex

#include <memory>
namespace imex {

// Converts an Attribute representing memory space to xegpu::MemorySpaceAttr.
// It currently only supports memory space represented as integer attribute.
// TODO: improve it to support other types of memory space attributes, e.g.,
// gpu::MemorySpaceAttr, and spirv::MemorySpaceAttr, etc.
static mlir::xegpu::MemorySpaceAttr convertMemorySpace(mlir::Attribute attr) {
  auto space = mlir::xegpu::MemorySpace::Global; // default to global value
  if (auto IntAttr = mlir::dyn_cast_if_present<mlir::IntegerAttr>(attr)) {
    space = IntAttr.getInt() == 3 ? mlir::xegpu::MemorySpace::SLM
                                  : mlir::xegpu::MemorySpace::Global;
  }
  return attr ? mlir::xegpu::MemorySpaceAttr::get(attr.getContext(), space)
              : mlir::xegpu::MemorySpaceAttr();
}

static mlir::xegpu::CachePolicy
translateCachePolicy(imex::xetile::CachePolicyAttr val,
                     mlir::xegpu::CachePolicy defaultVal) {
  if (!val)
    return defaultVal;

  switch (val.getValue()) {
  case imex::xetile::CachePolicy::CACHED:
    return mlir::xegpu::CachePolicy::CACHED;
  case imex::xetile::CachePolicy::UNCACHED:
    return mlir::xegpu::CachePolicy::UNCACHED;
  case imex::xetile::CachePolicy::STREAMING:
    return mlir::xegpu::CachePolicy::STREAMING;
  case imex::xetile::CachePolicy::READ_INVALIDATE:
    return mlir::xegpu::CachePolicy::READ_INVALIDATE;
  case imex::xetile::CachePolicy::WRITE_BACK:
    return mlir::xegpu::CachePolicy::WRITE_BACK;
  case imex::xetile::CachePolicy::WRITE_THROUGH:
    return mlir::xegpu::CachePolicy::WRITE_THROUGH;
  }
  llvm_unreachable("Invalid CachePolicy value");
}

template <typename OpTy>
static auto getCachePolicy(OpTy op, mlir::xegpu::CachePolicy defaultVal =
                                        mlir::xegpu::CachePolicy::CACHED) {

  auto getCachePolicyAttr = [&](imex::xetile::CachePolicyAttr val) {
    return mlir::xegpu::CachePolicyAttr::get(
        op.getContext(), translateCachePolicy(val, defaultVal));
  };

  auto L1 = getCachePolicyAttr(op.getL1HintAttr());
  auto L2 = getCachePolicyAttr(op.getL2HintAttr());
  auto L3 = getCachePolicyAttr(op.getL3HintAttr());

  return std::make_tuple(L1, L2, L3);
}

// convert init_tile to xegpu::CreateNdDescOp if the tile is for
// blocked load/store on global memory, otherwise, convert it to
// xegpu::CreateDescOp.
class InitOpPattern final
    : public mlir::OpConversionPattern<xetile::InitTileOp> {
public:
  using mlir::OpConversionPattern<xetile::InitTileOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(xetile::InitTileOp op, OpAdaptor adator,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tileTy = op.getType();
    auto source = op.getSource();

    auto converter = getTypeConverter();
    auto tdescTy = converter->convertType<mlir::xegpu::TensorDescType>(tileTy);
    auto memSpaceAttr = convertMemorySpace(tileTy.getMemorySpace());
    auto memSpace = memSpaceAttr ? memSpaceAttr.getValue()
                                 : mlir::xegpu::MemorySpace::Global;

    // here it cannot use tdescTy.isScattered(), because block tile on SLM could
    // be lowered into scattered TensorDesc too.
    bool isScattered =
        tileTy.getScatterAttr() ? tileTy.getScatterAttr().getValue() : false;

    mlir::Value newOp;
    if (isScattered) {
      auto idxTy = mlir::VectorType::get(tdescTy.getNumElements(),
                                         rewriter.getIndexType());
      auto indices = rewriter.create<mlir::vector::ShapeCastOp>(
          loc, idxTy, op.getIndices());
      newOp = rewriter.create<mlir::xegpu::CreateDescOp>(loc, tdescTy, source,
                                                         indices);
    } else if (memSpace == mlir::xegpu::MemorySpace::Global) {
      newOp = rewriter.create<mlir::xegpu::CreateNdDescOp>(
          loc, tdescTy, source, op.getOffsets(), op.getSizes(), op.getStrides(),
          op.getConstOffsetsAttr(), op.getConstSizesAttr(),
          op.getConstStridesAttr());
    } else {
      // TODO: Lowering strategy for blocked tiles on SLM is not finalized yet.
      return mlir::failure();
    }
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

// convert update_tile_offset to xegpu::UpdateNdOffsetOp if the tile
// is for blocked load/store on global memory, otherwise, convert it to
// xegpu::UpdateOffsetOp.
class UpdateOpPattern
    : public mlir::OpConversionPattern<xetile::UpdateTileOffsetOp> {
public:
  using mlir::OpConversionPattern<
      xetile::UpdateTileOffsetOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(xetile::UpdateTileOffsetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto tileTy = op.getTile().getType();
    auto scatterAttr = tileTy.getScatterAttr();
    auto isScattered = scatterAttr ? scatterAttr.getValue() : false;
    auto tdesc = adaptor.getTile();
    if (isScattered) {
      auto indicesTy = op.getIndices().getType();
      auto flatTy = mlir::VectorType::get(indicesTy.getNumElements(),
                                          indicesTy.getElementType());
      auto indices = rewriter.create<mlir::vector::ShapeCastOp>(
          op.getLoc(), flatTy, adaptor.getIndices());
      rewriter.replaceOpWithNewOp<mlir::xegpu::UpdateOffsetOp>(
          op, tdesc.getType(), tdesc, indices);
    } else {
      auto x = op.getOffsetX();
      auto y = op.getOffsetY();
      int64_t kDynamics[2] = {mlir::ShapedType::kDynamic,
                              mlir::ShapedType::kDynamic};
      rewriter.replaceOpWithNewOp<mlir::xegpu::UpdateNdOffsetOp>(
          op, tdesc.getType(), tdesc, mlir::ValueRange({x, y}),
          llvm::ArrayRef<int64_t>(kDynamics, 2));
    }

    return mlir::success();
  }
};

// convert prefetch_tile to xegpu::PrefetchNdOp if the tile is for
// blocked load/store on global memory, or xegpu::PrefetchOp if the
// tile is for scattered op on global memory. Otherwise, drop it.
class PrefetchOpPattern
    : public mlir::OpConversionPattern<xetile::PrefetchTileOp> {
public:
  using mlir::OpConversionPattern<xetile::PrefetchTileOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(xetile::PrefetchTileOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto tileTy = op.getTile().getType();
    auto memSpaceAttr = convertMemorySpace(tileTy.getMemorySpace());
    auto memSpace = memSpaceAttr ? memSpaceAttr.getValue()
                                 : mlir::xegpu::MemorySpace::Global;
    auto scatterAttr = tileTy.getScatterAttr();
    auto isScattered = scatterAttr ? scatterAttr.getValue() : false;

    auto [L1, L2, L3] = getCachePolicy(op);

    // Accesses to SLM doesn't need to be prefetched.
    if (memSpace == mlir::xegpu::MemorySpace::SLM)
      rewriter.eraseOp(op);

    auto tile = adaptor.getTile();
    if (isScattered) {
      rewriter.replaceOpWithNewOp<mlir::xegpu::PrefetchOp>(op, tile, L1, L2,
                                                           L3);
    } else {
      rewriter.replaceOpWithNewOp<mlir::xegpu::PrefetchNdOp>(op, tile, L1, L2,
                                                             L3);
    }

    return mlir::success();
  }
};

// convert load_tile to xegpu::LoadNdOp if the tile is for blocked
// load, or xegpu::LoadGatherOp if the tile is for scattered load
// (for SLM access).
class LoadOpPattern : public mlir::OpConversionPattern<xetile::LoadTileOp> {
public:
  using mlir::OpConversionPattern<xetile::LoadTileOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(xetile::LoadTileOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto tileTy = op.getSource().getType();
    auto memSpaceAttr = convertMemorySpace(tileTy.getMemorySpace());
    auto memSpace = memSpaceAttr ? memSpaceAttr.getValue()
                                 : mlir::xegpu::MemorySpace::Global;

    // TODO: add Lowering for blocked tiles on SLM.
    if (memSpace == mlir::xegpu::MemorySpace::Global) {
      auto [L1, L2, L3] = getCachePolicy(op);
      auto packAttr = mlir::UnitAttr();
      auto transAttr = mlir::DenseI64ArrayAttr();
      auto bitWidthAttr = mlir::IntegerAttr();
      rewriter.replaceOpWithNewOp<mlir::xegpu::LoadNdOp>(
          op, op.getType(), adaptor.getSource(), packAttr, transAttr,
          bitWidthAttr, L1, L2, L3);
      return mlir::success();
    }
    return mlir::failure();
  }
};

// convert xetile.load to xegpu::LoadGatherOp.
class GatherOpPattern : public mlir::OpConversionPattern<xetile::LoadGatherOp> {
public:
  using mlir::OpConversionPattern<xetile::LoadGatherOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(xetile::LoadGatherOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = op.getValue().getType();
    auto elemTy = type.getElementType();
    auto ldTy = mlir::VectorType::get(type.getNumElements(), elemTy);
    auto maskTy = mlir::VectorType::get(type.getNumElements(),
                                        rewriter.getIntegerType(1));
    auto transposeAttr = mlir::UnitAttr();
    auto [L1, L2, L3] = getCachePolicy(op);
    auto mask = rewriter.create<mlir::vector::ShapeCastOp>(loc, maskTy,
                                                           adaptor.getMask());
    auto ldOp = rewriter.create<mlir::xegpu::LoadGatherOp>(
        loc, ldTy, adaptor.getTile(), mask, transposeAttr, L1, L2, L3);
    auto v =
        rewriter.create<mlir::vector::ShapeCastOp>(loc, op.getType(), ldOp);
    rewriter.replaceOp(op, v);
    return mlir::success();
  }
};

// convert store_tile to xegpu::StoreNdOp if the tile is for blocked store.
// Otherwise, convert it to xegpu::StoreScatterOp (for SLM access).
class StoreOpPattern : public mlir::OpConversionPattern<xetile::StoreTileOp> {
public:
  using mlir::OpConversionPattern<xetile::StoreTileOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(xetile::StoreTileOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto tileTy = op.getTile().getType();
    auto memSpaceAttr = convertMemorySpace(tileTy.getMemorySpace());
    auto memSpace = memSpaceAttr ? memSpaceAttr.getValue()
                                 : mlir::xegpu::MemorySpace::Global;

    // TODO: add Lowering for blocked tiles on SLM.
    if (memSpace == mlir::xegpu::MemorySpace::Global) {
      auto [L1, L2, L3] =
          getCachePolicy(op, mlir::xegpu::CachePolicy::WRITE_BACK);
      rewriter.replaceOpWithNewOp<mlir::xegpu::StoreNdOp>(
          op, adaptor.getValue(), adaptor.getTile(), L1, L2, L3);
      return mlir::success();
    }
    return mlir::failure();
  }
};

// convert xetile.store to xegpu::StoreScatterOp.
class ScatterOpPattern
    : public mlir::OpConversionPattern<xetile::StoreScatterOp> {
public:
  using mlir::OpConversionPattern<xetile::StoreScatterOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(xetile::StoreScatterOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto value = adaptor.getValue();
    auto tdesc = adaptor.getTile();
    auto mask = adaptor.getMask();

    auto tileTy = op.getTile().getType();
    auto valTy =
        mlir::VectorType::get(tileTy.getNumElements(), tileTy.getElementType());
    auto maskTy = mlir::VectorType::get(tileTy.getNumElements(),
                                        rewriter.getIntegerType(1));
    auto transposeAttr = mlir::UnitAttr();
    auto [L1, L2, L3] =
        getCachePolicy(op, mlir::xegpu::CachePolicy::WRITE_BACK);
    mask =
        rewriter.create<mlir::vector::ShapeCastOp>(op.getLoc(), maskTy, mask);
    value =
        rewriter.create<mlir::vector::ShapeCastOp>(op.getLoc(), valTy, value);
    rewriter.replaceOpWithNewOp<mlir::xegpu::StoreScatterOp>(
        op, value, tdesc, mask, transposeAttr, L1, L2, L3);
    return mlir::success();
  }
};

// convert xetile.mma to xegpu::DpasOp.
class MMAOpPattern : public mlir::OpConversionPattern<xetile::TileMMAOp> {
public:
  using mlir::OpConversionPattern<xetile::TileMMAOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(xetile::TileMMAOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::xegpu::DpasOp>(
        op, op.getType(), adaptor.getA(), adaptor.getB(), adaptor.getC());
    return mlir::success();
  }
};

// convert xetile.broadcast. It will be coverted to vector::BroadcastOp
// if the broadcast dim is 0; otherwise, it will be converted to
// vector::InsertStridedSliceOp.
class BroadcastOpPattern
    : public mlir::OpConversionPattern<xetile::BroadcastOp> {
public:
  using mlir::OpConversionPattern<xetile::BroadcastOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(xetile::BroadcastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = op.getResult().getType();
    auto dim = op.getBroadcastDim();
    if (dim.size() != 1 || resTy.getRank() != 2 || dim[0] > 1)
      return rewriter.notifyMatchFailure(
          op, "Only support reduction with one broadcast dim on 2D vector.");

    if (dim[0] == 0) {
      auto srcTy = op.getSource().getType();
      auto newTy =
          mlir::VectorType::get(srcTy.getNumElements(), srcTy.getElementType());
      auto cast = rewriter.create<mlir::vector::ShapeCastOp>(
          op.getLoc(), newTy, adaptor.getSource());
      rewriter.replaceOpWithNewOp<mlir::vector::BroadcastOp>(op, resTy, cast);
      return mlir::success();
    }

    if (dim[0] == 1) {
      auto srcTy = op.getSource().getType();
      auto elemTy = srcTy.getElementType();
      auto attr = elemTy.isInteger()
                      ? (mlir::Attribute)rewriter.getIntegerAttr(elemTy, 0)
                      : (mlir::Attribute)rewriter.getFloatAttr(elemTy, 0.0);

      mlir::Value result = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), resTy, mlir::DenseElementsAttr::get(resTy, attr));

      for (int64_t j = 0; j < resTy.getShape()[1]; j++) {
        result = rewriter.create<mlir::vector::InsertStridedSliceOp>(
            op.getLoc(), adaptor.getSource(), result,
            llvm::ArrayRef<int64_t>({0, j}), llvm::ArrayRef<int64_t>({1, 1}));
      }
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return mlir::failure();
  }
};

// convert xetile.reduce to vector::MultiDimReductionOp.
class ReduceOpPattern : public mlir::OpConversionPattern<xetile::ReductionOp> {
public:
  using mlir::OpConversionPattern<xetile::ReductionOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(xetile::ReductionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = op.getResult().getType();
    auto newTy =
        mlir::VectorType::get(resTy.getNumElements(), resTy.getElementType());
    auto acc = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), newTy, mlir::DenseElementsAttr::get(newTy, 0));
    auto result =
        rewriter.replaceOpWithNewOp<mlir::vector::MultiDimReductionOp>(
            op, op.getType(), op.getKindAttr(), adaptor.getSource(), acc,
            op.getReductionDimsAttr());
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

// convert xetile.transpose to vector::TransposeOp.
class TransposeOpPattern
    : public mlir::OpConversionPattern<xetile::TransposeOp> {
public:
  using mlir::OpConversionPattern<xetile::TransposeOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(xetile::TransposeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::vector::TransposeOp>(
        op, adaptor.getVector(), adaptor.getPermutation());
    return mlir::success();
  }
};

// update SCF ForOp, majory convert the argument type from TileType to
// TensorDescType
class SCFForOpPattern : public mlir::OpConversionPattern<mlir::scf::ForOp> {
public:
  using mlir::OpConversionPattern<mlir::scf::ForOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<mlir::scf::ForOp>(
        op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(),
        adaptor.getInitArgs());
    mlir::Block *newBlock = newOp.getBody();
    // remove the terminator of the new block
    if (newBlock->mightHaveTerminator())
      rewriter.eraseOp(newBlock->getTerminator());

    mlir::Block *block = op.getBody();
    mlir::TypeConverter::SignatureConversion mapping(block->getNumArguments());
    for (auto [i, ty] : llvm::enumerate(newBlock->getArgumentTypes()))
      mapping.addInputs(i, ty);
    block = rewriter.applySignatureConversion(block, mapping);
    rewriter.mergeBlocks(block, newBlock, newBlock->getArguments());
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  };
};

// update SCF YieldOp, update the operands
class SCFYieldOpPattern : public mlir::OpConversionPattern<mlir::scf::YieldOp> {
public:
  using mlir::OpConversionPattern<mlir::scf::YieldOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, adaptor.getResults());
    return mlir::success();
  }
};

class XeTileConversionTarget : public mlir::ConversionTarget {
public:
  explicit XeTileConversionTarget(mlir::MLIRContext &context,
                                  std::shared_ptr<XeuArchInterface> ptruArch)
      : mlir::ConversionTarget(context) {

    this->uArchInterface = ptruArch;

    addDynamicallyLegalOp<mlir::xegpu::DpasOp>(
        [&](mlir::Operation *op) -> bool {
          return (uArchInterface &&
                  mlir::succeeded(uArchInterface->isLegalDpasOp(op)));
        });

    addDynamicallyLegalOp<mlir::xegpu::LoadNdOp>(
        [&](mlir::Operation *op) -> bool {
          return (uArchInterface &&
                  mlir::succeeded(uArchInterface->isLegalLoad2dOp(op)));
        });

    addDynamicallyLegalOp<mlir::xegpu::StoreNdOp>(
        [&](mlir::Operation *op) -> bool {
          return (uArchInterface &&
                  mlir::succeeded(uArchInterface->isLegalStore2dOp(op)));
        });

    addDynamicallyLegalOp<mlir::xegpu::PrefetchNdOp>(
        [&](mlir::Operation *op) -> bool {
          return (uArchInterface &&
                  mlir::succeeded(uArchInterface->isLegalPrefetch2dOp(op)));
        });

    addIllegalDialect<imex::xetile::XeTileDialect>();
    addLegalDialect<mlir::xegpu::XeGPUDialect>();
    addLegalOp<mlir::vector::ShapeCastOp>();
    markUnknownOpDynamicallyLegal([&](mlir::Operation *op) {
      for (auto ty : op->getResultTypes()) {
        if (mlir::isa<xetile::TileType>(ty))
          return false;
      }
      for (auto ty : op->getOperandTypes()) {
        if (mlir::isa<xetile::TileType>(ty))
          return false;
      }
      return true;
    });
  }

private:
  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;
};

// Full Pass
struct ConvertXeTileToXeGPUPass // convert XeTile to XeGPU
    : public imex::impl::ConvertXeTileToXeGPUBase<ConvertXeTileToXeGPUPass> {
  ConvertXeTileToXeGPUPass() = default;

  ConvertXeTileToXeGPUPass(const std::string &deviceName) {
    if (this->device.getNumOccurrences() == 0) {
      this->device = deviceName;

      if (deviceName == "pvc") {
        uArchInterface = std::make_shared<XePVCuArch>();
      }
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
    auto mod = getOperation();
    mlir::MLIRContext &context = getContext();

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

    XeTileConversionTarget target(context, uArchInterface);
    mlir::RewritePatternSet patterns(&context);

    mlir::TypeConverter typeConverter;

    typeConverter.addConversion(
        [&](mlir::Type type) -> mlir::Type { return type; });

    typeConverter.addConversion(
        [&](xetile::TileType type) -> mlir::xegpu::TensorDescType {
          auto context = type.getContext();
          auto elemTy = type.getElementType();
          auto scatterAttr = type.getScatterAttr();
          bool isScattered = scatterAttr ? scatterAttr.getValue() : false;

          mlir::xegpu::SGMapAttr sgMap = nullptr;
          if (auto attr = type.getSgMap()) {
            auto layout =
                llvm::to_vector_of<uint32_t>(attr.getWiLayout().asArrayRef());
            auto data =
                llvm::to_vector_of<uint32_t>(attr.getWiData().asArrayRef());
            sgMap = mlir::xegpu::SGMapAttr::get(context, layout, data);
          }

          auto memSpaceAttr = convertMemorySpace(type.getMemorySpace());
          auto memSpace = memSpaceAttr ? memSpaceAttr.getValue()
                                       : mlir::xegpu::MemorySpace::Global;

          mlir::Attribute encoding;
          llvm::SmallVector<int64_t> shape;
          if (isScattered) {
            // Scattered tile is lowered to scattered tensor_desc with chunk
            // size 1. It supports both global memory and shared memory. while
            // scattered tile can support 2D shape, scattered tensor_desc only
            // support 1D shape.
            auto chunkSizeAttr =
                mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), 1);
            encoding = mlir::xegpu::ScatterTensorDescAttr::get(
                context, memSpaceAttr, chunkSizeAttr);
            shape.push_back(type.getNumElements());
          } else if (memSpace == mlir::xegpu::MemorySpace::Global) {
            // Blocked tile on global memory is lowered to blocked tensor_desc
            // with the same shape.
            // TODO: update TileType with array_length and use it here.
            auto arrayLenAttr =
                mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), 1);
            auto boundaryCheckAttr = mlir::BoolAttr::get(context, true);
            encoding = mlir::xegpu::BlockTensorDescAttr::get(
                context, memSpaceAttr, arrayLenAttr, boundaryCheckAttr);
            shape = llvm::to_vector(type.getShape());
          } else {
            // TODO: Lowering strategy for blocked tiles on SLM is not
            // finalized yet.
            assert(0 && "SLM space for blocked tile is not supported yet.");
          }
          return mlir::xegpu::TensorDescType::get(context, shape, elemTy,
                                                  encoding, sgMap);
        });

    auto materializeWithCast = [&](mlir::OpBuilder &builder, mlir::Type type,
                                   mlir::ValueRange inputs,
                                   mlir::Location loc) -> mlir::Value {
      assert(inputs.size() == 1 && "Expecting single input");
      return builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };

    typeConverter.addArgumentMaterialization(materializeWithCast);
    typeConverter.addTargetMaterialization(materializeWithCast);
    typeConverter.addSourceMaterialization(materializeWithCast);

    populateXeTileToXeGPUConversionPatterns(typeConverter, patterns);

    if (mlir::failed(
            mlir::applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }

private:
  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;
};

/// Populate the given list with patterns that convert XeTile to XeGPU
void populateXeTileToXeGPUConversionPatterns(
    mlir::TypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<InitOpPattern, UpdateOpPattern, PrefetchOpPattern, LoadOpPattern,
               StoreOpPattern, GatherOpPattern, ScatterOpPattern, MMAOpPattern,
               BroadcastOpPattern, ReduceOpPattern, TransposeOpPattern,
               SCFForOpPattern, SCFYieldOpPattern>(converter,
                                                   patterns.getContext());
}

/// Create a pass that convert XeTile to XeGPU
std::unique_ptr<::mlir::OperationPass<::mlir::gpu::GPUModuleOp>>
createConvertXeTileToXeGPUPass(const std::string &deviceName) {
  return std::make_unique<ConvertXeTileToXeGPUPass>(deviceName);
}

} // namespace imex
