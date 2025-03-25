//===-- XeVMToLLVM.cpp - XeVM to LLVM dialect conversion --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "imex/Conversion/XeGPUToXeVM/XeGPUToXeVM.h"
#include "imex/Dialect/LLVMIR/XeVMDialect.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "xegpu-to-xevm"

namespace imex {
#define GEN_PASS_DEF_CONVERTXEGPUTOXEVMPASS
#include "imex/Conversion/Passes.h.inc"
} // namespace imex

using namespace mlir;
using namespace imex::xevm;
using namespace xegpu;

namespace {

enum class NdDescI32Layout : uint32_t {
  BasePtr = 0,
  BaseShapeW = 2,
  BaseShapeH = 3,
  TensorOffsetW = 4,
  TensorOffsetH = 5
};

static int32_t getNumericXeVMAddrSpace(xegpu::MemorySpace xeGpuMemspace) {
  switch (xeGpuMemspace) {
  case xegpu::MemorySpace::Global:
    return static_cast<int>(imex::xevm::XeVMAddrSpace::kGlobal);
  case xegpu::MemorySpace::SLM:
    return static_cast<int>(imex::xevm::XeVMAddrSpace::kShared);
  }
  llvm_unreachable("Unknown XeGPU memory space.");
}

template <typename T>
std::tuple<bool, int32_t, int32_t> checkAllLinear(SmallVector<T> denseAttr) {
  assert(!denseAttr.empty());
  const int32_t intercept{static_cast<int32_t>(denseAttr[0])};
  if (denseAttr.size() < 2)
    return {true, 0, intercept};
  const T slope{denseAttr[1] - denseAttr[0]};
  for (size_t i = 1; i < denseAttr.size(); ++i)
    if (denseAttr[i] - denseAttr[i - 1] != slope)
      return {false, 0, 0};
  return {true, static_cast<int32_t>(slope), intercept};
}

mlir::VectorType encodeVectorTypeTo(mlir::VectorType currentVecType,
                                    mlir::Type toElemType) {
  auto elemType = currentVecType.getElementType();
  auto currentBitWidth = elemType.getIntOrFloatBitWidth();
  auto newBitWidth = toElemType.getIntOrFloatBitWidth();
  const int size =
      currentVecType.getNumElements() * currentBitWidth / newBitWidth;
  return mlir::VectorType::get(size, toElemType);
}

imex::xevm::LoadCacheControl
translateLoadXeGPUCacheHint(std::optional<xegpu::CachePolicy> hint) {
  auto hintVal = hint.has_value() ? hint.value() : xegpu::CachePolicy::UNCACHED;
  switch (hintVal) {
  case xegpu::CachePolicy::CACHED:
    return imex::xevm::LoadCacheControl::C;
  case xegpu::CachePolicy::UNCACHED:
    return imex::xevm::LoadCacheControl::UC;
  case xegpu::CachePolicy::STREAMING:
    return imex::xevm::LoadCacheControl::S;
  case xegpu::CachePolicy::READ_INVALIDATE:
    return imex::xevm::LoadCacheControl::IAR;
  default:
    llvm_unreachable("Unsupported cache control.");
  }
};

imex::xevm::StoreCacheControl
translateStoreXeGPUCacheHint(std::optional<xegpu::CachePolicy> hint) {
  auto hintVal = hint.has_value() ? hint.value() : xegpu::CachePolicy::UNCACHED;
  switch (hintVal) {
  case xegpu::CachePolicy::UNCACHED:
    return imex::xevm::StoreCacheControl::UC;
  case xegpu::CachePolicy::STREAMING:
    return imex::xevm::StoreCacheControl::S;
  case xegpu::CachePolicy::WRITE_BACK:
    return imex::xevm::StoreCacheControl::WB;
  case xegpu::CachePolicy::WRITE_THROUGH:
    return imex::xevm::StoreCacheControl::WT;
  default:
    llvm_unreachable("Unsupported cache control.");
  }
};

class CreateNdDescToXeVMPattern
    : public OpConversionPattern<xegpu::CreateNdDescOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::CreateNdDescOp op,
                  xegpu::CreateNdDescOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultDesc = cast<TensorDescType>(op.getResult().getType());
    auto sgMap = resultDesc.getSGMapAttr();
    if (!sgMap) {
      op.emitError() << "XeVM expects SGMap attribute to be present for tensor "
                        "descriptors";
      return mlir::failure();
    }
    auto source = op.getSource();
    Type payloadElemTy = rewriter.getI32Type();
    Type i64Ty = rewriter.getI64Type();
    VectorType payloadTy = VectorType::get(8, payloadElemTy);
    VectorType payloadI64Ty = VectorType::get(4, i64Ty);
    Value payload = rewriter.create<arith::ConstantOp>(
        loc,
        DenseElementsAttr::get(payloadTy, IntegerAttr::get(payloadElemTy, 0)));

    Value baseAddr;
    Value baseShapeW;
    Value baseShapeH;
    Value offsetW;
    Value offsetH;

    if (auto sourceTy = source.getType(); isa<MemRefType>(sourceTy)) {
      baseAddr =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, source);
      auto sourceMemrefTy = cast<MemRefType>(sourceTy);
      if (!sourceMemrefTy.hasStaticShape()) {
        op.emitError() << "Expected static memref shape.";
        return mlir::failure();
      }
      auto rank = sourceMemrefTy.getRank();
      if (rank != 2) {
        op.emitError() << "Expected a 2D memref.";
        return mlir::failure();
      }
      auto createOffset = [&](unsigned idx) -> Value {
        Value val;
        OpFoldResult ofr = op.getMixedOffsets()[idx];
        if (auto v = llvm::dyn_cast_if_present<Value>(ofr)) {
          val = rewriter.create<arith::IndexCastOp>(loc, i64Ty, v);
          val = rewriter.create<arith::TruncIOp>(loc, payloadElemTy, val);
        } else {
          int32_t off = llvm::cast<IntegerAttr>(cast<Attribute>(ofr)).getInt();
          val = rewriter.create<arith::ConstantIntOp>(loc, off, payloadElemTy);
        }
        return val;
      };
      offsetW = createOffset(rank - 1);
      offsetH = createOffset(rank - 2);
      baseShapeW = rewriter.create<arith::ConstantIntOp>(
          loc, sourceMemrefTy.getDimSize(rank - 1), payloadElemTy);
      baseShapeH = rewriter.create<arith::ConstantIntOp>(
          loc, sourceMemrefTy.getDimSize(rank - 2), payloadElemTy);
    } else if (isa<IntegerType>(sourceTy)) {
      op.emitError()
          << "Integer as source are currently not supported by the pass.";
      return mlir::failure();
    } else {
      op.emitError() << "Unknown source type.";
      return mlir::failure();
    }

    baseAddr = rewriter.create<arith::IndexCastUIOp>(loc, i64Ty, baseAddr);
    Value payLoadAsI64 =
        rewriter.create<vector::BitCastOp>(loc, payloadI64Ty, payload);
    payLoadAsI64 = rewriter.create<vector::InsertOp>(
        loc, baseAddr, payLoadAsI64,
        static_cast<int>(NdDescI32Layout::BasePtr));
    payload = rewriter.create<vector::BitCastOp>(loc, payloadTy, payLoadAsI64);
    payload = rewriter.create<vector::InsertOp>(
        loc, baseShapeW, payload,
        static_cast<int>(NdDescI32Layout::BaseShapeW));
    payload = rewriter.create<vector::InsertOp>(
        loc, baseShapeH, payload,
        static_cast<int>(NdDescI32Layout::BaseShapeH));
    payload = rewriter.create<vector::InsertOp>(
        loc, offsetW, payload,
        static_cast<int>(NdDescI32Layout::TensorOffsetW));
    payload = rewriter.create<vector::InsertOp>(
        loc, offsetH, payload,
        static_cast<int>(NdDescI32Layout::TensorOffsetH));
    rewriter.replaceOp(op, payload);
    return success();
  }
};

class UpdateNdOffsetToXeVMPattern
    : public OpConversionPattern<xegpu::UpdateNdOffsetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::UpdateNdOffsetOp op,
                  xegpu::UpdateNdOffsetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto offsets = op.getOffsets();
    auto tdesc = adaptor.getTensorDesc();
    for (size_t offsetDim = 0; offsetDim < offsets.size(); offsetDim++) {
      auto offset = offsets[offsetDim];
      if (auto cst =
              dyn_cast_if_present<arith::ConstantOp>(offset.getDefiningOp()))
        if (auto attr = dyn_cast_if_present<mlir::IntegerAttr>(cst.getValue());
            attr && !attr.getInt())
          continue;
      const int offsetPos =
          static_cast<int>(offsetDim ? NdDescI32Layout::TensorOffsetW
                                     : NdDescI32Layout::TensorOffsetH);
      auto oldOffset =
          rewriter.create<vector::ExtractOp>(loc, tdesc, offsetPos);
      offset = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getI32Type(),
                                                     offset);
      auto newOffset = rewriter.create<arith::AddIOp>(loc, oldOffset, offset);
      tdesc =
          rewriter.create<vector::InsertOp>(loc, newOffset, tdesc, offsetPos);
    }
    rewriter.replaceOp(op, tdesc);
    return success();
  }
};

template <typename OpType,
          typename = std::enable_if_t<llvm::is_one_of<
              OpType, LoadNdOp, StoreNdOp, PrefetchNdOp>::value>>
class LoadStorePrefetchNdToXeVMPattern : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();

    auto tdesc = adaptor.getTensorDesc();
    auto tdescTy = op.getTensorDescType();

    VectorType payloadI64Ty = VectorType::get(4, rewriter.getI64Type());
    Value payLoadAsI64 =
        rewriter.create<vector::BitCastOp>(loc, payloadI64Ty, tdesc);
    Value basePtr = rewriter.create<vector::ExtractOp>(
        loc, payLoadAsI64, static_cast<int>(NdDescI32Layout::BasePtr));
    Value baseShapeW = rewriter.create<vector::ExtractOp>(
        loc, tdesc, static_cast<int>(NdDescI32Layout::BaseShapeW));
    Value baseShapeH = rewriter.create<vector::ExtractOp>(
        loc, tdesc, static_cast<int>(NdDescI32Layout::BaseShapeH));
    Value offsetW = rewriter.create<vector::ExtractOp>(
        loc, tdesc, static_cast<int>(NdDescI32Layout::TensorOffsetW));
    Value offsetH = rewriter.create<vector::ExtractOp>(
        loc, tdesc, static_cast<int>(NdDescI32Layout::TensorOffsetH));
    auto ptrTypeLLVM = LLVM::LLVMPointerType::get(
        ctxt, getNumericXeVMAddrSpace(tdescTy.getMemorySpace()));
    Value basePtrLLVM =
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrTypeLLVM, basePtr);
    auto elemType = tdescTy.getElementType();
    const uint32_t elemBitSize = elemType.getIntOrFloatBitWidth();
    Value elemByteSize = rewriter.create<arith::ConstantIntOp>(
        loc, elemBitSize / 8, rewriter.getI32Type());
    Value surfaceW =
        rewriter.create<arith::MulIOp>(loc, baseShapeW, elemByteSize);

    auto tileW = tdescTy.getDimSize(1);
    auto tileH = tdescTy.getDimSize(0);
    int32_t vblocks = 1;
    if (elemBitSize == 16) {
      vblocks = (tileW + 16 - 1) / 16;
      tileW = 16;
    }

    if constexpr (std::is_same_v<OpType, StoreNdOp>) {
      VectorType srcVecTy = cast<VectorType>(op.getValue().getType());
      auto l1 = translateStoreXeGPUCacheHint(op.getL1Hint());
      auto l3 = translateStoreXeGPUCacheHint(op.getL3Hint());
      VectorType srcFlatVecTy =
          VectorType::get(srcVecTy.getNumElements(), srcVecTy.getElementType());
      Value srcFlatVec = rewriter.create<vector::ShapeCastOp>(loc, srcFlatVecTy,
                                                              op.getValue());
      srcFlatVecTy = encodeVectorTypeTo(srcFlatVecTy,
                                        rewriter.getIntegerType(elemBitSize));
      srcFlatVec =
          rewriter.create<vector::BitCastOp>(loc, srcFlatVecTy, srcFlatVec);
      rewriter.create<imex::xevm::BlockStore2dOp>(
          loc, basePtrLLVM, surfaceW, baseShapeH, surfaceW, offsetW, offsetH,
          elemBitSize, tileW, tileH, vblocks, srcFlatVec, l1, l3);
      rewriter.eraseOp(op);
    } else {
      auto l1 = translateLoadXeGPUCacheHint(op.getL1Hint());
      auto l3 = translateLoadXeGPUCacheHint(op.getL3Hint());
      if constexpr (std::is_same_v<OpType, PrefetchNdOp>) {
        rewriter.create<imex::xevm::BlockPrefetch2dOp>(
            loc, basePtrLLVM, surfaceW, baseShapeH, surfaceW, offsetW, offsetH,
            elemBitSize, tileW, tileH, vblocks, l1, l3);
        rewriter.eraseOp(op);
      } else {
        VectorType dstVecTy = cast<VectorType>(op.getValue().getType());
        const bool vnni = op.getPacked().value_or(false);
        auto transposeValue = op.getTranspose();
        bool transpose =
            transposeValue.has_value() && transposeValue.value()[0] == 1;
        VectorType loadedTy = encodeVectorTypeTo(
            dstVecTy, vnni ? rewriter.getI32Type()
                           : rewriter.getIntegerType(elemBitSize));

        Value resultFlatVec = rewriter.create<imex::xevm::BlockLoad2dOp>(
            loc, loadedTy, basePtrLLVM, surfaceW, baseShapeH, surfaceW, offsetW,
            offsetH, elemBitSize, tileW, tileH, vblocks, transpose, vnni, l1,
            l3);
        resultFlatVec = rewriter.create<vector::BitCastOp>(
            loc, encodeVectorTypeTo(loadedTy, dstVecTy.getElementType()),
            resultFlatVec);
        auto newOp =
            rewriter.create<vector::ShapeCastOp>(loc, dstVecTy, resultFlatVec);
        rewriter.replaceOp(op, newOp);
      }
    }
    return success();
  }
};

class CreateDescToXeVMPattern
    : public OpConversionPattern<xegpu::CreateDescOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::CreateDescOp op, xegpu::CreateDescOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto offsets = op.getOffsets();
    bool allLinear{false};
    int32_t slope{0};
    int32_t intercept{0};
    if (auto cstOp = dyn_cast<arith::ConstantOp>(offsets.getDefiningOp())) {
      if (auto denseAttr = cstOp->getAttrOfType<DenseElementsAttr>(
              cstOp.getValueAttrName())) {
        SmallVector<int32_t> intValues;
        for (APInt val : denseAttr.getValues<APInt>())
          intValues.push_back(static_cast<int32_t>(val.getSExtValue()));
        std::tie(allLinear, slope, intercept) = checkAllLinear(intValues);
      } else {
        op.emitError() << "Unknown offsets source, expected a dense array.";
        return failure();
      }
    } else {
      op.emitError()
          << "Unknown offsets source, must be a compile-time constant array.";
      return failure();
    }
    if (!allLinear) {
      op.emitError() << "Expected linear offsets pattern.";
      return failure();
    }

    auto memrefTy = cast<MemRefType>(op.getSource().getType());
    Value subGroupAddr =
        rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc,
                                                                op.getSource());
    Value elemByteWidth = rewriter.create<arith::ConstantIndexOp>(
        loc, memrefTy.getElementTypeBitWidth() / 8);
    Value offsetIntercept =
        rewriter.create<arith::ConstantIndexOp>(loc, intercept);
    offsetIntercept =
        rewriter.create<arith::MulIOp>(loc, elemByteWidth, offsetIntercept);
    Value offsetSlope = rewriter.create<arith::ConstantIndexOp>(loc, slope);
    offsetSlope =
        rewriter.create<arith::MulIOp>(loc, elemByteWidth, offsetSlope);
    Value laneId = rewriter.create<gpu::LaneIdOp>(loc, /*upperBound=*/nullptr);
    Value laneOffset = rewriter.create<arith::MulIOp>(loc, laneId, offsetSlope);
    laneOffset =
        rewriter.create<arith::AddIOp>(loc, laneOffset, offsetIntercept);
    auto laneAddr =
        rewriter.create<arith::AddIOp>(loc, subGroupAddr, laneOffset);
    rewriter.replaceOp(op, laneAddr);
    return success();
  }
};

class UpdateOffsetToXeVMPattern
    : public OpConversionPattern<xegpu::UpdateOffsetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::UpdateOffsetOp op,
                  xegpu::UpdateOffsetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto elemByteSize =
        op.getTensorDesc().getType().getElementType().getIntOrFloatBitWidth() /
        8;
    Value laneId = rewriter.create<gpu::LaneIdOp>(loc, /*upperBound=*/nullptr);
    Value offsetForLane =
        rewriter.create<vector::ExtractOp>(loc, adaptor.getOffsets(), laneId);
    Value factor = rewriter.create<arith::ConstantIndexOp>(loc, elemByteSize);
    offsetForLane = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), offsetForLane);
    offsetForLane = rewriter.create<arith::MulIOp>(loc, factor, offsetForLane);
    Value newOffsetForLane = rewriter.create<arith::AddIOp>(
        loc, adaptor.getTensorDesc(), offsetForLane);
    rewriter.replaceOp(op, newOffsetForLane);
    return success();
  }
};

template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                               OpType, LoadGatherOp, StoreScatterOp>::value>>
class LoadStoreToXeVMPattern : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    auto tdesc = op.getTensorDescType();
    auto ptrTypeLLVM = LLVM::LLVMPointerType::get(
        ctxt, getNumericXeVMAddrSpace(tdesc.getMemorySpace()));
    Value basePtrI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), adaptor.getTensorDesc());
    Value basePtrLLVM =
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrTypeLLVM, basePtrI64);
    VectorType srcOrDstVecTy = cast<VectorType>(op.getValue().getType());
    VectorType srcOrDstFlatVecTy = VectorType::get(
        srcOrDstVecTy.getNumElements(), srcOrDstVecTy.getElementType());
    if constexpr (std::is_same_v<OpType, LoadGatherOp>) {
      Value loaded =
          rewriter.create<LLVM::LoadOp>(loc, srcOrDstFlatVecTy, basePtrLLVM);
      auto newOp =
          rewriter.create<vector::ShapeCastOp>(loc, srcOrDstVecTy, loaded);
      rewriter.replaceOp(op, newOp);
    } else {
      Value srcFlatVec = rewriter.create<vector::ShapeCastOp>(
          loc, srcOrDstFlatVecTy, op.getValue());
      rewriter.create<LLVM::StoreOp>(loc, srcFlatVec, basePtrLLVM);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

class PrefetchToXeVMPattern : public OpConversionPattern<xegpu::PrefetchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::PrefetchOp op, xegpu::PrefetchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    auto tdescTy = op.getTensorDescType();
    auto ptrTypeLLVM = LLVM::LLVMPointerType::get(
        ctxt, getNumericXeVMAddrSpace(tdescTy.getMemorySpace()));
    Value basePtrI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), adaptor.getTensorDesc());
    Value ptrLLVM =
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrTypeLLVM, basePtrI64);
    rewriter.create<imex::xevm::PrefetchOp>(
        loc, ptrLLVM, imex::xevm::AddrSpace::GLOBAL,
        translateLoadXeGPUCacheHint(op.getL1Hint()),
        translateLoadXeGPUCacheHint(op.getL3Hint()));
    return success();
  }
};
class FenceToXeVMPattern : public OpConversionPattern<xegpu::FenceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::FenceOp op, xegpu::FenceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    imex::xevm::MemoryScope memScope;
    switch (op.getFenceScope()) {
    case xegpu::FenceScope::Workgroup:
      memScope = imex::xevm::MemoryScope::WORKGROUP;
      break;
    case xegpu::FenceScope::GPU:
      memScope = imex::xevm::MemoryScope::GPU;
      break;
      llvm_unreachable("Unknown XeGPU fence scope.");
    }
    imex::xevm::AddrSpace addrSpace;
    switch (op.getMemoryKind()) {
    case xegpu::MemorySpace::Global:
      addrSpace = imex::xevm::AddrSpace::GLOBAL;
      break;
    case xegpu::MemorySpace::SLM:
      addrSpace = imex::xevm::AddrSpace::SHARED;
      break;
      llvm_unreachable("Unknown XeGPU fence scope.");
    }
    rewriter.create<imex::xevm::MemfenceOp>(loc, memScope, addrSpace);
    rewriter.eraseOp(op);
    return success();
  }
};

class DpasToXeVMPattern : public OpConversionPattern<xegpu::DpasOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::DpasOp op, xegpu::DpasOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    auto aTy = mlir::cast<VectorType>(op.getLhs().getType());
    auto bTy = mlir::cast<VectorType>(op.getRhs().getType());
    auto resultType = mlir::cast<VectorType>(op.getResultType());

    auto encodePrecision = [&](Type type) -> imex::xevm::PrecisionType {
      if (type == rewriter.getBF16Type())
        return imex::xevm::PrecisionType::BF16;
      else if (type == rewriter.getF16Type())
        return imex::xevm::PrecisionType::FP16;
      else if (type == rewriter.getTF32Type())
        return imex::xevm::PrecisionType::TF32;
      else if (type.isInteger(8)) {
        if (type.isUnsignedInteger())
          return imex::xevm::PrecisionType::U8;
        return imex::xevm::PrecisionType::S8;
      }
      llvm_unreachable("add more support for PrecisionType");
      return imex::xevm::PrecisionType::UNUSED;
    };
    imex::xevm::PrecisionType precATy = encodePrecision(aTy.getElementType());
    imex::xevm::PrecisionType precBTy = encodePrecision(bTy.getElementType());
    auto precA = imex::xevm::PrecisionTypeAttr::get(ctxt, precATy);
    auto precB = imex::xevm::PrecisionTypeAttr::get(ctxt, precBTy);
    Value c = op.getAcc();
    if (!c) {
      auto elementTy = resultType.getElementType();
      Attribute initValueAttr;
      if (isa<FloatType>(elementTy))
        initValueAttr = FloatAttr::get(elementTy, 0.0);
      else
        initValueAttr = IntegerAttr::get(elementTy, 0);
      c = rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(resultType, initValueAttr));
    }
    auto rc = IntegerAttr::get(rewriter.getI32Type(), 8);

    VectorType aNty =
        VectorType::get(aTy.getNumElements(), aTy.getElementType());
    Value aVec = rewriter.create<vector::ShapeCastOp>(loc, aNty, op.getLhs());

    VectorType bNty =
        VectorType::get(bTy.getNumElements(), bTy.getElementType());
    Value bVec = rewriter.create<vector::ShapeCastOp>(loc, bNty, op.getRhs());

    auto cvecty = cast<VectorType>(c.getType());
    VectorType cNty =
        VectorType::get(cvecty.getNumElements(), cvecty.getElementType());
    if (cvecty != cNty)
      c = rewriter.create<vector::ShapeCastOp>(loc, cNty, c);
    Value dpasRes = rewriter.create<imex::xevm::DpasOp>(loc, cNty, c, aVec,
                                                        bVec, precA, precB, rc);
    if (cvecty != cNty)
      dpasRes = rewriter.create<vector::ShapeCastOp>(loc, resultType, dpasRes);
    rewriter.replaceOp(op, dpasRes);
    return success();
  }
};

static std::optional<LLVM::AtomicBinOp>
matchSimpleAtomicOp(arith::AtomicRMWKind arithKind) {
  switch (arithKind) {
  case arith::AtomicRMWKind::addf:
    return LLVM::AtomicBinOp::fadd;
  case arith::AtomicRMWKind::addi:
    return LLVM::AtomicBinOp::add;
  case arith::AtomicRMWKind::assign:
    return LLVM::AtomicBinOp::xchg;
  case arith::AtomicRMWKind::maximumf:
    return LLVM::AtomicBinOp::fmax;
  case arith::AtomicRMWKind::maxs:
    return LLVM::AtomicBinOp::max;
  case arith::AtomicRMWKind::maxu:
    return LLVM::AtomicBinOp::umax;
  case arith::AtomicRMWKind::minimumf:
    return LLVM::AtomicBinOp::fmin;
  case arith::AtomicRMWKind::mins:
    return LLVM::AtomicBinOp::min;
  case arith::AtomicRMWKind::minu:
    return LLVM::AtomicBinOp::umin;
  case arith::AtomicRMWKind::ori:
    return LLVM::AtomicBinOp::_or;
  case arith::AtomicRMWKind::andi:
    return LLVM::AtomicBinOp::_and;
  default:
    return std::nullopt;
  }
  llvm_unreachable("Invalid AtomicRMWKind");
}

class AtomicRMWToXeVMPattern : public OpConversionPattern<xegpu::AtomicRMWOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xegpu::AtomicRMWOp op, xegpu::AtomicRMWOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctxt = rewriter.getContext();
    auto tdesc = op.getTensorDesc().getType();
    auto ptrTypeLLVM = LLVM::LLVMPointerType::get(
        ctxt, getNumericXeVMAddrSpace(tdesc.getMemorySpace()));
    Value basePtrI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), adaptor.getTensorDesc());
    Value basePtrLLVM =
        rewriter.create<LLVM::IntToPtrOp>(loc, ptrTypeLLVM, basePtrI64);
    VectorType srcOrDstVecTy = cast<VectorType>(op.getValue().getType());
    VectorType srcOrDstFlatVecTy = VectorType::get(
        srcOrDstVecTy.getNumElements(), srcOrDstVecTy.getElementType());
    Value srcFlatVec = rewriter.create<vector::ShapeCastOp>(
        loc, srcOrDstFlatVecTy, op.getValue());
    auto atomicKind = matchSimpleAtomicOp(op.getKind());
    assert(atomicKind.has_value());
    Value resVec = srcFlatVec;
    for (int i = 0; i < srcOrDstVecTy.getNumElements(); i++) {
      auto val = rewriter.create<vector::ExtractOp>(loc, resVec, i);
      Value idx = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(),
                                                    rewriter.getIndexAttr(i));
      Value currPtr = rewriter.create<LLVM::GEPOp>(
          loc, ptrTypeLLVM, srcOrDstVecTy.getElementType(), basePtrLLVM, idx);
      Value newVal = rewriter.create<LLVM::AtomicRMWOp>(
          loc, atomicKind.value(), currPtr, val, LLVM::AtomicOrdering::seq_cst);
      resVec = rewriter.create<vector::InsertOp>(loc, newVal, resVec, i);
    }
    rewriter.replaceOp(op, resVec);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ConvertXeGPUToXeVMPass
    : public imex::impl::ConvertXeGPUToXeVMPassBase<ConvertXeGPUToXeVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, XeGPUDialect, imex::xevm::XeVMDialect,
                    vector::VectorDialect, arith::ArithDialect,
                    memref::MemRefDialect, gpu::GPUDialect>();
  }

  void runOnOperation() override {
    LLVMTypeConverter typeConverter(&getContext());
    typeConverter.addConversion([&](IndexType type) -> Type { return type; });
    typeConverter.addConversion([&](VectorType type) -> Type {
      unsigned rank = type.getRank();
      auto elemType = type.getElementType();
      if (llvm::isa<mlir::IndexType>(elemType))
        elemType = mlir::IntegerType::get(&getContext(), 64);
      if (rank < 1 || type.getNumElements() == 1)
        return elemType;
      unsigned sum = 1;
      for (unsigned i = 0; i < rank; i++) {
        sum *= type.getShape()[i];
      }
      return VectorType::get(sum, elemType);
    });
    typeConverter.addConversion([&](xegpu::TensorDescType type) -> Type {
      if (type.isScattered()) {
        return IndexType::get(&getContext());
      }
      auto i32Type = IntegerType::get(&getContext(), 32);
      return VectorType::get(8, i32Type);
    });

    ConversionTarget target(getContext());
    target.addLegalDialect<imex::xevm::XeVMDialect, LLVM::LLVMDialect,
                           vector::VectorDialect, arith::ArithDialect,
                           memref::MemRefDialect, gpu::GPUDialect>();
    target.addIllegalDialect<XeGPUDialect>();

    RewritePatternSet patterns(&getContext());
    imex::populateXeGPUToXeVMConversionPatterns(patterns, typeConverter);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//
void imex::populateXeGPUToXeVMConversionPatterns(
    RewritePatternSet &patterns, LLVMTypeConverter &typeConverter) {
  patterns.add<CreateNdDescToXeVMPattern, UpdateNdOffsetToXeVMPattern,
               LoadStorePrefetchNdToXeVMPattern<xegpu::LoadNdOp>,
               LoadStorePrefetchNdToXeVMPattern<xegpu::StoreNdOp>,
               LoadStorePrefetchNdToXeVMPattern<xegpu::PrefetchNdOp>>(
      typeConverter, patterns.getContext());
  patterns.add<CreateDescToXeVMPattern, UpdateOffsetToXeVMPattern,
               AtomicRMWToXeVMPattern, PrefetchToXeVMPattern,
               LoadStoreToXeVMPattern<xegpu::LoadGatherOp>,
               LoadStoreToXeVMPattern<xegpu::StoreScatterOp>>(
      typeConverter, patterns.getContext());
  patterns.add<FenceToXeVMPattern, DpasToXeVMPattern>(typeConverter,
                                                      patterns.getContext());
}
