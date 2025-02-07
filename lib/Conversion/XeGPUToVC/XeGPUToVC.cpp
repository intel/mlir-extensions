//===- XeGPUToVC.cpp -  XeGPU to VC Lowering pass  --------------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a pass to generate Func calls to intel VC intrinsics
/// functions for XeGPU dialect ops
///
//===----------------------------------------------------------------------===//

#include "imex/Conversion/XeGPUToVC/XeGPUToVC.h"
#include "imex/Conversion/ArithToVC/ArithToVC.h"
#include "imex/Conversion/MathToVC/MathToVC.h"
#include "imex/Utils/VCUtils.h"
#include "imex/Utils/XeCommon.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>

#include "LscIntrinsicEnums.h"

namespace imex {
#define GEN_PASS_DEF_CONVERTXEGPUTOVC
#include "imex/Conversion/Passes.h.inc"
} // namespace imex

using namespace mlir;
using mlir::scf::ForOp;
using mlir::scf::YieldOp;
using mlir::vector::ShapeCastOp;
using mlir::xegpu::AllocNbarrierOp;
using mlir::xegpu::CompileHintOp;
using mlir::xegpu::CreateDescOp;
using mlir::xegpu::CreateNdDescOp;
using mlir::xegpu::InitNbarrierOp;
using mlir::xegpu::NbarrierWaitOp;
using mlir::xegpu::UpdateNdOffsetOp;
using mlir::xegpu::UpdateOffsetOp;

namespace imex {

extern void populateAtomicAndFenceLSCPatterns(TypeConverter &converter,
                                              RewritePatternSet &patterns);
extern void populateLoadStoreLSCPatterns(TypeConverter &converter,
                                         RewritePatternSet &patterns);

static bool isZero(OpFoldResult ofr) { return isConstantIntValue(ofr, 0); }

static bool isOneOrUnknow(OpFoldResult ofr) {
  auto val = getConstantIntValue(ofr);
  return !val || *val == 1;
}

static Value castValueTo(Value val, Type toType, Location loc,
                         ConversionPatternRewriter &rewriter) {

  auto fromType = val.getType();
  if (fromType == toType)
    return val;

  auto cst = val.getDefiningOp<arith::ConstantOp>();
  if (cst) {
    auto attr = dyn_cast<IntegerAttr>(cst.getValue());
    if (attr)
      return rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(toType, attr.getInt()));
  }

  if (auto vectorTy = dyn_cast<VectorType>(fromType)) {
    if (vectorTy.getElementType().isIndex()) {
      return rewriter.createOrFold<arith::IndexCastUIOp>(loc, toType, val);
    }
  }

  if (fromType.isIndex() && toType.isInteger())
    return rewriter.createOrFold<arith::IndexCastUIOp>(loc, toType, val);

  // return original value for unsupported conversion
  return val;
}

// Given an n-dim memref, a tensor descriptor with tile rank of 2 defines a
// 2d memory region with respect to the two inner-most dimensions. Other
// outer dimensions affect the base address of the 2d plane. For 2d, we
// compute the base address of 2d plane, assuming the coordinates [0, 0] for
// the innermost 2 dimensions. The payload will record tile offset within
// the 2d plane in separate field. For example, given
//   %m: memref<2x7x32x64xf16>
// And this access
//   %m[%a, %b, %c, %d]
//
// The base address will be adjusted as follows:
//   base address of plane for 2d tile = base(%m) + %b * (32*64*2) + %a *
//                                       (7*32*64*2)
// 2 is the number of bytes of the element type.
//
// For 1d, we compute the base address of the 1d tile, not the plane.
// So the tile offset is also added to the base address.
//
// For tile rank of 1, the base address will be adjusted as:
//   base address of tile for 1d tile = base(%m) + %d * (2) + %c * (64*2) +
//                                      %b * (32*64*2) + %a * (7*32*64*2)
static Value adjustBasePointer(ConversionPatternRewriter &rewriter,
                               CreateNdDescOp op, Value memrefBaseAddr) {
  auto loc = op.getLoc();
  auto tileRank = op.getTensorDesc().getType().getRank();
  auto offsets = op.getMixedOffsets();

  auto strides = getStridesOrOffsetsOrShapesInValueType(
      rewriter, op.getMixedStrides(), loc);

  // Calculate the effective rank of the source based on strides arrayref
  // size
  auto effectiveRank = strides.size();
  int64_t ranksToAdjust = effectiveRank;
  auto eTyBitWidth =
      op.getTensorDesc().getType().getElementType().getIntOrFloatBitWidth();
  auto bytesPerElem = eTyBitWidth / 8;
  Value eTyBitWidthVal = index_val(eTyBitWidth);
  Value bytesPerElemVal = index_val(bytesPerElem);

  // We only need combine ranks that are larger than tileRank (e.g., if we the
  // source is 4-D, and the tile is 2-D, we only need to combine/adjust the base
  // for 4-2=2 ranks/dims )
  ranksToAdjust -= tileRank;
  offsets.pop_back_n(tileRank);

  auto computeBase = [&](Value base) {
    for (auto i = 0; i < ranksToAdjust; i++) {
      Value factor;
      if (eTyBitWidth < 8)
        factor = muli(strides[i], eTyBitWidthVal);
      else
        factor = muli(strides[i], bytesPerElemVal);
      auto offsetVal =
          mlir::getValueOrCreateConstantIndexOp(rewriter, loc, offsets[i]);
      auto linearOffset = muli(offsetVal, factor);
      if (eTyBitWidth < 8)
        linearOffset = divi(linearOffset, index_val(8));
      base = addi(base, linearOffset);
    }

    return base;
  };

  return computeBase(memrefBaseAddr);
}

class CreateNdDescPattern : public OpConversionPattern<CreateNdDescOp> {
public:
  using OpConversionPattern<CreateNdDescOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CreateNdDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tdescTy = op.getType();
    auto scope = tdescTy.getMemorySpace();
    auto rank = tdescTy.getRank();
    auto eTyBitWidth = tdescTy.getElementType().getIntOrFloatBitWidth();

    // SLM has to use 32-bit address, while ugm needs to use 64-bit address.
    auto addrTy =
        (scope == xegpu::MemorySpace::SLM) ? (Type)i32Ty : (Type)i64Ty;

    // Handle different source types: memref and i64/i32/ui64/ui32
    auto memRefType = dyn_cast<MemRefType>(op.getSource().getType());
    Value base;
    if (memRefType)
      base = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
          loc, adaptor.getSource());
    else { // Handle i64/i32/ui64/ui32 passed as a source
      base = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(),
                                                   op.getSource());
    }

    base = adjustBasePointer(rewriter, op, base);
    base = rewriter.create<arith::IndexCastUIOp>(loc, addrTy, base);

    if (scope == xegpu::MemorySpace::SLM || rank == 1) {
      // for SLM and 1D, we need to create message for use regular load/store
      // instead of matrix descriptor, the shape of accepted TensorDescs are
      // limited to 1xN (rank = 2 with leading dimension to be 1) or N (rank =
      // 1). It is similar to CreateDesc, but simd lanes fixed to 1.
      assert(rank == 1 && "Currently only rank 1 is supported for SLM.");

      // as mentioned above, the payload is fixed to vector<1xi32> for slm or
      // vector<1xindex> for others
      const int simd_lanes = 1;
      auto payloadTy = VectorType::get(simd_lanes, addrTy);

      // adjust base address to get absolute offset in unit of bytes.
      // the computation is simply: base + linearOffset (in bytes)
      Value offset =
          getValueOrConstantOp(op.getMixedOffsets().back(), loc, rewriter);
      offset = castValueTo(offset, addrTy, loc, rewriter);
      Value numOffsetBytes =
          getOffsetInUnitOfBytes(rewriter, loc, addrTy, offset, eTyBitWidth);
      auto payload = addi(base, numOffsetBytes);

      // convert the payload into vector type
      payload = rewriter.create<vector::BroadcastOp>(loc, payloadTy, payload);

      rewriter.replaceOp(op, payload);
      return success();
    } else if (rank == 2) {
      // matrix descriptor (payload) is represented as vector<16xi32>
      // since base address is 64-bit and it has to be 64 bytes aligned,
      // we start with vector<8xi64> for convinience and later cast it to
      // vector<16xi32> for payload.
      // matrix descriptor (v16i32) encodes the following information:
      //    DWORD0..DWORD1 [63:0] 2D Block Base Address
      //    DWORD2: Matrix width in bytes, minus 1
      //    DWORD3: Matrix height in rows, minus 1
      //    DWORD4: Matrix pitch in bytes, minus 1
      //    DWORD5: Block Start OffsetX in elements, signed
      //    DWORD6: Block Start OffsetY in elements, signed
      //    DWORD7: Block size encoded as follows:
      //            [7:0]: block width in elements, minus 1
      //           [15:8]: block height in elements, minus 1
      //          [16:23]: number of blocks (array_length), minus 1
      //    DWORD[8-15]: Reserved
      Value payload = dense_vector_int_val(0, i64Ty, 8);

      // encode base address
      payload = rewriter.create<vector::InsertOp>(loc, base, payload, 0);

      // In Matrix descriptor (payload), shape and offset are encoded
      // with 32-bit data
      auto encodeShapeAndOffset = [&](OpFoldResult ofr, unsigned mul,
                                      unsigned minus = 0) -> Value {
        auto value = llvm::dyn_cast_if_present<Value>(ofr);
        if (value) {
          value = rewriter.create<arith::IndexCastUIOp>(loc, i32Ty, value);
          if (mul > 8)
            value =
                rewriter.create<arith::MulIOp>(loc, value, i32_val(mul / 8));
          else if (mul >= 4)
            value = getOffsetInUnitOfBytes(rewriter, loc, i32Ty, value, mul);
          return (!minus) ? value : subi(value, i32_val(minus));
        } else {
          int value =
              llvm::cast<IntegerAttr>(llvm::cast<Attribute>(ofr)).getInt();
          return i32_val(((value * mul) / 8) - minus);
        }
      };

      payload =
          rewriter.create<vector::BitCastOp>(loc, vecTy(16, i32Ty), payload);

      // encode the surface width and height. width is in bytes minus 1, height
      // is in rows.
      auto matrixShape = op.getMixedSizes();
      auto size = matrixShape.size();
      auto surfaceW =
          encodeShapeAndOffset(matrixShape[size - 1], eTyBitWidth, 1);
      auto surfaceH = encodeShapeAndOffset(matrixShape[size - 2], 8, 1);

      // encode the pitch, which is in bytes minus 1
      auto matrixStrides = op.getMixedStrides();
      size = matrixStrides.size();
      // if strides are static, the fast changing dim has to be 1.
      // Otherwise (referred as unknow, e.g., passed via func arguments),
      // we assume users give correct setups.
      assert(isOneOrUnknow(matrixStrides[size - 1]) &&
             "Fast Changing Dimension can only have stride of 1.");
      auto surfaceP =
          encodeShapeAndOffset(matrixStrides[size - 2], eTyBitWidth, 1);

      payload = rewriter.create<vector::InsertOp>(loc, surfaceW, payload, 2);
      payload = rewriter.create<vector::InsertOp>(loc, surfaceH, payload, 3);
      payload = rewriter.create<vector::InsertOp>(loc, surfaceP, payload, 4);

      // encode the offset, they are in elements
      auto offsets = op.getMixedOffsets();
      auto offsetX = encodeShapeAndOffset(offsets[size - 1], 8, 0);
      auto offsetY = encodeShapeAndOffset(offsets[size - 2], 8, 0);
      payload = rewriter.create<vector::InsertOp>(loc, offsetX, payload, 5);
      payload = rewriter.create<vector::InsertOp>(loc, offsetY, payload, 6);

      // encode the block size
      int nblks = tdescTy.getArrayLength();
      int blkW = tdescTy.getShape()[1];
      int blkH = tdescTy.getShape()[0];
      auto block =
          i32_val(((nblks - 1) << 16) | ((blkH - 1) << 8) | (blkW - 1));
      payload = rewriter.create<vector::InsertOp>(loc, block, payload, 7);
      rewriter.replaceOp(op, payload);
    } else {
      llvm_unreachable("Unsupported TensorDesc.");
    }
    return success();
  }
};

class UpdateNDOffsetPattern : public OpConversionPattern<UpdateNdOffsetOp> {
public:
  using OpConversionPattern<UpdateNdOffsetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UpdateNdOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto tdescTy = op.getTensorDescType();
    auto scope = tdescTy.getMemorySpace();
    auto rank = tdescTy.getRank();

    auto addrTy =
        (scope == xegpu::MemorySpace::SLM) ? (Type)i32Ty : (Type)i64Ty;

    auto desc = adaptor.getTensorDesc();
    if (scope == xegpu::MemorySpace::SLM || rank == 1) {
      // for SLM and 1D, we need to create message for use regular load/store
      // instead of matrix descriptor

      auto offsets = op.getMixedOffsets();
      // since stride info is not available, the leading ranks
      // of offsets have be to be 0, otherwise we cannot generate
      // correct codes
      for (auto i = 0; i < rank - 1; i++) {
        if (!isZero(offsets[i]))
          return rewriter.notifyMatchFailure(op, "unsupported TensorDescType.");
      }

      // update offset from unit of elements to unit of bytes
      auto offset = getValueOrConstantOp(offsets.back(), loc, rewriter);
      offset = castValueTo(offset, addrTy, loc, rewriter);
      auto eTyBitWidth = tdescTy.getElementType().getIntOrFloatBitWidth();
      offset =
          getOffsetInUnitOfBytes(rewriter, loc, addrTy, offset, eTyBitWidth);

      // convert offset to vector type and update the payload
      const int simd_lanes = 1;
      auto payloadTy = VectorType::get(simd_lanes, addrTy);
      offset = rewriter.create<vector::BroadcastOp>(loc, payloadTy, offset);

      auto payload = addi(desc, offset);
      rewriter.replaceOp(op, payload);
      return success();
    } else if (rank == 2) {
      auto offsets = op.getMixedOffsets();
      assert(offsets.size() >= 2 && "Invalid offsets");
      auto offsetXY = llvm::ArrayRef<OpFoldResult>(offsets).take_back(2);

      // {5, 6} are the indices of DWORD5 and DWORD6 in the payload
      // {1, 0} are the indices of OffsetX and OffsetY in the offsets
      int payloadIndexs[] = {5, 6};
      int offsetXYIndexs[] = {1, 0};
      for (auto [i, j] : llvm::zip_equal(payloadIndexs, offsetXYIndexs)) {
        // no need to update if offset is zero
        if (isZero(offsetXY[j]))
          continue;

        auto offset = getValueOrConstantOp(offsetXY[j], loc, rewriter);
        offset = castValueTo(offset, i32Ty, loc, rewriter);

        // Get 2D Block OffsetX/OffsetY from DWORD5/DWORD6 of payload
        auto oldOffset = rewriter.create<vector::ExtractOp>(loc, desc, i);

        auto newOffset = addi(oldOffset, offset);
        desc = rewriter.create<vector::InsertOp>(loc, newOffset, desc, i);
      }
    } else {
      llvm_unreachable("unsupported TensorDesc.");
    }
    rewriter.replaceOp(op, desc);
    return success();
  }
};

class CreateDescPattern : public OpConversionPattern<CreateDescOp> {
public:
  using OpConversionPattern<CreateDescOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CreateDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tdescTy = op.getTensorDescType();
    auto elemTy = tdescTy.getElementType();
    assert(elemTy.isIntOrFloat() && "only support int or float element type.");

    // use 32-bit address for SLM and 64-bit address for UGM
    auto scope = tdescTy.getMemorySpace();
    auto addrTy = scope == xegpu::MemorySpace::SLM ? (Type)i32Ty : (Type)i64Ty;

    Value base = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, adaptor.getSource());
    base = rewriter.create<arith::IndexCastUIOp>(loc, addrTy, base);

    // Using an 1-D vector of index type elements to represent the payload
    // It essentially holds the absolute address of the base pointer with
    // each element in the vector representing the address for a simd land
    auto simd_lanes = tdescTy.getShape()[0];
    auto payloadTy = vecTy(simd_lanes, addrTy);

    // offset is represented in number of elements, need to scale it to bytes
    auto eTyBitWidth = elemTy.getIntOrFloatBitWidth();
    Value offsets = castValueTo(adaptor.getOffsets(), payloadTy, loc, rewriter);
    offsets = getVecOffsetInUnitOfBytes(rewriter, loc, simd_lanes, addrTy,
                                        offsets, eTyBitWidth);

    // create a payload with the base address broadcasted to all simd lanes
    Value payload = rewriter.create<vector::BroadcastOp>(loc, payloadTy, base);

    // performing base + offsets to get the final address per simd lane
    payload = addi(payload, offsets);

    rewriter.replaceOp(op, payload);
    return success();
  }
};

class UpdateOffsetOpPattern : public OpConversionPattern<UpdateOffsetOp> {
public:
  using OpConversionPattern<UpdateOffsetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UpdateOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tdescTy = op.getTensorDescType();
    auto elemTy = tdescTy.getElementType();

    assert(elemTy.isIntOrFloat() && "only support int or float element type.");

    // use 32-bit address for SLM and 64-bit address for UGM
    auto scope = tdescTy.getMemorySpace();
    auto addrTy = scope == xegpu::MemorySpace::SLM ? (Type)i32Ty : (Type)i64Ty;

    auto simd_lanes = tdescTy.getShape()[0];
    auto payloadTy = VectorType::get(simd_lanes, addrTy);

    auto eTyBitWidth = elemTy.getIntOrFloatBitWidth();
    Value offsets = castValueTo(adaptor.getOffsets(), payloadTy, loc, rewriter);
    offsets = getVecOffsetInUnitOfBytes(rewriter, loc, simd_lanes, addrTy,
                                        offsets, eTyBitWidth);

    auto payload = addi(adaptor.getTensorDesc(), offsets);
    rewriter.replaceOp(op, payload);
    return success();
  }
};

class DpasPattern : public OpConversionPattern<DpasOp> {
public:
  using OpConversionPattern<DpasOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DpasOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhsType = op.getLhsType();
    auto rhsType = op.getRhsType();
    auto resultType = op.getResultType();
    uint8_t rc = lhsType.getShape()[0];
    uint8_t sd = rhsType.getShape()[0];
    auto encodePrecision = [&](Type type) -> uint8_t {
      if (type.isBF16())
        return (uint8_t)GenPrecision::BF16;
      else if (type.isF16())
        return (uint8_t)GenPrecision::FP16;
      else if (type.isTF32())
        return (uint8_t)GenPrecision::TF32;
      else {
        assert(0 && "add more support");
        return (uint8_t)GenPrecision::INVALID;
      }
    };
    uint8_t prec1 = encodePrecision(rhsType.getElementType());
    uint8_t prec2 = encodePrecision(lhsType.getElementType());
    unsigned infoVal = (rc << 24) | (sd << 16) | (prec2 << 8) | (prec1);
    auto infoAttr = rewriter.getIntegerAttr(rewriter.getI32Type(), infoVal);
    auto info = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(),
                                                   infoAttr);

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    // Intrinsic accepts i32 type, therefore the element type should be casted
    // to i32
    auto [lhsName, lhsNewType] = encodeVectorType(rewriter, lhsType);
    auto [rhsName, rhsNewType] = encodeVectorType(rewriter, rhsType);
    auto [resultName, newResultType] =
        encodeVectorType(rewriter, resultType, /*use64bitData=*/false,
                         /*enforceInteger=*/false, /*keepF16=*/true);

    if (lhsNewType != lhs.getType()) {
      lhs = rewriter.create<vector::BitCastOp>(loc, lhsNewType, lhs);
    }
    if (rhsNewType != rhs.getType()) {
      rhs = rewriter.create<vector::BitCastOp>(loc, rhsNewType, rhs);
    }
    SmallVector<Value, 4> args{rhs, lhs, info};
    std::string funcName = "llvm.genx.dpas.nosrc0.";
    if (op.getAcc()) {
      funcName = "llvm.genx.dpas2.";
      auto i32Type = rewriter.getI32Type();
      auto createIntConstant = [&](Type type, unsigned value) {
        auto attr = rewriter.getIntegerAttr(type, value);
        return rewriter.create<arith::ConstantOp>(loc, type, attr);
      };
      auto prec1Arg = createIntConstant(i32Type, prec1);
      auto prec2Arg = createIntConstant(i32Type, prec2);
      auto sdArg = createIntConstant(i32Type, sd);
      auto rcArg = createIntConstant(i32Type, rc);
      auto signless = createIntConstant(i32Type, 0);
      args.assign({adaptor.getAcc(), rhs, lhs, prec1Arg, prec2Arg, sdArg, rcArg,
                   signless, signless});
    }
    funcName += resultName;
    funcName += ".";
    funcName += rhsName;
    funcName += ".";
    funcName += lhsName;
    auto funcOp = createFuncCall(rewriter, loc, funcName,
                                 TypeRange{newResultType}, args, false);
    rewriter.replaceOp(op, funcOp.getResult(0));
    return success();
  }
};

class AllocNbarrierPattern : public OpConversionPattern<AllocNbarrierOp> {
public:
  using OpConversionPattern<AllocNbarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AllocNbarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    auto func = op->getParentOfType<gpu::GPUFuncOp>();
    rewriter.setInsertionPointAfter(func);
    auto executionModeAttr = spirv::ExecutionModeAttr::get(
        rewriter.getContext(), spirv::ExecutionMode::NamedBarrierCountINTEL);

    auto execModeFuncAttr = spirv::ExecutionModeFuncAttributeAttr::get(
        rewriter.getContext(), executionModeAttr, op.getNbarrierNum());

    func->setAttr("spirv.execution_mode", execModeFuncAttr);

    rewriter.eraseOp(op);
    return success();
  }
};

#define zext(...) rewriter.createOrFold<arith::ExtUIOp>(loc, __VA_ARGS__)
#define logic_shl(...) rewriter.createOrFold<arith::ShLIOp>(loc, __VA_ARGS__)
#define bitwise_or(...) rewriter.createOrFold<arith::OrIOp>(loc, __VA_ARGS__)
#define bitwise_and(...) rewriter.createOrFold<arith::AndIOp>(loc, __VA_ARGS__)

class InitNbarrierPattern : public OpConversionPattern<InitNbarrierOp> {
public:
  using OpConversionPattern<InitNbarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InitNbarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto nbarrier_id = op.getNbarrierId();

    // a participant is both a producer or a consumer (0)
    auto nbarrier_role = i8_val(0);
    auto num_participants = op.getParticipantThreadNum();
    Value num_producers = num_participants;
    Value num_consumers = num_participants;

    auto nbarrier = rewriter.create<::mlir::UnrealizedConversionCastOp>(
        loc, ::mlir::TypeRange{op.getType()},
        ::mlir::ValueRange{nbarrier_id, nbarrier_role, num_producers,
                           num_consumers});
    rewriter.replaceOp(op, nbarrier);

    return success();
  }
};

class NbarrierArrivePattern : public OpConversionPattern<NbarrierArriveOp> {
public:
  using OpConversionPattern<NbarrierArriveOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NbarrierArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto payload = adaptor.getNbarrier();
    auto dop = payload.getDefiningOp();

    std::string funcName = "llvm.genx.nbarrier.arrive";

    SmallVector<Value> args{dop->getOperand(0), dop->getOperand(1),
                            dop->getOperand(2), dop->getOperand(3)};

    createFuncCall(rewriter, loc, funcName, TypeRange{}, args, false);

    rewriter.eraseOp(op);
    return success();
  }
};

class NbarrierWaitPattern : public OpConversionPattern<NbarrierWaitOp> {
public:
  using OpConversionPattern<NbarrierWaitOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NbarrierWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto payload = adaptor.getNbarrier();
    auto nbarrier_id = payload.getDefiningOp()->getOperand(0);

    Value signal_flag = i8_val(0); // 0b0: wait 0b1: signal
    Value num_threads = i8_val(0); // This field is ignored for nbarrier.wait

    std::string funcName = "llvm.genx.nbarrier";
    SmallVector<Value> args{signal_flag, nbarrier_id, num_threads};

    createFuncCall(rewriter, loc, funcName, TypeRange{}, args, false);
    rewriter.eraseOp(op);
    return success();
  }
};

class CompilerHintPattern : public OpConversionPattern<CompileHintOp> {
public:
  using OpConversionPattern<CompileHintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CompileHintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    std::string funcName = "llvm.genx.fence";
    Value fence_flag = i8_val(-128);
    SmallVector<Value> args{fence_flag};

    createFuncCall(rewriter, loc, funcName, TypeRange{}, args, false);
    rewriter.eraseOp(op);
    return success();
  }
};

bool isLegalXeGPUSCFOp(Operation *op, TypeConverter typeConverter) {
  llvm::SmallVector<Value> args;
  if (llvm::isa<ForOp>(op))
    args = llvm::cast<ForOp>(op).getInitArgs();
  else if (llvm::isa<scf::YieldOp>(op))
    args = llvm::cast<scf::YieldOp>(op).getResults();
  // Check the legality of arguments using the type converter.
  for (const auto &arg : args) {
    if (isa<xegpu::TensorDescType>(arg.getType()))
      return false;
  }
  return true;
}

struct XeGPUToVCPass : public imex::impl::ConvertXeGPUToVCBase<XeGPUToVCPass> {
  using Base::Base;

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();
    TypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    // Configure the legality of the conversion target for MathToVC patterns.
    configureMathToVCConversionLegality(target);
    configureArithToVCConversionLegality(target);

    target.addLegalDialect<func::FuncDialect, arith::ArithDialect,
                           memref::MemRefDialect, vector::VectorDialect,
                           spirv::SPIRVDialect>();
    target.addIllegalDialect<xegpu::XeGPUDialect>();

    // TODO: can we change it to addDynamicLegalOp?
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();

    // Don't convert other types, e.g., IndexType
    typeConverter.addConversion([&](Type type) { return type; });

    typeConverter.addConversion([&](xegpu::TensorDescType type) -> Type {
      auto scope = type.getMemorySpace();
      auto rank = type.getRank();
      auto i32Type = IntegerType::get(&getContext(), 32);
      auto i64Type = IntegerType::get(&getContext(), 64);

      if (type.isScattered() || rank == 1 || scope == xegpu::MemorySpace::SLM) {
        auto addrTy =
            scope == xegpu::MemorySpace::SLM ? (Type)i32Type : (Type)i64Type;
        auto simd_lanes = type.isScattered() ? type.getShape()[0] : 1;
        return VectorType::get(simd_lanes, addrTy);
      } else if (rank == 2) {
        return VectorType::get(16, i32Type);
      }
      return type;
    });

    typeConverter.addConversion([&](VectorType type) {
      auto elemType = type.getElementType();
      if (!mlir::vector::isLinearizableVector(type))
        return type;
      return VectorType::get(type.getNumElements(), elemType,
                             type.isScalable());
    });

    // a materialization function used by typeConverter to materialize
    // arguments, source and target. If input and output are vectors,
    // shapecast will used, otherwise unrealized_conversion_cast will be used.
    auto materializeCast = [&](mlir::OpBuilder &builder, mlir::Type type,
                               mlir::ValueRange inputs,
                               mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1)
        return nullptr;
      auto valTy = inputs.front().getType();
      if (isa<VectorType>(type) && isa<VectorType>(valTy))
        return builder.create<ShapeCastOp>(loc, type, inputs.front());

      if (isa<xegpu::TensorDescType>(type) || isa<xegpu::TensorDescType>(valTy))
        return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
            .getResult(0);

      return nullptr;
    };
    typeConverter.addArgumentMaterialization(materializeCast);
    typeConverter.addSourceMaterialization(materializeCast);
    typeConverter.addTargetMaterialization(materializeCast);

    // We don't have a type conversion for xegpu.nbarrier, so barrier
    // related ops don't need typeconverters.
    patterns.add<AllocNbarrierPattern, InitNbarrierPattern, NbarrierWaitPattern,
                 NbarrierArrivePattern>(patterns.getContext());

    patterns.add<CreateNdDescPattern, UpdateNDOffsetPattern, CreateDescPattern,
                 UpdateOffsetOpPattern, CompilerHintPattern, DpasPattern>(
        typeConverter, patterns.getContext());

    // Ops to LSC only patterns
    populateAtomicAndFenceLSCPatterns(typeConverter, patterns);

    populateLoadStoreLSCPatterns(typeConverter, patterns);

    populateArithToVCPatterns(typeConverter, patterns);

    populateMathToVCPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      return signalPassFailure();

    // handle unrealized_conversion_cast generated for ops e.g., scf.for and
    // scf.yield
    m.walk([&](mlir::UnrealizedConversionCastOp op) {
      if (op.getInputs().size() != 1 || op.getResults().size() != 1)
        return WalkResult::skip();

      auto input = op.getOperand(0);
      auto output = op.getResult(0);

      // resolve TensorDesc -> vector unrealized type conversion. It is
      // typically inserted for BlockArguments in e.g., SCF for ops.
      // We simply need to update the argument type and replace the
      // users with the argument. For the loop op, we also need to update
      // the type of corresponding tied loop result.
      if (mlir::isa<xegpu::TensorDescType>(input.getType())) {
        auto argument = mlir::dyn_cast<mlir::BlockArgument>(input);
        if (argument) {
          argument.setType(output.getType());
          output.replaceAllUsesWith(argument);
          if (auto loopOp = mlir::dyn_cast<mlir::LoopLikeOpInterface>(
                  argument.getOwner()->getParentOp())) {
            auto result = loopOp.getTiedLoopResult(argument);
            result.setType(output.getType());
          }
        }
      }

      // resolve vector -> TensorDesc unrealized type conversion. It is
      // typically inserted for SCF yield ops.
      if (mlir::isa<xegpu::TensorDescType>(output.getType()))
        output.replaceAllUsesWith(input);

      if (op->use_empty())
        op->erase();
      return WalkResult::advance();
    });
  }
};

std::unique_ptr<OperationPass<gpu::GPUModuleOp>> createConvertXeGPUToVCPass() {
  return std::make_unique<XeGPUToVCPass>();
}

} // namespace imex
