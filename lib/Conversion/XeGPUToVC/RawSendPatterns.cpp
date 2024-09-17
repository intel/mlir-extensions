//===-- RawSendPatterns.cpp - XeGPU to VC Lowering pass ---------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements patterns to lower load/store to RawSend messages
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/XeGPUToVC/XeGPUToVC.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"

#include "Utils.h"

using namespace mlir;
using mlir::xegpu::DpasOp;
using mlir::xegpu::LoadGatherOp;
using mlir::xegpu::LoadNdOp;
using mlir::xegpu::NbarrierArriveOp;
using mlir::xegpu::PrefetchNdOp;
using mlir::xegpu::PrefetchOp;
using mlir::xegpu::StoreNdOp;
using mlir::xegpu::StoreScatterOp;

namespace imex {
namespace RawSend {

template <typename OpType>
class LoadStorePrefetchNdToRawSendPattern : public OpConversionPattern<OpType> {
public:
  using OpConversionPattern<OpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto tileType = op.getTensorDesc().getType();
    auto rank = tileType.getRank();
    assert(rank <= 2 && "only support 1d/2d load/store/prefetch for now");
    auto loc = op.getLoc();
    constexpr bool isLoad = std::is_same_v<OpType, LoadNdOp>;
    constexpr bool isPrefetch = std::is_same_v<OpType, PrefetchNdOp>;
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<arith::ConstantOp>(loc, type, attr);
    };

    /// collect common info
    auto i1Type = rewriter.getI1Type();
    auto i8Type = rewriter.getI8Type();
    auto i32Type = rewriter.getI32Type();
    auto vnni = false;
    auto transpose = false;
    if constexpr (isLoad) {
      vnni = op.getPacked().value_or(false);
      auto transposeValue = op.getTranspose();
      transpose = transposeValue.has_value() && transposeValue.value()[0] == 1
                      ? true
                      : false;
    }
    auto elmType = tileType.getElementType();
    VectorType newType = VectorType::get(1, i32Type);
    std::string funcName;
    if constexpr (isPrefetch) {
      funcName = "llvm.genx.raw.send2.noresult.i1.v16i32";
    } else {
      VectorType vecType;
      if constexpr (isLoad) {
        vecType = cast<VectorType>(op.getResult().getType());
        funcName = "llvm.genx.raw.send2.";
      } else {
        vecType = cast<VectorType>(op.getValue().getType());
        funcName = "llvm.genx.raw.sends2.noresult.i1.v16i32.";
      }
      std::string typeStr;
      std::tie(typeStr, newType) =
          encodeVectorType(rewriter, vecType, rank == 1);
      funcName += typeStr;
    }
    unsigned cacheHint = encodeCacheHint(op);

    /// fill in parameters for raw.send
    // bit[1:0] EOT,sendc
    auto modifier = createIntConstant(i8Type, 0);
    auto execSize = createIntConstant(i8Type, 0);
    auto pred = createIntConstant(i1Type, 1);
    auto numSrc1 = createIntConstant(i8Type, 1);
    // numDstVal: "Dest Length" is a value with the unit of a register
    unsigned numDstVal =
        (newType.getNumElements() + 16 - 1) / 16; // TODO: clarify 16
    if (rank == 1) {
      numDstVal *= 2;
    }
    // numDstVal is represented using only 5 bits in the raw_send message.
    // So, value 32 is represented as 31 and data port hardware derives the
    // correct destination length based on message parameters.
    if (numDstVal == 32)
      numDstVal = 31;
    auto numDst = createIntConstant(i8Type, numDstVal);
    // 15 for ugm
    auto sfid = createIntConstant(i8Type, 15);
    auto extMsg = createIntConstant(i32Type, 0);
    auto dataSize2D = (encodeDataum(elmType) - 1);
    auto payLoad = adaptor.getTensorDesc();

    // TODO: currently limit transposeBitWidth to 32, it is
    // an architecture feature, and 32 works on PVC.
    // To support other bits, we cannot hardcode
    // with i32Type, and need to generalize the logic.
    auto loadOp = llvm::dyn_cast<LoadNdOp>(op.getOperation());
    if (loadOp && transpose && loadOp.getTransposeBitWidth() == 32) {
      // in raw_send msg set vnni effect to false and update data size of
      // payload item to 32 bits
      vnni = false;
      dataSize2D = (encodeDataum(i32Type) - 1);

      // we also need to update the payload (address descriptor) to reflect that
      // now we are viewing the memref and tile in 32 bit data type not original
      // type. This requires updaing the offsetX (row dim offset) and block
      // width (divide the value by vnni factor).
      auto vnniFactor = 32 / elmType.getIntOrFloatBitWidth();
      auto getLog2OfVnniFactor = [&]() -> unsigned {
        if (vnniFactor == 2)
          return 1;
        else if (vnniFactor == 4)
          return 2;
        else
          assert(false && "invalid vnni Factor!");
      };
      auto oldOffsetX = rewriter.create<vector::ExtractOp>(loc, payLoad, 5);
      // do an aritmetic right shift instead of divide.
      auto newOffsetX = rewriter.create<arith::ShRUIOp>(
          loc, oldOffsetX, createIntConstant(i32Type, getLog2OfVnniFactor()));
      payLoad = rewriter.create<vector::InsertOp>(loc, newOffsetX, payLoad, 5);
      int array_length = op.getTensorDescType().getArrayLength();
      unsigned blockVal = (array_length - 1) << 16;
      auto blockWidth = tileType.getShape()[1];
      auto blockHeight = tileType.getShape()[0];
      auto newBlockWidth = blockWidth / vnniFactor;
      blockVal |= ((blockHeight - 1) << 8) | (newBlockWidth - 1);
      auto blockInfo = createIntConstant(i32Type, blockVal);
      payLoad = rewriter.create<vector::InsertOp>(loc, blockInfo, payLoad, 7);
    }
    // message descriptor
    uint32_t rawSendMsg = 0;
    if (rank == 2) {
      rawSendMsg |= (isLoad || isPrefetch) ? 3 : 7;
      rawSendMsg |= (vnni ? 1 : 0) << 7;
      rawSendMsg |= dataSize2D << 9;
      rawSendMsg |= (transpose ? 1 : 0) << 15;
      rawSendMsg |= cacheHint << 17;
      rawSendMsg |= (isLoad ? numDstVal : 0) << 20;
      rawSendMsg |= 1 << 25;
    } else {
      // rank == 1
      rawSendMsg |= (isLoad || isPrefetch) ? 0 : 4;
      rawSendMsg |= 3 << 7;
      rawSendMsg |= 3 << 9;
      rawSendMsg |= int(log2(newType.getNumElements()) + 1) << 12;
      rawSendMsg |= 1 << 15;
      rawSendMsg |= cacheHint << 17;
      rawSendMsg |= (isLoad ? 2 * numDstVal : 0) << 20;
      rawSendMsg |= 1 << 25;
    }
    auto msg = createIntConstant(i32Type, rawSendMsg);

    SmallVector<Value> args{modifier, execSize, pred, numSrc1, numDst,
                            sfid,     extMsg,   msg,  payLoad};
    if constexpr (isLoad) {
      funcName += ".i1.v16i32";
      auto elementTy = newType.getElementType();
      Attribute initValueAttr;
      if (isa<FloatType>(elementTy))
        initValueAttr = FloatAttr::get(elementTy, 0.0);
      else
        initValueAttr = IntegerAttr::get(elementTy, 0);
      Value old = rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(newType, initValueAttr));

      args.push_back(old);
      auto retType = newType;
      auto funcOp = createFuncCall(rewriter, loc, funcName, TypeRange{retType},
                                   args, false);
      if (rank == 2) {
        // Intrinsic accepts and returns i32 type, but we want to return a
        // vector of the original element type
        auto loadResultInOrigType = encodeVectorTypeTo(newType, elmType);
        if (loadResultInOrigType != funcOp->getResult(0).getType()) {
          auto cast = rewriter.create<vector::BitCastOp>(
              loc, loadResultInOrigType, funcOp->getResult(0));
          rewriter.replaceOp(op, cast);
        } else {
          rewriter.replaceOp(op, funcOp);
        }
      } else {
        auto cast = rewriter.create<vector::BitCastOp>(loc, op.getType(),
                                                       funcOp->getResult(0));
        rewriter.replaceOp(op, cast);
      }
    } else {
      if constexpr (isPrefetch)
        args.erase(args.begin() + 4);
      else {
        if (rank == 2) {
          args.push_back(adaptor.getValue());
        } else if (rank == 1) {
          auto cast = rewriter.create<vector::BitCastOp>(loc, newType,
                                                         adaptor.getValue());
          args.push_back(cast);
        }
      }
      createFuncCall(rewriter, loc, funcName, TypeRange{}, args, false);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

template <typename OpType>
class GatherScatterToRawSend : public OpConversionPattern<OpType> {
public:
  using OpConversionPattern<OpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto tileType = op.getTensorDesc().getType();
    auto rank = tileType.getRank();
    assert(rank <= 2 && "only support 1d/2d for now");
    auto loc = op->getLoc();
    constexpr bool isLoad = std::is_same_v<OpType, LoadGatherOp>;
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<arith::ConstantOp>(loc, type, attr);
    };

    /// collect common info
    auto i8Type = rewriter.getI8Type();
    auto i32Type = rewriter.getI32Type();
    std::string funcName;
    VectorType vecType;
    std::string_view payloadType{"v16i64"};
    std::string_view maskType{"v16i1"};
    if constexpr (isLoad) {
      vecType = cast<VectorType>(op.getResult().getType());
      funcName = "llvm.genx.raw.send2.";
    } else {
      vecType = cast<VectorType>(op.getValue().getType());
      funcName = llvm::formatv("llvm.genx.raw.sends2.noresult.{0}.{1}.",
                               maskType, payloadType)
                     .str();
    }
    std::string typeStr;
    VectorType newType = VectorType::get(1, i32Type);
    std::tie(typeStr, newType) = encodeVectorType(rewriter, vecType);
    funcName += typeStr;
    unsigned cacheHint = encodeCacheHint(op);

    /// fill in parameters for raw.send
    // bit[1:0] EOT,sendc
    auto modifier = createIntConstant(i8Type, 0);
    auto execSize = createIntConstant(i8Type, 4);
    auto pred = adaptor.getMask();
    auto numSrc1 = createIntConstant(i8Type, 2);
    unsigned numDstVal = (newType.getNumElements() + 16 - 1) / 16;
    auto numDst = createIntConstant(i8Type, numDstVal);
    // 15 for ugm
    auto sfid = createIntConstant(i8Type, 15);
    auto extMsg = createIntConstant(i32Type, 0);
    auto vecSize = 0;
    if (numDstVal <= 4) {
      vecSize = numDstVal - 1;
    } else {
      vecSize = log2(numDstVal) + 1;
    }
    // message descriptor
    uint32_t rawSendMsg = 0;
    rawSendMsg |= (isLoad) ? 0 : 4;
    rawSendMsg |= 3 << 7; // A64
    rawSendMsg |= 2 << 9; // D32
    rawSendMsg |= vecSize << 12;
    rawSendMsg |= cacheHint << 17;
    rawSendMsg |= (isLoad ? numDstVal : 0) << 20;
    rawSendMsg |= 2 << 25;
    auto msg = createIntConstant(i32Type, rawSendMsg);
    // payload
    auto payLoad = adaptor.getTensorDesc();
    SmallVector<Value> args{modifier, execSize, pred, numSrc1, numDst,
                            sfid,     extMsg,   msg,  payLoad};
    if constexpr (isLoad) {
      funcName += llvm::formatv(".{0}.{1}", maskType, payloadType).str();
      auto elementTy = newType.getElementType();
      Attribute initValueAttr;
      if (isa<FloatType>(elementTy))
        initValueAttr = FloatAttr::get(elementTy, 0.0);
      else
        initValueAttr = IntegerAttr::get(elementTy, 0);
      Value old = rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(newType, initValueAttr));
      args.push_back(old);
      auto retType = newType;
      auto funcOp = createFuncCall(rewriter, loc, funcName, TypeRange{retType},
                                   args, false);
      auto *converter = this->getTypeConverter();
      auto castTy = converter->convertType(op.getType());
      auto cast =
          rewriter.create<vector::BitCastOp>(loc, castTy, funcOp->getResult(0));
      rewriter.replaceOp(op, cast);
    } else {
      Value data = adaptor.getValue();
      if (data.getType() != newType) {
        data = rewriter.create<vector::BitCastOp>(loc, newType, data);
      }
      args.push_back(data);
      createFuncCall(rewriter, loc, funcName, TypeRange{}, args, false);
      rewriter.eraseOp(op);
    }
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

    std::string funcName = "llvm.genx.raw.send2.noresult.i1.v8i32";

    // desc format
    Value modifier = i8_val(0);
    Value exec_size = i8_val(0);
    Value predicate = i1_val(1);
    Value numsrc1 = i8_val(1); // register nums of payload
    Value sfid = i8_val(3);
    Value etDesc = i32_val(0);
    Value msg_desc = i32_val(0x2000004);

    SmallVector<Value> args{modifier, exec_size, predicate, numsrc1,
                            sfid,     etDesc,    msg_desc,  payload};

    createFuncCall(rewriter, loc, funcName, TypeRange{}, args, false);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace RawSend

void populateNbarrierArriveRawSendPatterns(TypeConverter &converter,
                                           RewritePatternSet &patterns) {
  patterns.add<RawSend::NbarrierArrivePattern>(patterns.getContext());
}

void populateLoadStoreRawSendPatterns(TypeConverter &converter,
                                      RewritePatternSet &patterns) {

  patterns.add<RawSend::LoadStorePrefetchNdToRawSendPattern<LoadNdOp>,
               RawSend::LoadStorePrefetchNdToRawSendPattern<StoreNdOp>,
               RawSend::LoadStorePrefetchNdToRawSendPattern<PrefetchNdOp>>(
      patterns.getContext());

  patterns.add<RawSend::GatherScatterToRawSend<LoadGatherOp>,
               RawSend::GatherScatterToRawSend<StoreScatterOp>>(
      converter, patterns.getContext());
}

} // namespace imex
