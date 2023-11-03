//===- XeGPUToSPIRV.cpp -  --------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements patterns to convert XeGPU to SPIRV with
/// VC-Intrinsics/JointMatrix
///
//===----------------------------------------------------------------------===//
#include "imex/Conversion/XeGPUToSPIRV/XeGPUToSPIRV.h"
#include "imex/Dialect/XeGPU/IR/XeGPUOps.h"

#include "../PassDetail.h"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace imex;
using namespace imex::xegpu;
using namespace mlir;

namespace {
/// @brief encodeVectorType(xxx, 8x8x2xf16, true) returns ["v64i32", 64xi32]
std::pair<std::string, VectorType>
encodeVectorType(ConversionPatternRewriter &rewriter, VectorType type,
                 bool cast = true) {
  auto elemType = type.getElementType();
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  auto rank = type.getRank();
  auto shape = type.getShape();
  auto size = shape[0] * shape[1];
  if (!cast && bitWidth == 16) {
    assert(shape[rank - 1] == 2);
    size *= 2;
  }
  std::string str;
  switch (size) {
  case 16:
    str += "v16";
    break;
  case 32:
    str += "v32";
    break;
  case 64:
    str += "v64";
    break;
  case 128:
    str += "v128";
    break;
  default:
    assert(0 && "add more support");
    break;
  }
  if (elemType == rewriter.getF32Type())
    str += "f32";
  else if (elemType == rewriter.getF16Type()) {
    if (cast) {
      assert(shape[rank - 1] == 2);
      str += "i32";
      elemType = rewriter.getI32Type();
    } else {
      str += "f16";
    }
  } else
    assert(0 && "add more support");
  auto newType = VectorType::get(size, elemType);
  return std::make_pair(str, newType);
}
unsigned encodeDataum(Type type) {
  switch (type.getIntOrFloatBitWidth()) {
  case 8:
    return 1;
  case 16:
    return 2;
  case 32:
    return 3;
  case 64:
    return 4;
  default:
    assert(0 && "add more support");
    return 0;
  }
}

void lookupOrInsertIntrinsic(ConversionPatternRewriter &rewriter, Operation *op,
                             std::string name, FunctionType funcType) {
  auto funcAttr = StringAttr::get(rewriter.getContext(), name);
  Operation *found = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
  if (!found) {
    OpBuilder::InsertionGuard guard(rewriter);
    auto kernel = op->getParentOfType<spirv::FuncOp>();
    rewriter.setInsertionPoint(kernel);
    auto func = rewriter.create<spirv::FuncOp>(kernel.getLoc(), name, funcType);
    auto linkageTypeAttr =
        rewriter.getAttr<spirv::LinkageTypeAttr>(spirv::LinkageType::Import);
    std::replace(name.begin(), name.end(), '_', '.');
    auto linkage = spirv::LinkageAttributesAttr::get(rewriter.getContext(),
                                                     name, linkageTypeAttr);
    func.setLinkageAttributesAttr(linkage);
    func->setAttr("VectorComputeFunctionINTEL", rewriter.getUnitAttr());
  }
}

class CreateNdDescToVCPattern : public OpConversionPattern<CreateNdDescOp> {
public:
  using OpConversionPattern<CreateNdDescOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CreateNdDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<spirv::ConvertPtrToUOp>(
        op, rewriter.getI64Type(), adaptor.getSource());
    return success();
  }
};
template <typename OpType>
class LoadStorePrefetchNdToLsc : public OpConversionPattern<OpType> {
public:
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op.getTensorDesc().getType().getShape().size() == 2 &&
           "only support 2d load/store/prefetch for now");
    auto loc = op.getLoc();
    ::mlir::VectorType vecType;
    std::string funcName;
    constexpr bool isLoad = std::is_same_v<OpType, LoadNDOp>;
    constexpr bool isPrefetch = std::is_same_v<OpType, PrefetchNDOp>;
    if constexpr (isLoad) {
      vecType = cast<VectorType>(op.getResult().getType());
      funcName = "llvm_genx_lsc_load2d_stateless_";
    } else if constexpr (isPrefetch) {
      vecType = VectorType::get({8, 16}, rewriter.getF32Type());
      funcName = "llvm_genx_lsc_prefetch2d_stateless_i1_i64";
    } else {
      vecType = cast<VectorType>(op.getValue().getType());
      funcName = "llvm_genx_lsc_store2d_stateless_i1_i64_";
    }
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<spirv::ConstantOp>(loc, type, attr);
    };
    auto i8Type = rewriter.getI8Type();
    auto i32Type = rewriter.getI32Type();
    auto vnni = false;
    auto transpose = false;
    if constexpr (isLoad) {
      auto vnniValue = op.getVnniAxis();
      vnni = vnniValue.has_value() && vnniValue.value() == 0 ? true : false;
      auto transposeValue = op.getTranspose();
      transpose = transposeValue.has_value() && transposeValue.value()[0] == 1
                      ? true
                      : false;
    }
    auto l1hint = op.getL1Hint();
    // auto l2hint = op.getL2Hint();
    auto l3hint = op.getL3Hint();

    // predicate(true for now)
    auto pred = createIntConstant(rewriter.getI1Type(), 1);
    auto l1CacheHint =
        createIntConstant(i8Type, l1hint.has_value() ? (int)l1hint.value() : 0);
    auto l3CacheHint =
        createIntConstant(i8Type, l3hint.has_value() ? (int)l3hint.value() : 0);
    unsigned cst = encodeDataum(vecType.getElementType());
    auto dataum = createIntConstant(i8Type, cst);
    auto trans = createIntConstant(i8Type, transpose ? 2 : 1);
    // number of blocks(1 for now)
    auto nBlks = createIntConstant(i8Type, 1);
    auto tensorType = op.getTensorDesc().getType();
    auto blockWidth = tensorType.getShape()[1];
    auto blockHeight = tensorType.getShape()[0];
    auto blockW = createIntConstant(i32Type, blockWidth);
    auto blockH = createIntConstant(i32Type, blockHeight);
    auto transform = createIntConstant(i8Type, vnni ? 1 : 0);
    auto base = adaptor.getTensorDesc();
    // static memref for now
    auto createDescOp =
        op.getTensorDesc().template getDefiningOp<CreateNdDescOp>();
    auto memType = cast<MemRefType>(createDescOp.getSource().getType());
    unsigned bitWidth = memType.getElementType().getIntOrFloatBitWidth();
    auto surfaceWidth = memType.getShape()[1] * (bitWidth / 8) - 1;
    auto surfaceHeight = memType.getShape()[0] - 1;
    // pitch = width for now
    auto surfacePitch = surfaceWidth;
    auto surfaceW = createIntConstant(i32Type, surfaceWidth);
    auto surfaceH = createIntConstant(i32Type, surfaceHeight);
    auto surfaceP = createIntConstant(i32Type, surfacePitch);
    auto createOffset = [&](unsigned idx) -> Value {
      Value val;
      if (ShapedType::isDynamic(createDescOp.getStaticOffsets()[idx])) {
        val = createDescOp.getOffsets()[idx];
        val = rewriter.create<arith::TruncIOp>(loc, i32Type, val);
      } else {
        val = createIntConstant(i32Type, createDescOp.getStaticOffsets()[idx]);
      }
      return val;
    };
    auto offsetX = createOffset(1);
    auto offsetY = createOffset(0);

    SmallVector<Value> args{pred,      l1CacheHint, l3CacheHint, dataum,
                            trans,     nBlks,       blockW,      blockH,
                            transform, base,        surfaceW,    surfaceH,
                            surfaceP,  offsetX,     offsetY};
    std::string typeStr;
    VectorType newType;
    std::tie(typeStr, newType) = encodeVectorType(rewriter, vecType);
    if constexpr (!isLoad && !isPrefetch) {
      args.push_back(adaptor.getValue());
    }
    if constexpr (!isPrefetch)
      funcName += typeStr;
    if constexpr (isLoad) {
      funcName += "_i1_i64";
      auto retType = newType;
      auto funcType =
          rewriter.getFunctionType(ValueRange(args).getTypes(), retType);
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
      auto funcOp =
          rewriter.create<spirv::FunctionCallOp>(loc, retType, funcName, args);
      rewriter.replaceOp(op, funcOp);
    } else {
      auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
      auto funcOp = rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(),
                                                           funcName, args);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

template <typename OpType>
class LoadStorePrefetchNdToRawSend : public OpConversionPattern<OpType> {
public:
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op.getTensorDesc().getType().getShape().size() == 2 &&
           "only support 2d load/store/prefetch for now");
    auto loc = op->getLoc();
    constexpr bool isLoad = std::is_same_v<OpType, LoadNDOp>;
    constexpr bool isPrefetch = std::is_same_v<OpType, PrefetchNDOp>;
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<spirv::ConstantOp>(loc, type, attr);
    };

    /// collect common info
    auto i1Type = rewriter.getI1Type();
    auto i8Type = rewriter.getI8Type();
    auto i16Type = rewriter.getI16Type();
    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();
    auto vnni = false;
    auto transpose = false;
    if constexpr (isLoad) {
      auto vnniValue = op.getVnniAxis();
      vnni = vnniValue.has_value() && vnniValue.value() == 0 ? true : false;
      auto transposeValue = op.getTranspose();
      transpose = transposeValue.has_value() && transposeValue.value()[0] == 1
                      ? true
                      : false;
    }
    auto l1hint = op.getL1Hint();
    // auto l2hint = op.getL2Hint();
    auto l3hint = op.getL3Hint();
    auto tileType = op.getTensorDesc().getType();
    auto blockWidth = tileType.getShape()[1];
    auto blockHeight = tileType.getShape()[0];
    auto elmType = tileType.getElementType();
    auto base = adaptor.getTensorDesc();
    VectorType newType = VectorType::get(1, i32Type);
    std::string funcName;
    if constexpr (isPrefetch) {
      funcName = "llvm_genx_raw_send2_noresult_i1_v8i32";
    } else {
      VectorType vecType;
      if constexpr (isLoad) {
        vecType = cast<VectorType>(op.getResult().getType());
        funcName = "llvm_genx_raw_send2_";
      } else {
        vecType = cast<VectorType>(op.getValue().getType());
        funcName = "llvm_genx_raw_sends2_noresult_i1_v8i32_";
      }
      std::string typeStr;
      std::tie(typeStr, newType) = encodeVectorType(rewriter, vecType);
      funcName += typeStr;
    }
    auto createDescOp =
        op.getTensorDesc().template getDefiningOp<CreateNdDescOp>();
    // fixme: support memref for now
    auto memType = cast<MemRefType>(createDescOp.getSource().getType());
    unsigned bitWidth = memType.getElementType().getIntOrFloatBitWidth();
    auto surfaceWidth = memType.getShape()[1] * (bitWidth / 8) - 1;
    auto surfaceHeight = memType.getShape()[0] - 1;
    // fixme: pitch = width for now
    auto surfacePitch = surfaceWidth;
    auto surfaceW = createIntConstant(i32Type, surfaceWidth);
    auto surfaceH = createIntConstant(i32Type, surfaceHeight);
    auto surfaceP = createIntConstant(i32Type, surfacePitch);
    auto createOffset = [&](unsigned idx) -> Value {
      Value val;
      if (ShapedType::isDynamic(createDescOp.getStaticOffsets()[idx])) {
        val = createDescOp.getOffsets()[idx];
        val = rewriter.create<arith::TruncIOp>(loc, i32Type, val);
      } else {
        val = createIntConstant(i32Type, createDescOp.getStaticOffsets()[idx]);
      }
      return val;
    };
    auto offsetX = createOffset(1);
    auto offsetY = createOffset(0);
    int cacheHint = 1;
    if constexpr (isLoad || isPrefetch) {
      auto l1CacheValue =
          l1hint.has_value() ? l1hint.value() : xegpu::CacheReadHint::UNCACHED;
      auto l3CacheValue =
          l3hint.has_value() ? l3hint.value() : xegpu::CacheReadHint::UNCACHED;
      if (l1CacheValue == xegpu::CacheReadHint::UNCACHED) {
        if (l3CacheValue == xegpu::CacheReadHint::UNCACHED)
          cacheHint = 1;
        else if (l3CacheValue == xegpu::CacheReadHint::CACHED)
          cacheHint = 2;
      } else if (l1CacheValue == xegpu::CacheReadHint::CACHED) {
        if (l3CacheValue == xegpu::CacheReadHint::UNCACHED)
          cacheHint = 3;
        else if (l3CacheValue == xegpu::CacheReadHint::CACHED)
          cacheHint = 4;
      } else if (l1CacheValue == xegpu::CacheReadHint::STREAMING) {
        if (l3CacheValue == xegpu::CacheReadHint::UNCACHED)
          cacheHint = 5;
        else if (l3CacheValue == xegpu::CacheReadHint::CACHED)
          cacheHint = 6;
      } else if (l1CacheValue == xegpu::CacheReadHint::READ_INVALIDATE) {
        if (l3CacheValue == xegpu::CacheReadHint::CACHED)
          cacheHint = 7;
      }
    } else {
      auto l1CacheValue =
          l1hint.has_value() ? l1hint.value() : xegpu::CacheWriteHint::UNCACHED;
      auto l3CacheValue =
          l3hint.has_value() ? l3hint.value() : xegpu::CacheWriteHint::UNCACHED;
      if (l1CacheValue == xegpu::CacheWriteHint::UNCACHED) {
        if (l3CacheValue == xegpu::CacheWriteHint::UNCACHED)
          cacheHint = 1;
        else if (l3CacheValue == xegpu::CacheWriteHint::WRITE_BACK)
          cacheHint = 2;
      } else if (l1CacheValue == xegpu::CacheWriteHint::WRITE_THROUGH) {
        if (l3CacheValue == xegpu::CacheWriteHint::UNCACHED)
          cacheHint = 3;
        else if (l3CacheValue == xegpu::CacheWriteHint::WRITE_BACK)
          cacheHint = 4;
      } else if (l1CacheValue == xegpu::CacheWriteHint::STREAMING) {
        if (l3CacheValue == xegpu::CacheWriteHint::UNCACHED)
          cacheHint = 5;
        else if (l3CacheValue == xegpu::CacheWriteHint::WRITE_BACK)
          cacheHint = 6;
      } else if (l1CacheValue == xegpu::CacheWriteHint::WRITE_BACK) {
        if (l3CacheValue == xegpu::CacheWriteHint::WRITE_BACK)
          cacheHint = 7;
      }
    }

    /// fill in parameters for raw.send
    // bit[1:0] EOT,sendc
    auto modifier = createIntConstant(i8Type, 0);
    auto execSize = createIntConstant(i8Type, 0);
    auto pred = createIntConstant(i1Type, 1);
    auto numSrc1 = createIntConstant(i8Type, 1);
    unsigned numDstVal = newType.getNumElements() / 16;
    auto numDst = createIntConstant(i8Type, numDstVal);
    // 15 for ugm
    auto sfid = createIntConstant(i8Type, 15);
    auto extMsg = createIntConstant(i32Type, 0);
    // message descriptor
    // https://gfxspecs.intel.com/Predator/Home/Index/53680
    uint32_t rawSendMsg = (isLoad || isPrefetch) ? 3 : 7;
    rawSendMsg |= (vnni ? 1 : 0) << 7;
    rawSendMsg |= (encodeDataum(elmType) - 1) << 9;
    rawSendMsg |= (transpose ? 1 : 0) << 15;
    rawSendMsg |= cacheHint << 17;
    rawSendMsg |= (isLoad ? numDstVal : 0) << 20;
    rawSendMsg |= 1 << 25;
    auto msg = createIntConstant(i32Type, rawSendMsg);
    // payload
    auto v8i32 = VectorType::get(8, i32Type);
    auto v4i64 = VectorType::get(4, i64Type);
    Value payLoad = rewriter.create<spirv::UndefOp>(loc, v4i64);
    auto idx0 = createIntConstant(i32Type, 0);
    auto idx2 = createIntConstant(i32Type, 2);
    auto idx3 = createIntConstant(i32Type, 3);
    auto idx4 = createIntConstant(i32Type, 4);
    auto idx5 = createIntConstant(i32Type, 5);
    auto idx6 = createIntConstant(i32Type, 6);
    auto idx7 = createIntConstant(i32Type, 7);
    payLoad =
        rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad, base, idx0);
    payLoad = rewriter.create<spirv::BitcastOp>(loc, v8i32, payLoad);
    payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                            surfaceW, idx2);
    payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                            surfaceH, idx3);
    payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                            surfaceP, idx4);
    payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                            offsetX, idx5);
    payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                            offsetY, idx6);
    unsigned blockVal = ((blockHeight - 1) << 8) | (blockWidth - 1);
    auto blockInfo = createIntConstant(i32Type, blockVal);
    payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                            blockInfo, idx7);
    SmallVector<Value> args{modifier, execSize, pred, numSrc1, numDst,
                            sfid,     extMsg,   msg,  payLoad};
    if constexpr (isLoad) {
      funcName += "_i1_v8i32";
      auto old = rewriter.create<spirv::UndefOp>(loc, newType);
      args.push_back(old);
      auto retType = newType;
      auto funcType =
          rewriter.getFunctionType(ValueRange(args).getTypes(), retType);
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
      auto funcOp =
          rewriter.create<spirv::FunctionCallOp>(loc, retType, funcName, args);
      rewriter.replaceOp(op, funcOp);
    } else {
      if constexpr (isPrefetch)
        args.erase(args.begin() + 4);
      else
        args.push_back(adaptor.getValue());
      auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
      rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(), funcName, args);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

class DpasToVCPattern : public OpConversionPattern<DpasOp> {
public:
  using OpConversionPattern<DpasOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(DpasOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhsType = op.getLhs().getType().cast<VectorType>();
    auto rhsType = op.getRhs().getType().cast<VectorType>();
    auto resultType = op.getResultType().cast<VectorType>();
    uint8_t rc = lhsType.getShape()[0];
    uint8_t sd = lhsType.getShape()[1];
    // refer to IGC/visa/Common_ISA_util.cpp#87
    auto encodePrecision = [&](Type type) -> uint8_t {
      if (type == rewriter.getBF16Type())
        return 9;
      else if (type == rewriter.getF16Type())
        return 10;
      else if (type == rewriter.getTF32Type())
        return 12;
      else {
        assert(0 && "add more support");
        return 0;
      }
    };
    uint8_t prec1 = encodePrecision(rhsType.getElementType());
    uint8_t prec2 = encodePrecision(lhsType.getElementType());
    unsigned infoVal = (rc << 24) | (sd << 16) | (prec2 << 8) | (prec1);
    auto infoAttr = rewriter.getIntegerAttr(rewriter.getI32Type(), infoVal);
    auto info = rewriter.create<spirv::ConstantOp>(loc, rewriter.getI32Type(),
                                                   infoAttr);
    auto newResultType = encodeVectorType(rewriter, resultType).second;
    SmallVector<Value, 4> args{adaptor.getRhs(), adaptor.getLhs(), info};
    std::string funcName = "llvm_genx_dpas_nosrc0_";
    if (op.getAcc()) {
      funcName = "llvm_genx_dpas2_";
      auto i32Type = rewriter.getI32Type();
      auto createIntConstant = [&](Type type, unsigned value) {
        auto attr = rewriter.getIntegerAttr(type, value);
        return rewriter.create<spirv::ConstantOp>(loc, type, attr);
      };
      auto prec1Arg = createIntConstant(i32Type, prec1);
      auto prec2Arg = createIntConstant(i32Type, prec2);
      auto sdArg = createIntConstant(i32Type, sd);
      auto rcArg = createIntConstant(i32Type, rc);
      auto signless = createIntConstant(i32Type, 0);
      args.assign({adaptor.getAcc(), adaptor.getRhs(), adaptor.getLhs(),
                   prec1Arg, prec2Arg, sdArg, rcArg, signless, signless});
    }
    funcName += encodeVectorType(rewriter, resultType).first;
    funcName += "_";
    funcName += encodeVectorType(rewriter, rhsType).first;
    funcName += "_";
    funcName += encodeVectorType(rewriter, lhsType).first;
    auto funcType =
        rewriter.getFunctionType(ValueRange(args).getTypes(), newResultType);
    Operation *opPtr = op;
    lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
    auto funcOp = rewriter.create<spirv::FunctionCallOp>(loc, newResultType,
                                                         funcName, args);
    rewriter.replaceOp(op, funcOp);
    return success();
  }
};
} // namespace

void imex::populateXeGPUToVCIntrinsicsPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<CreateNdDescToVCPattern, DpasToVCPattern>(typeConverter,
                                                         patterns.getContext());
  if (getenv("IMEX_NOT_PREFER_RAWSEND"))
    patterns.add<LoadStorePrefetchNdToLsc<LoadNDOp>,
                 LoadStorePrefetchNdToLsc<StoreNDOp>,
                 LoadStorePrefetchNdToLsc<PrefetchNDOp>>(typeConverter,
                                                         patterns.getContext());
  else
    patterns.add<LoadStorePrefetchNdToRawSend<LoadNDOp>,
                 LoadStorePrefetchNdToRawSend<StoreNDOp>,
                 LoadStorePrefetchNdToRawSend<PrefetchNDOp>>(
        typeConverter, patterns.getContext());
}
