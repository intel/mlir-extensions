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

class InitTileToVCPattern : public OpConversionPattern<InitTileOp> {
public:
  using OpConversionPattern<InitTileOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(InitTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<spirv::ConvertPtrToUOp>(
        op, rewriter.getI64Type(), adaptor.getSource());
    return success();
  }
};
template <typename OpType>
class LoadStore2dToVCPattern : public OpConversionPattern<OpType> {
public:
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ::mlir::VectorType vecType;
    std::string funcName;
    constexpr bool isLoad = std::is_same_v<OpType, Load2DOp>;
    if constexpr (isLoad) {
      vecType = cast<VectorType>(op.getResult().getType());
      funcName = "llvm_genx_lsc_load2d_stateless_";
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
    // predicate(true for now)
    auto pred = createIntConstant(rewriter.getI1Type(), 1);
    // cached(0 for now)
    auto cacheHint = createIntConstant(i8Type, 0);
    unsigned cst = encodeDataum(vecType.getElementType());
    auto dataum = createIntConstant(i8Type, cst);
    auto transpose =
        createIntConstant(i8Type, op->hasAttr("Transpose") ? 2 : 1);
    // number of blocks(1 for now)
    auto nBlks = createIntConstant(i8Type, 1);
    // tile shape is in-memory shape?
    auto tileType = op.getTile().getType();
    auto blockWidth = tileType.getShape()[1];
    auto blockHeight = tileType.getShape()[0];
    auto blockW = createIntConstant(i32Type, blockWidth);
    auto blockH = createIntConstant(i32Type, blockHeight);
    auto transform =
        createIntConstant(i8Type, op->hasAttr("VNNI_AXIS") ? 1 : 0);
    auto base = adaptor.getTile();
    auto initOp = op.getTile().template getDefiningOp<InitTileOp>();
    auto memType = cast<MemRefType>(initOp.getSource().getType());
    unsigned bitWidth = memType.getElementType().getIntOrFloatBitWidth();
    auto surfaceWidth = memType.getShape()[1] * (bitWidth / 8) - 1;
    auto surfaceHeight = memType.getShape()[0] - 1;
    // FIXME: pitch = width for now
    auto surfacePitch = surfaceWidth;
    auto surfaceW = createIntConstant(i32Type, surfaceWidth);
    auto surfaceH = createIntConstant(i32Type, surfaceHeight);
    auto surfaceP = createIntConstant(i32Type, surfacePitch);
    auto offsetX = createIntConstant(i32Type, initOp.getStaticOffsets()[1]);
    auto offsetY = createIntConstant(i32Type, initOp.getStaticOffsets()[0]);
    SmallVector<Value> args{pred,      cacheHint, cacheHint, dataum,
                            transpose, nBlks,     blockW,    blockH,
                            transform, base,      surfaceW,  surfaceH,
                            surfaceP,  offsetX,   offsetY};
    std::string typeStr;
    VectorType newType;
    std::tie(typeStr, newType) = encodeVectorType(rewriter, vecType);
    if constexpr (!isLoad) {
      args.push_back(adaptor.getValue());
    }
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
    unsigned rank = lhsType.getRank();
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
    uint8_t prec1 = encodePrecision(lhsType.getElementType());
    uint8_t prec2 = encodePrecision(rhsType.getElementType());
    unsigned infoVal = (rc << 24) | (sd << 16) | (prec2 << 8) | (prec1);
    auto infoAttr = rewriter.getIntegerAttr(rewriter.getI32Type(), infoVal);
    auto info = rewriter.create<spirv::ConstantOp>(loc, rewriter.getI32Type(),
                                                   infoAttr);
    auto newResultType = encodeVectorType(rewriter, resultType).second;
    SmallVector<Value, 4> args{adaptor.getRhs(), adaptor.getLhs(), info};
    std::string funcName = "llvm_genx_dpas_nosrc0_";
    funcName += encodeVectorType(rewriter, resultType).first;
    funcName += "_";
    funcName += encodeVectorType(rewriter, lhsType).first;
    funcName += "_";
    funcName += encodeVectorType(rewriter, rhsType).first;
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
  patterns.add<InitTileToVCPattern, LoadStore2dToVCPattern<Load2DOp>,
               LoadStore2dToVCPattern<Store2DOp>, DpasToVCPattern>(
      typeConverter, patterns.getContext());
}
