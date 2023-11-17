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
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace imex;
using namespace imex::xegpu;
using namespace mlir;

namespace {
/// @brief encodeVectorType(xxx, 8x8x2xf16, false) returns ["v64i32", 64xi32]
std::pair<std::string, VectorType>
encodeVectorType(ConversionPatternRewriter &rewriter, VectorType type,
                 bool use64bitData = false, bool enforceInteger = false) {
  auto elemType = type.getElementType();
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  int size = type.getNumElements() * bitWidth / 32;
  if (use64bitData) {
    size /= 2;
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
  if (use64bitData) {
    str += "i64";
    elemType = rewriter.getI64Type();
  } else if (enforceInteger) {
    str += "i32";
    elemType = rewriter.getI32Type();
  } else if (elemType == rewriter.getF32Type())
    str += "f32";
  else if (elemType == rewriter.getF16Type()) {
    str += "i32";
    elemType = rewriter.getI32Type();
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

template <typename OpType> unsigned encodeCacheHint(OpType op) {
  auto l1hint = op.getL1Hint();
  // auto l2hint = op.getL2Hint();
  auto l3hint = op.getL3Hint();
  constexpr bool isWrite = std::is_same_v<OpType, StoreNDOp> ||
                           std::is_same_v<OpType, StoreScatterOp>;
  unsigned cacheHint = 1;
  if constexpr (!isWrite) {
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
  return cacheHint;
}

unsigned encodeOpcode(xegpu::AtomicRMWKind kind) {
  unsigned encode = 0;
  switch (kind) {
  case xegpu::AtomicRMWKind::addf:
    encode = 19;
    break;
  case xegpu::AtomicRMWKind::addi:
    encode = 12;
    break;
  case xegpu::AtomicRMWKind::assign:
    encode = 10;
    break;
  case xegpu::AtomicRMWKind::maxf:
    encode = 22;
    break;
  case xegpu::AtomicRMWKind::maxs:
    encode = 15;
    break;
  case xegpu::AtomicRMWKind::maxu:
    encode = 17;
    break;
  case xegpu::AtomicRMWKind::minf:
    encode = 21;
    break;
  case xegpu::AtomicRMWKind::mins:
    encode = 14;
    break;
  case xegpu::AtomicRMWKind::minu:
    encode = 16;
    break;
  // case xegpu::AtomicRMWKind::mulf:
  // case xegpu::AtomicRMWKind::muli:
  case xegpu::AtomicRMWKind::ori:
    encode = 25;
    break;
  case xegpu::AtomicRMWKind::andi:
    encode = 24;
    break;
  default:
    assert(0 && "to be supported");
    break;
  }
  return encode;
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
    auto nameAttr = StringAttr::get(rewriter.getContext(), name);
    auto linkage = spirv::LinkageAttributesAttr::get(rewriter.getContext(),
                                                     nameAttr, linkageTypeAttr);
    func.setLinkageAttributesAttr(linkage);
    func->setAttr("VectorComputeFunctionINTEL", rewriter.getUnitAttr());
  }
}

/// @brief
/// convert the tensor descriptor to [2xi64] which is of the format
/// -> [base pointer: i64, offsetX: i32, offsetY: i32] for 2D tensor desc
/// -> [base pointer: i64, unused] for 1D and scattered tensor desc
class CreateNdDescToVCPattern : public OpConversionPattern<CreateNdDescOp> {
public:
  using OpConversionPattern<CreateNdDescOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CreateNdDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();
    // payload
    auto v4i32 = VectorType::get(4, i32Type);
    auto v2i64 = VectorType::get(2, i64Type);
    Value payLoad = rewriter.create<spirv::UndefOp>(loc, v2i64);
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<spirv::ConstantOp>(loc, type, attr);
    };
    auto base = rewriter.create<spirv::ConvertPtrToUOp>(loc, i64Type,
                                                        adaptor.getSource());
    auto idx0 = createIntConstant(i32Type, 0);
    payLoad =
        rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad, base, idx0);
    auto tileType = op.getTensorDesc().getType();
    auto rank = tileType.getRank();
    if (rank == 2) {
      payLoad = rewriter.create<spirv::BitcastOp>(loc, v4i32, payLoad);
      auto createOffset = [&](unsigned idx) -> Value {
        Value val;
        if (ShapedType::isDynamic(op.getStaticOffsets()[idx])) {
          val = op.getOffsets()[idx];
          val = rewriter.create<arith::TruncIOp>(loc, i32Type, val);
        } else {
          val = createIntConstant(i32Type, op.getStaticOffsets()[idx]);
        }
        return val;
      };
      auto offsetX = createOffset(1);
      auto offsetY = createOffset(0);
      auto idx2 = createIntConstant(i32Type, 2);
      auto idx3 = createIntConstant(i32Type, 3);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              offsetX, idx2);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              offsetY, idx3);
      payLoad = rewriter.create<spirv::BitcastOp>(loc, v2i64, payLoad);
    }
    rewriter.replaceOp(op, payLoad);
    return success();
  }
};

class UpdateNDOffsetToVCPattern : public OpConversionPattern<UpdateNDOffsetOp> {
public:
  using OpConversionPattern<UpdateNDOffsetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UpdateNDOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto desc = adaptor.getTensorDesc();
    auto i32Type = rewriter.getI32Type();
    auto v4i32 = VectorType::get(4, i32Type);
    auto v2i64 = VectorType::get(2, rewriter.getI64Type());
    Value cast = rewriter.create<spirv::BitcastOp>(loc, v4i32, desc);
    auto offsets = adaptor.getOffsets();
    for (auto i = 0; i < offsets.size(); i++) {
      auto offset = offsets[i];
      if (auto cst = dyn_cast<spirv::ConstantOp>(offset.getDefiningOp()))
        if (auto attr = dyn_cast<mlir::IntegerAttr>(cst.getValue());
            attr && attr.getInt() == 0)
          continue;
      auto idx2 = rewriter.create<spirv::ConstantOp>(
          loc, i32Type, rewriter.getIntegerAttr(i32Type, 2));
      auto idx3 = rewriter.create<spirv::ConstantOp>(
          loc, i32Type, rewriter.getIntegerAttr(i32Type, 3));
      Value idx = i == 0 ? idx3 : idx2;
      auto oldOffset =
          rewriter.create<spirv::VectorExtractDynamicOp>(loc, cast, idx);
      offset = rewriter.create<arith::TruncIOp>(loc, i32Type, offset);
      auto newOffset =
          rewriter.create<spirv::IAddOp>(loc, i32Type, oldOffset, offset);
      cast = rewriter.create<spirv::VectorInsertDynamicOp>(loc, v4i32, cast,
                                                           newOffset, idx);
    }
    rewriter.replaceOpWithNewOp<spirv::BitcastOp>(op, v2i64, cast);
    return success();
  }
};

class CreateDescToVCPattern : public OpConversionPattern<CreateDescOp> {
public:
  using OpConversionPattern<CreateDescOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CreateDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();
    auto v2i64 = VectorType::get(2, i64Type);
    Value payLoad = rewriter.create<spirv::UndefOp>(loc, v2i64);
    auto base = rewriter.create<spirv::ConvertPtrToUOp>(loc, i64Type,
                                                        adaptor.getSource());
    auto idx0 = rewriter.create<spirv::ConstantOp>(
        loc, i32Type, rewriter.getIntegerAttr(i32Type, 0));
    payLoad =
        rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad, base, idx0);
    rewriter.replaceOp(op, payLoad);
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
    auto tileType = op.getTensorDesc().getType();
    int rank = tileType.getRank();
    assert(rank <= 2 && "only support 1d/2d load/store/prefetch for now");
    auto loc = op.getLoc();
    ::mlir::VectorType vecType;
    std::string funcName;
    constexpr bool isLoad = std::is_same_v<OpType, LoadNDOp>;
    constexpr bool isPrefetch = std::is_same_v<OpType, PrefetchNDOp>;
    if constexpr (isLoad) {
      vecType = cast<VectorType>(op.getResult().getType());
      funcName = rank == 2 ? "llvm_genx_lsc_load2d_stateless_"
                           : "llvm_genx_lsc_load_stateless_";
    } else if constexpr (isPrefetch) {
      vecType = VectorType::get({8, 16}, rewriter.getF32Type());
      funcName = rank == 2 ? "llvm_genx_lsc_prefetch2d_stateless_i1_i64"
                           : "llvm_genx_lsc_prefetch_stateless_";
    } else {
      vecType = cast<VectorType>(op.getValue().getType());
      funcName = rank == 2 ? "llvm_genx_lsc_store2d_stateless_i1_i64_"
                           : "llvm_genx_lsc_store_stateless_i1_i64_";
    }
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<spirv::ConstantOp>(loc, type, attr);
    };
    auto i8Type = rewriter.getI8Type();
    auto i16Type = rewriter.getI16Type();
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
    unsigned dataSize = encodeDataum(vecType.getElementType());
    auto dataum = createIntConstant(i8Type, dataSize);
    auto trans = createIntConstant(i8Type, transpose ? 2 : 1);
    // number of blocks(1 for now)
    auto nBlks = createIntConstant(i8Type, 1);
    auto tensorDesc = adaptor.getTensorDesc();
    auto idx0 = createIntConstant(i32Type, 0);
    auto base =
        rewriter.create<spirv::VectorExtractDynamicOp>(loc, tensorDesc, idx0);
    std::string typeStr;
    VectorType newType;
    std::tie(typeStr, newType) = encodeVectorType(rewriter, vecType, rank == 1);
    SmallVector<Value> args;
    if (rank == 2) {
      auto blockWidth = tileType.getShape()[1];
      auto blockHeight = tileType.getShape()[0];
      auto blockW = createIntConstant(i32Type, blockWidth);
      auto blockH = createIntConstant(i32Type, blockHeight);
      auto transform = createIntConstant(i8Type, vnni ? 1 : 0);
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
      auto v4i32 = VectorType::get(4, i32Type);
      tensorDesc = rewriter.create<spirv::BitcastOp>(loc, v4i32, tensorDesc);
      auto idx2 = createIntConstant(i32Type, 2);
      auto idx3 = createIntConstant(i32Type, 3);
      auto offsetX =
          rewriter.create<spirv::VectorExtractDynamicOp>(loc, tensorDesc, idx2);
      auto offsetY =
          rewriter.create<spirv::VectorExtractDynamicOp>(loc, tensorDesc, idx3);
      args.assign({pred, l1CacheHint, l3CacheHint, dataum, trans, nBlks, blockW,
                   blockH, transform, base, surfaceW, surfaceH, surfaceP,
                   offsetX, offsetY});
      if constexpr (!isLoad && !isPrefetch) {
        args.push_back(adaptor.getValue());
      }
    } else if (rank == 1) {
      auto subOpcode =
          createIntConstant(i8Type, (isLoad || isPrefetch) ? 0 : 4);
      auto addrScale = createIntConstant(i16Type, 1);
      auto immOffset = createIntConstant(i32Type, 0);
      auto dataumSize = createIntConstant(i8Type, 4);
      int lscVecSize = 0;
      int numElts = newType.getNumElements();
      if (numElts <= 4) {
        lscVecSize = numElts;
      } else {
        lscVecSize = log2(numElts) + 2;
      }
      auto vecSize = createIntConstant(i8Type, lscVecSize);
      auto transposed = createIntConstant(i8Type, 2); // transpose
      auto mask = createIntConstant(i8Type, 0);
      auto surface = createIntConstant(i32Type, 0);
      args.assign({
          pred,
          subOpcode,
          l1CacheHint,
          l3CacheHint,
          addrScale,
          immOffset,
          dataumSize,
          vecSize,
          transposed,
          mask,
          base,
      });
      if constexpr (!isLoad && !isPrefetch) {
        auto cast =
            rewriter.create<spirv::BitcastOp>(loc, newType, adaptor.getValue());
        args.push_back(cast);
      }
      args.push_back(surface);
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
      if (rank == 2) {
        rewriter.replaceOp(op, funcOp);
      } else {
        auto cast = rewriter.create<spirv::BitcastOp>(loc, op.getType(),
                                                      funcOp->getResult(0));
        rewriter.replaceOp(op, cast);
      }
    } else {
      auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
      rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(), funcName, args);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

std::optional<xegpu::CreateNdDescOp> findDescOp(mlir::Value val) {
  if (auto op = val.getDefiningOp()) {
    if (auto descOp = dyn_cast<xegpu::CreateNdDescOp>(op)) {
      return descOp;
    } else if (auto update = dyn_cast<xegpu::UpdateNDOffsetOp>(op)) {
      return findDescOp(update.getTensorDesc());
    }
  } else if (auto arg = dyn_cast<BlockArgument>(val)) {
    auto ownerOp = arg.getOwner()->getParentOp();
    auto forOp = cast<scf::ForOp>(ownerOp);
    auto init = forOp.getInits()[arg.getArgNumber() - 1];
    return findDescOp(init);
  }
  // Add more support
  return std::nullopt;
}

template <typename OpType>
class LoadStorePrefetchNdToRawSend : public OpConversionPattern<OpType> {
public:
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileType = op.getTensorDesc().getType();
    auto rank = tileType.getRank();
    assert(rank <= 2 && "only support 1d/2d load/store/prefetch for now");
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
    auto elmType = tileType.getElementType();
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
    unsigned numDstVal = newType.getNumElements() / 16;
    if (rank == 1) {
      numDstVal *= 2;
    }
    auto numDst = createIntConstant(i8Type, numDstVal);
    // 15 for ugm
    auto sfid = createIntConstant(i8Type, 15);
    auto extMsg = createIntConstant(i32Type, 0);
    // message descriptor
    uint32_t rawSendMsg = 0;
    if (rank == 2) {
      rawSendMsg |= (isLoad || isPrefetch) ? 3 : 7;
      rawSendMsg |= (vnni ? 1 : 0) << 7;
      rawSendMsg |= (encodeDataum(elmType) - 1) << 9;
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
    // payload
    // payload is v8i32 = [base:i64, surfaceWidth:i32, surfaceHeight:i32,
    // surefacePitch:i32, offsetX:i32, offsetY:i32, blockInfo:i32]
    // the base/surfaceInfo/blockInfo are staticly from the tensor desc
    // while the offsetX/Y are dynamicly udpated
    auto insertPoint = rewriter.saveInsertionPoint();
    CreateNdDescOp createDescOp = *findDescOp(op.getTensorDesc());
    rewriter.setInsertionPointAfter(createDescOp);
    auto v8i32 = VectorType::get(8, i32Type);
    auto v4i64 = VectorType::get(4, i64Type);
    Value payLoad = rewriter.create<spirv::UndefOp>(loc, v4i64);
    auto idx0 = createIntConstant(i32Type, 0);
    auto desc = rewriter.getRemappedValue(createDescOp);
    auto base = rewriter.create<spirv::VectorExtractDynamicOp>(loc, desc, idx0);
    payLoad =
        rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad, base, idx0);
    payLoad = rewriter.create<spirv::BitcastOp>(loc, v8i32, payLoad);
    if (rank == 2) {
      auto idx2 = createIntConstant(i32Type, 2);
      auto idx3 = createIntConstant(i32Type, 3);
      auto idx4 = createIntConstant(i32Type, 4);
      auto idx5 = createIntConstant(i32Type, 5);
      auto idx6 = createIntConstant(i32Type, 6);
      auto idx7 = createIntConstant(i32Type, 7);
      auto blockWidth = tileType.getShape()[1];
      auto blockHeight = tileType.getShape()[0];
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
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              surfaceW, idx2);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              surfaceH, idx3);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              surfaceP, idx4);
      unsigned blockVal = ((blockHeight - 1) << 8) | (blockWidth - 1);
      auto blockInfo = createIntConstant(i32Type, blockVal);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              blockInfo, idx7);
      rewriter.restoreInsertionPoint(insertPoint);
      auto v4i32 = VectorType::get(4, i32Type);
      auto tensorDesc = adaptor.getTensorDesc();
      tensorDesc = rewriter.create<spirv::BitcastOp>(loc, v4i32, tensorDesc);
      auto offsetX =
          rewriter.create<spirv::VectorExtractDynamicOp>(loc, tensorDesc, idx2);
      auto offsetY =
          rewriter.create<spirv::VectorExtractDynamicOp>(loc, tensorDesc, idx3);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              offsetX, idx5);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              offsetY, idx6);
    }
    rewriter.restoreInsertionPoint(insertPoint);
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
      if (rank == 2) {
        rewriter.replaceOp(op, funcOp);
      } else {
        auto cast = rewriter.create<spirv::BitcastOp>(loc, op.getType(),
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
          auto cast = rewriter.create<spirv::BitcastOp>(loc, newType,
                                                        adaptor.getValue());
          args.push_back(cast);
        }
      }
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
      return rewriter.create<spirv::ConstantOp>(loc, type, attr);
    };

    /// collect common info
    auto i1Type = rewriter.getI1Type();
    auto i8Type = rewriter.getI8Type();
    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();
    auto tensorDesc = adaptor.getTensorDesc();
    auto idx0 = createIntConstant(i32Type, 0);
    auto base =
        rewriter.create<spirv::VectorExtractDynamicOp>(loc, tensorDesc, idx0);
    VectorType newType = VectorType::get(1, i32Type);
    std::string funcName;
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
    unsigned cacheHint = encodeCacheHint(op);

    /// fill in parameters for raw.send
    // bit[1:0] EOT,sendc
    auto modifier = createIntConstant(i8Type, 0);
    auto execSize = createIntConstant(i8Type, 4);
    auto pred = createIntConstant(i1Type, 1);
    auto numSrc1 = createIntConstant(i8Type, 2);
    unsigned numDstVal = newType.getNumElements() / 16;
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
    auto v16i64 = VectorType::get(16, i64Type);
    Value payLoad = rewriter.create<spirv::UndefOp>(loc, v16i64);
    payLoad =
        rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad, base, idx0);
    SmallVector<int32_t, 16> indices(16, 0);
    payLoad = rewriter.create<spirv::VectorShuffleOp>(
        loc, v16i64, payLoad, payLoad, rewriter.getI32ArrayAttr(indices));
    auto createDescOp =
        op.getTensorDesc().template getDefiningOp<CreateDescOp>();
    auto offsets = rewriter.getRemappedValue(createDescOp.getOffsets());
    payLoad = rewriter.create<spirv::IAddOp>(loc, v16i64, payLoad, offsets);
    SmallVector<Value> args{modifier, execSize, pred, numSrc1, numDst,
                            sfid,     extMsg,   msg,  payLoad};
    if constexpr (isLoad) {
      funcName += "_i1_v16i64";
      auto old = rewriter.create<spirv::UndefOp>(loc, newType);
      args.push_back(old);
      auto retType = newType;
      auto funcType =
          rewriter.getFunctionType(ValueRange(args).getTypes(), retType);
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
      auto funcOp =
          rewriter.create<spirv::FunctionCallOp>(loc, retType, funcName, args);
      auto castTy = this->getTypeConverter()->convertType(op.getType());
      auto cast =
          rewriter.create<spirv::BitcastOp>(loc, castTy, funcOp->getResult(0));
      rewriter.replaceOp(op, cast);
    } else {
      Value data = adaptor.getValue();
      if (data.getType() != newType) {
        data = rewriter.create<spirv::BitcastOp>(loc, newType, data);
      }
      args.push_back(data);
      auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
      rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(), funcName, args);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

class AtomicToLsc : public OpConversionPattern<AtomicRMWOp> {
public:
  using OpConversionPattern<AtomicRMWOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileType = op.getTensorDesc().getType();
    auto rank = tileType.getRank();
    assert(rank <= 2 && "only support 1d/2d for now");
    auto loc = op->getLoc();
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
    VectorType vecType = cast<VectorType>(op.getResult().getType());
    std::string funcName = "llvm_genx_lsc_xatomic_stateless_";
    auto [typeStr, newType] = encodeVectorType(rewriter, vecType, false, true);
    funcName += typeStr;

    /// fill in parameters for lsc
    auto v16i1 = VectorType::get(16, i1Type);
    auto vecAttr = DenseElementsAttr::get(v16i1, true);
    auto pred = rewriter.create<spirv::ConstantOp>(loc, v16i1, vecAttr);
    auto subOpcode = createIntConstant(i8Type, encodeOpcode(op.getKind()));
    auto l1CacheHint = createIntConstant(i8Type, 1);
    auto l3CacheHint = createIntConstant(i8Type, 1);
    auto addrScale = createIntConstant(i16Type, 1);
    auto immOffset = createIntConstant(i32Type, 0);
    unsigned dataSize = encodeDataum(vecType.getElementType());
    auto dataumSize = createIntConstant(i8Type, dataSize);
    unsigned numDstVal = newType.getNumElements() / 16;
    auto lscVecSize = 0;
    if (numDstVal <= 4) {
      lscVecSize = numDstVal;
    } else {
      lscVecSize = log2(numDstVal) + 2;
    }
    auto vecSize = createIntConstant(i8Type, lscVecSize);
    auto transposed = createIntConstant(i8Type, 1);
    auto mask = createIntConstant(i8Type, 0);

    auto tensorDesc = adaptor.getTensorDesc();
    auto idx0 = createIntConstant(i32Type, 0);
    auto base =
        rewriter.create<spirv::VectorExtractDynamicOp>(loc, tensorDesc, idx0);
    // payload
    auto v16i64 = VectorType::get(16, i64Type);
    Value payLoad = rewriter.create<spirv::UndefOp>(loc, v16i64);
    payLoad =
        rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad, base, idx0);
    SmallVector<int32_t, 16> indices(16, 0);
    payLoad = rewriter.create<spirv::VectorShuffleOp>(
        loc, v16i64, payLoad, payLoad, rewriter.getI32ArrayAttr(indices));
    auto createDescOp =
        op.getTensorDesc().template getDefiningOp<CreateDescOp>();
    auto offsets = rewriter.getRemappedValue(createDescOp.getOffsets());
    payLoad = rewriter.create<spirv::IAddOp>(loc, v16i64, payLoad, offsets);
    // src
    auto v16i32 = VectorType::get(16, i32Type);
    Value undef = rewriter.create<spirv::UndefOp>(loc, v16i32);
    Value src0 = undef;
    if (op.getValue()) {
      src0 = op.getValue();
      if (src0.getType() != newType) {
        src0 = rewriter.create<spirv::BitcastOp>(loc, newType, src0);
      }
    }
    Value src1 = undef;
    auto surface = createIntConstant(i32Type, 0);
    SmallVector<Value> args{pred,       subOpcode, l1CacheHint, l3CacheHint,
                            addrScale,  immOffset, dataumSize,  vecSize,
                            transposed, mask,      payLoad,     src0,
                            src1,       surface,   undef};
    funcName += "_v16i1_v16i64";
    auto retType = newType;
    auto funcType =
        rewriter.getFunctionType(ValueRange(args).getTypes(), retType);
    Operation *opPtr = op;
    lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
    auto funcOp =
        rewriter.create<spirv::FunctionCallOp>(loc, retType, funcName, args);
    auto castTy = this->getTypeConverter()->convertType(op.getType());
    auto cast =
        rewriter.create<spirv::BitcastOp>(loc, castTy, funcOp->getResult(0));
    rewriter.replaceOp(op, cast);
    return success();
  }
};

Value createConstantI32(Location loc, PatternRewriter &rewriter, int32_t v) {
  auto i32ty = rewriter.getIntegerType(32);
  return rewriter.create<spirv::ConstantOp>(loc, i32ty,
                                            IntegerAttr::get(i32ty, v));
}

#define zext(...) rewriter.create<spirv::UConvertOp>(loc, __VA_ARGS__)
#define logic_shl(...)                                                         \
  rewriter.create<spirv::ShiftLeftLogicalOp>(loc, __VA_ARGS__)
#define bitwise_or(...) rewriter.create<spirv::BitwiseOrOp>(loc, __VA_ARGS__)
#define bitwise_and(...) rewriter.create<spirv::BitwiseAndOp>(loc, __VA_ARGS__)
#define i32_val(...) createConstantI32(loc, rewriter, __VA_ARGS__)
#define i8_val(value)                                                          \
  rewriter.create<spirv::ConstantOp>(loc, rewriter.getIntegerType(8),          \
                                     rewriter.getI8IntegerAttr(value))
#define i1_val(value)                                                          \
  rewriter.create<spirv::ConstantOp>(loc, rewriter.getI1Type(),                \
                                     rewriter.getBoolAttr(value))

class AllocNbarrierToVCPattern : public OpConversionPattern<AllocNbarrierOp> {
public:
  using OpConversionPattern<AllocNbarrierOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AllocNbarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    auto func = op->getParentOfType<spirv::FuncOp>();
    rewriter.setInsertionPointAfter(func);
    rewriter.create<spirv::ExecutionModeOp>(
        op.getLoc(), func, spirv::ExecutionMode::NamedBarrierCountINTEL,
        op.getNbarrierCount());
    rewriter.eraseOp(op);
    return success();
  }
};

class CreateNbarrierToVCPattern : public OpConversionPattern<CreateNbarrierOp> {
public:
  using OpConversionPattern<CreateNbarrierOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CreateNbarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto nbarrier_id = op.getNbarrierId();
    auto nbarrier_role = op.getNbarrierRole();
    auto num_producers = op.getNumProducers();
    auto num_consumers = op.getNumConsumers();

    auto i32Type = rewriter.getIntegerType(32);
    auto v8i32Type = mlir::VectorType::get(8, i32Type);

    DenseElementsAttr constantData = DenseElementsAttr::get(
        v8i32Type, ArrayRef<int>(std::vector<int>(1, 0)));
    Value nbarrier_src =
        rewriter.create<spirv::ConstantOp>(loc, v8i32Type, constantData);

    Value payload = zext(i32Type, nbarrier_id);

    Value payload_nbarrier_role =
        logic_shl(i32Type, zext(i32Type, nbarrier_role), i32_val(14));
    payload = bitwise_or(i32Type, payload, payload_nbarrier_role);

    Value payload_num_producers =
        logic_shl(i32Type, i32_val(num_producers), i32_val(16));
    payload = bitwise_or(i32Type, payload, payload_num_producers);

    Value payload_num_consumers =
        logic_shl(i32Type, i32_val(num_consumers), i32_val(24));
    payload = bitwise_or(i32Type, payload, payload_num_consumers);

    nbarrier_src = rewriter.create<spirv::VectorInsertDynamicOp>(
        loc, v8i32Type, nbarrier_src, payload, i32_val(2));
    rewriter.replaceOp(op, nbarrier_src);

    return success();
  }
};

class NbarrierArriveToVCPattern : public OpConversionPattern<NbarrierArriveOp> {
public:
  using OpConversionPattern<NbarrierArriveOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NbarrierArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto payload = adaptor.getPayload();

    std::string funcName = "llvm_genx_raw_send2_noresult_i1_v8i32";

    // desc format
    // https://github.com/intel-innersource/drivers.gpu.compute.vc-intrinsics/blob/cmc_experimental/GenXIntrinsics/include/llvm/GenXIntrinsics/Intrinsic_definitions.py#L4595
    Value modifier = i8_val(0);
    Value exec_size = i8_val(0);
    Value predicate = i1_val(1);
    Value numsrc1 = i8_val(1); // register nums of payload
    Value sfid = i8_val(3);
    Value etDesc = i32_val(0);
    Value msg_desc = i32_val(0x2000004);

    SmallVector<Value> args{modifier, exec_size, predicate, numsrc1,
                            sfid,     etDesc,    msg_desc,  payload};

    auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});

    Operation *opPtr = op;
    lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
    rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(), funcName, args);

    rewriter.eraseOp(op);
    return success();
  }
};

class NbarrierWaitToVCPattern : public OpConversionPattern<NbarrierWaitOp> {
public:
  using OpConversionPattern<NbarrierWaitOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NbarrierWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto payload = adaptor.getPayload();

    auto i8Type = rewriter.getIntegerType(8);
    auto i32Type = rewriter.getIntegerType(32);
    auto nbarrier_src = rewriter.create<spirv::VectorExtractDynamicOp>(
        loc, i32Type, payload, i32_val(2));
    auto nbarrier_id =
        zext(i8Type, bitwise_and(i32Type, nbarrier_src, i32_val(0xFF)));

    Value signal_flag = i8_val(0); // 0b0: wait 0b1: signal
    Value num_threads = i8_val(0); // This field is ignored for nbarrier.wait

    std::string funcName = "llvm_genx_nbarrier";
    SmallVector<Value> args{signal_flag, nbarrier_id, num_threads};

    auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});

    Operation *opPtr = op;
    lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
    rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(), funcName, args);

    rewriter.eraseOp(op);
    return success();
  }
};

class CompilerHintToVCPattern : public OpConversionPattern<CompileHintOp> {
public:
  using OpConversionPattern<CompileHintOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CompileHintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    std::string funcName = "llvm_genx_fence";
    Value fence_flag = i8_val(-128);
    SmallVector<Value> args{fence_flag};
    auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});

    Operation *opPtr = op;
    lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
    rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(), funcName, args);

    rewriter.eraseOp(op);
    return success();
  }
};

class MfenceToVCPattern : public OpConversionPattern<MfenceOp> {
public:
  using OpConversionPattern<MfenceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MfenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto pred = i1_val(1);
    auto fence_op_attr = op.getFenceOpAttr().str();
    auto fence_scope_attr = op.getFenceScopeAttr().str();
    auto memory_kind_attr = op.getMemoryKindAttr().str();

    std::vector<std::string> lscFenceOp{"none",    "evict", "invalidate",
                                        "discard", "clean", "flushl3"};
    std::vector<std::string> lscFenceScope{"group", "local",  "tile",  "gpu",
                                           "gpus",  "system", "sysacq"};
    std::vector<std::string> lscSFID{"ugm", "ugml", "tgm", "slm"};

    uint8_t fence_op, fence_scope, sfid;

    auto it = std::find(lscFenceOp.begin(), lscFenceOp.end(), fence_op_attr);
    if (it != lscFenceOp.end()) {
      fence_op = std::distance(lscFenceOp.begin(), it);
    } else {
      llvm_unreachable("unsupported value for lsc_fence_op attribute");
    }

    it =
        std::find(lscFenceScope.begin(), lscFenceScope.end(), fence_scope_attr);
    if (it != lscFenceScope.end()) {
      fence_scope = std::distance(lscFenceScope.begin(), it);
    } else {
      llvm_unreachable("unsupported value for lsc_fence_scope attribute");
    }

    it = std::find(lscSFID.begin(), lscSFID.end(), memory_kind_attr);
    if (it != lscSFID.end()) {
      sfid = std::distance(lscSFID.begin(), it);
    } else {
      llvm_unreachable("unsupported value for memory_kind attribute");
    }

    SmallVector<Value> args{pred, i8_val(sfid), i8_val(fence_op),
                            i8_val(fence_scope)};
    auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});

    std::string funcName = "llvm.genx.lsc.fence.i1";

    Operation *opPtr = op;
    lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
    rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(), funcName, args);

    rewriter.eraseOp(op);
    return success();
  }
};
/// add necessary vectorTospirv patterns (different from upstream)
struct VectorShapeCast final : public OpConversionPattern<vector::ShapeCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(vector::ShapeCastOp shapeCastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type dstType = getTypeConverter()->convertType(shapeCastOp.getType());
    if (!dstType)
      return failure();
    if (dstType == adaptor.getSource().getType() ||
        shapeCastOp.getResultVectorType().getNumElements() == 1) {
      rewriter.replaceOp(shapeCastOp, adaptor.getSource());
      return success();
    }
    rewriter.replaceOpWithNewOp<spirv::BitcastOp>(shapeCastOp, dstType,
                                                  adaptor.getSource());
    return success();
  }
};
} // namespace

void imex::populateXeGPUToVCIntrinsicsPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<CreateNdDescToVCPattern, CreateDescToVCPattern, DpasToVCPattern,
               AllocNbarrierToVCPattern, CreateNbarrierToVCPattern,
               NbarrierArriveToVCPattern, NbarrierWaitToVCPattern,
               CompilerHintToVCPattern, MfenceToVCPattern, VectorShapeCast,
               GatherScatterToRawSend<LoadGatherOp>,
               GatherScatterToRawSend<StoreScatterOp>, AtomicToLsc,
               UpdateNDOffsetToVCPattern>(typeConverter, patterns.getContext());
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
