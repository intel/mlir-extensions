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
#include "imex/Dialect/XeGPU/IR/XeGPU.h"

#include "../PassDetail.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Debug.h>

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

#include <numeric>

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
  case 256:
    str += "v256";
    break;
  case 512:
    str += "v512";
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
                             std::string name, FunctionType funcType,
                             bool isVC = true) {
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
    if (isVC)
      func->setAttr("VectorComputeFunctionINTEL", rewriter.getUnitAttr());
  }
}

/// @brief
/// assemble the tensor descriptor payload[8xi32] which is of the format
/// -> [base pointer, surface width, surface height, surface pitch,
///     offsetX, offsetY, blockInfo] for 2D tensor desc
/// -> [base pointer, unused] for 1D and scattered tensor desc
/// only base pointer is i64, others are i32
class CreateNdDescToSPIRV : public OpConversionPattern<CreateNdDescOp> {
public:
  using OpConversionPattern<CreateNdDescOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CreateNdDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();
    // payload
    auto v8i32 = VectorType::get(8, i32Type);
    auto v4i64 = VectorType::get(4, i64Type);
    Value payLoad = rewriter.create<spirv::UndefOp>(loc, v4i64);
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<spirv::ConstantOp>(loc, type, attr);
    };
    auto base = rewriter.create<spirv::ConvertPtrToUOp>(loc, i64Type,
                                                        adaptor.getSource());
    auto idx0 = createIntConstant(i32Type, 0);
    payLoad =
        rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad, base, idx0);
    payLoad = rewriter.create<spirv::BitcastOp>(loc, v8i32, payLoad);
    auto tileType = op.getTensorDesc().getType();
    auto rank = tileType.getRank();
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
      auto memType = cast<MemRefType>(op.getSource().getType());
      unsigned bitWidth = memType.getElementType().getIntOrFloatBitWidth();
      Value surfaceW, surfaceH, surfaceP;
      if (memType.hasStaticShape()) {
        auto surfaceWidth = memType.getShape()[1] * (bitWidth / 8) - 1;
        auto surfaceHeight = memType.getShape()[0] - 1;
        // fixme: pitch = width for now
        auto surfacePitch = surfaceWidth;
        surfaceW = createIntConstant(i32Type, surfaceWidth);
        surfaceH = createIntConstant(i32Type, surfaceHeight);
        surfaceP = createIntConstant(i32Type, surfacePitch);

      } else {
        // get the surfaceWidth and Height from the op attributes
        // compute surface width
        auto bytesPerElem = createIntConstant(i32Type, bitWidth / 8);
        auto one = createIntConstant(i32Type, 1);
        surfaceW = rewriter.create<spirv::UConvertOp>(
            loc, i32Type, adaptor.getDynamicShape()[1]);
        surfaceW = rewriter.create<spirv::IMulOp>(loc, surfaceW, bytesPerElem);
        surfaceW = rewriter.create<spirv::ISubOp>(loc, surfaceW, one);
        // compute surface height
        surfaceH = rewriter.create<spirv::UConvertOp>(
            loc, i32Type, adaptor.getDynamicShape()[0]);
        surfaceH = rewriter.create<spirv::ISubOp>(loc, surfaceH, one);
        // fixme: pitch = width for now
        surfaceP = surfaceW;
      }

      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              surfaceW, idx2);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              surfaceH, idx3);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              surfaceP, idx4);
      auto createOffset = [&](unsigned idx) -> Value {
        Value val;
        OpFoldResult ofr = op.getOffsets()[idx];
        auto v = llvm::dyn_cast_if_present<Value>(ofr);
        if (v) {
          val = ofr.get<Value>();
          val = rewriter.create<arith::TruncIOp>(loc, i32Type, val);
        } else {
          int off = llvm::cast<IntegerAttr>(ofr.get<Attribute>()).getInt();
          val = createIntConstant(i32Type, off);
        }
        return val;
      };
      auto offsetX = createOffset(1);
      auto offsetY = createOffset(0);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              offsetX, idx5);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              offsetY, idx6);
      int array_length = op.getTensorDescType().getArrayLength();
      unsigned blockVal = (array_length - 1) << 16;
      blockVal |= ((blockHeight - 1) << 8) | (blockWidth - 1);
      auto blockInfo = createIntConstant(i32Type, blockVal);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              blockInfo, idx7);
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
    auto i32Type = rewriter.getI32Type();
    auto offsets = adaptor.getOffsets();
    auto desc = adaptor.getTensorDesc();
    for (size_t i = 0; i < offsets.size(); i++) {
      auto offset = offsets[i];
      if (auto cst = dyn_cast<spirv::ConstantOp>(offset.getDefiningOp()))
        if (auto attr = dyn_cast<mlir::IntegerAttr>(cst.getValue());
            attr && attr.getInt() == 0)
          continue;
      auto idx5 = rewriter.create<spirv::ConstantOp>(
          loc, i32Type, rewriter.getIntegerAttr(i32Type, 5));
      auto idx6 = rewriter.create<spirv::ConstantOp>(
          loc, i32Type, rewriter.getIntegerAttr(i32Type, 6));
      Value idx = i == 0 ? idx6 : idx5;
      auto oldOffset =
          rewriter.create<spirv::VectorExtractDynamicOp>(loc, desc, idx);
      offset = rewriter.create<arith::TruncIOp>(loc, i32Type, offset);
      auto newOffset =
          rewriter.create<spirv::IAddOp>(loc, i32Type, oldOffset, offset);
      desc = rewriter.create<spirv::VectorInsertDynamicOp>(loc, desc, newOffset,
                                                           idx);
    }
    rewriter.replaceOp(op, desc);
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
    auto v8i32 = VectorType::get(8, i32Type);
    auto v4i64 = VectorType::get(4, i64Type);
    Value payLoad = rewriter.create<spirv::UndefOp>(loc, v4i64);
    auto base = rewriter.create<spirv::ConvertPtrToUOp>(loc, i64Type,
                                                        adaptor.getSource());
    auto idx0 = rewriter.create<spirv::ConstantOp>(
        loc, i32Type, rewriter.getIntegerAttr(i32Type, 0));
    payLoad =
        rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad, base, idx0);
    payLoad = rewriter.create<spirv::BitcastOp>(loc, v8i32, payLoad);
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
    auto i64Type = rewriter.getI64Type();
    auto v4i64 = VectorType::get(4, i64Type);
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
    auto array_length = op.getTensorDescType().getArrayLength();
    auto nBlks = createIntConstant(i8Type, array_length);
    auto tensorDesc = adaptor.getTensorDesc();
    auto idx0 = createIntConstant(i32Type, 0);
    auto cast = rewriter.create<spirv::BitcastOp>(loc, v4i64, tensorDesc);
    auto base = rewriter.create<spirv::VectorExtractDynamicOp>(loc, cast, idx0);
    auto [typeStr, newType] = encodeVectorType(rewriter, vecType, rank == 1);
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
      auto memType = llvm::cast<MemRefType>(createDescOp.getSource().getType());
      unsigned bitWidth = memType.getElementType().getIntOrFloatBitWidth();
      auto surfaceWidth = memType.getShape()[1] * (bitWidth / 8) - 1;
      auto surfaceHeight = memType.getShape()[0] - 1;
      // pitch = width for now
      auto surfacePitch = surfaceWidth;
      auto surfaceW = createIntConstant(i32Type, surfaceWidth);
      auto surfaceH = createIntConstant(i32Type, surfaceHeight);
      auto surfaceP = createIntConstant(i32Type, surfacePitch);
      auto idx5 = createIntConstant(i32Type, 5);
      auto idx6 = createIntConstant(i32Type, 6);
      auto offsetX =
          rewriter.create<spirv::VectorExtractDynamicOp>(loc, tensorDesc, idx5);
      auto offsetY =
          rewriter.create<spirv::VectorExtractDynamicOp>(loc, tensorDesc, idx6);
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
    // vnni and transpose combination is required for the case where B matrix is
    // transposed and we need to load from B in DPAS layout. However, HW does
    // not support both vnni and transpose together. We can get the same layout
    // for the B load by doing the transpose in 32 bit granularity.
    // TODO: Transpose granularity must be explicitly represented in XeGPU op.
    if (vnni && transpose) {
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
      auto idx5 = createIntConstant(i32Type, 5);
      auto idx7 = createIntConstant(i32Type, 7);

      auto oldOffsetX =
          rewriter.create<spirv::VectorExtractDynamicOp>(loc, payLoad, idx5);
      // do an aritmetic right shift instead of divide.
      auto newOffsetX = rewriter.create<spirv::ShiftRightArithmeticOp>(
          loc, oldOffsetX, createIntConstant(i32Type, getLog2OfVnniFactor()));
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              newOffsetX, idx5);
      int array_length = op.getTensorDescType().getArrayLength();
      unsigned blockVal = (array_length - 1) << 16;
      auto blockWidth = tileType.getShape()[1];
      auto blockHeight = tileType.getShape()[0];
      auto newBlockWidth = blockWidth / vnniFactor;
      blockVal |= ((blockHeight - 1) << 8) | (newBlockWidth - 1);
      auto blockInfo = createIntConstant(i32Type, blockVal);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              blockInfo, idx7);
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
    auto v4i64 = VectorType::get(4, i64Type);
    auto tensorDesc = adaptor.getTensorDesc();
    auto idx0 = createIntConstant(i32Type, 0);
    tensorDesc = rewriter.create<spirv::BitcastOp>(loc, v4i64, tensorDesc);
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
    auto v4i64 = VectorType::get(4, i64Type);
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
    tensorDesc = rewriter.create<spirv::BitcastOp>(loc, v4i64, tensorDesc);
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
struct VectorExtract final : public OpConversionPattern<vector::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(vector::ExtractOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstTy = getTypeConverter()->convertType(extractOp.getType());
    if (!dstTy)
      return failure();

    // dynamic position is not supported
    if (extractOp.hasDynamicPosition())
      return failure();

    auto srcTy = extractOp.getVector().getType();
    auto shape = srcTy.getShape();
    auto size = srcTy.getNumElements();
    auto vec = adaptor.getVector();
    auto vecTy = llvm::dyn_cast_if_present<mlir::VectorType>(vec.getType());

    if (!vecTy)
      return mlir::failure();

    // for some cases of vector types, current VectorType Converter could
    // convert e.g., vector of fp16/bf16 into vector of i32s, so the size
    // and strides is reduced half. (This is partially because load_2d
    // intrinsics handles i32 types for fp16/bf16, and current converter guess
    // the vector type is used by load/store based on the dims of vector shape)
    auto factor = vecTy.getElementType().getIntOrFloatBitWidth() /
                  srcTy.getElementType().getIntOrFloatBitWidth();
    size /= factor;

    // dstTy is vector<2xf16> or f16, and srcTy is <8x16x2xf16>, but vecTy is
    // <128xi32> because src is the result of load. it is a mismatch of ty.
    // This is an issue raised from current desgin of type converter, and
    // needs to be fixed.
    auto ty = llvm::dyn_cast<mlir::VectorType>(dstTy);
    if ((ty && ty.getElementType() != vecTy.getElementType()) ||
        (!ty && dstTy != vecTy.getElementType()))
      return mlir::failure();

    // compute linearized offset
    int64_t linearizedOffset = 0;
    auto offsets = extractOp.getStaticPosition();
    for (auto [i, off] : llvm::enumerate(offsets)) {
      size /= shape[i];
      linearizedOffset += offsets[i] * size;
    }

    if (ty) { // use VectorShuffer for vector result
      llvm::SmallVector<int32_t, 2> indices(size);
      std::iota(indices.begin(), indices.end(), linearizedOffset);
      rewriter.replaceOpWithNewOp<mlir::spirv::VectorShuffleOp>(
          extractOp, dstTy, vec, vec, rewriter.getI32ArrayAttr(indices));
    } else { // use CompositExtract for scalar result
      rewriter.replaceOpWithNewOp<mlir::spirv::CompositeExtractOp>(
          extractOp, vec, linearizedOffset);
    }

    return success();
  }
};

static uint64_t getFirstIntValue(mlir::ArrayAttr attr) {
  return (*attr.getAsValueRange<IntegerAttr>().begin()).getZExtValue();
};

struct VectorExtractStridedSlice final
    : public OpConversionPattern<vector::ExtractStridedSliceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(vector::ExtractStridedSliceOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = getTypeConverter()->convertType(extractOp.getType());
    if (!dstType)
      return failure();

    // fixme : currently only support 1D vectors
    if (extractOp.getSourceVectorType().getRank() != 1)
      return failure();

    uint64_t offset = getFirstIntValue(extractOp.getOffsets());
    uint64_t size = getFirstIntValue(extractOp.getSizes());
    uint64_t stride = getFirstIntValue(extractOp.getStrides());

    if (stride != 1)
      return failure();

    Value srcVector = adaptor.getOperands().front();

    // Extract vector<1xT> case.
    if (isa<spirv::ScalarType>(dstType)) {
      rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(extractOp,
                                                             srcVector, offset);
      return success();
    }

    SmallVector<int32_t, 2> indices(size);
    std::iota(indices.begin(), indices.end(), offset);

    rewriter.replaceOpWithNewOp<spirv::VectorShuffleOp>(
        extractOp, dstType, srcVector, srcVector,
        rewriter.getI32ArrayAttr(indices));

    return success();
  }
};

struct VectorShuffle final : public OpConversionPattern<vector::ShuffleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ShuffleOp shuffleOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = getTypeConverter()->convertType(shuffleOp.getType());
    if (!dstType)
      return failure();

    auto vec1 = adaptor.getV1();
    auto vec2 = adaptor.getV2();

    int shuffleSliceLen = 1;
    int rank = shuffleOp.getV1().getType().getRank();

    // if rank > 1, we need to do the shuffle in the granularity of slices
    // instead of scalars. Size of the slice is equal to the rank-1 innermost
    // dims. Mask of the shuffle op specifies which slice to take from the
    // outermost dim.
    if (rank > 1) {
      auto shape = shuffleOp.getV1().getType().getShape();
      for (unsigned i = 1; i < shape.size(); i++) {
        shuffleSliceLen *= shape[i];
      }
    }

    // llvm shufflevector does not support  shuffling vectors with
    // unequal sizes. Howver if both vectors are constants this restriction
    // does not apply.
    // FIXME : Currently this only checks for spirv::ConstantOp as the operands.
    // Need to use constant analyis for better support.
    bool bothConstants = isa<spirv::ConstantOp>(vec1.getDefiningOp()) &&
                         isa<spirv::ConstantOp>(vec2.getDefiningOp());
    bool sizeMismatch = shuffleOp.getV1().getType().getShape()[0] !=
                        shuffleOp.getV2().getType().getShape()[0];
    if (!bothConstants && sizeMismatch)
      return rewriter.notifyMatchFailure(
          shuffleOp, "Two source vectors must have equal number of elements.");

    auto mask = shuffleOp.getMask();
    auto totalSize = mask.size() * shuffleSliceLen;

    SmallVector<int32_t, 2> indices(totalSize);
    for (auto [i, value] :
         llvm::enumerate(mask.getAsValueRange<IntegerAttr>())) {

      int32_t v = value.getZExtValue();
      std::iota(indices.begin() + shuffleSliceLen * i,
                indices.begin() + shuffleSliceLen * (i + 1),
                shuffleSliceLen * v);
    }

    rewriter.replaceOpWithNewOp<spirv::VectorShuffleOp>(
        shuffleOp, dstType, vec1, vec2, rewriter.getI32ArrayAttr(indices));

    return success();
  }
};
} // namespace

void imex::populateXeGPUToVCIntrinsicsPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<CreateNdDescToSPIRV, CreateDescToVCPattern, DpasToVCPattern,
               AllocNbarrierToVCPattern, CreateNbarrierToVCPattern,
               NbarrierArriveToVCPattern, NbarrierWaitToVCPattern,
               CompilerHintToVCPattern, MfenceToVCPattern, VectorShapeCast,
               VectorExtract, VectorExtractStridedSlice, VectorShuffle,
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

/// below is for XeGPU to SPIRV genISA Intrinsic

/// @brief encodeVectorType(xxx, 8x8x2xf16, false) returns ["v64i32", 64xi32]
std::pair<std::string, VectorType>
encodeGenISAVectorType(ConversionPatternRewriter &rewriter, VectorType type,
                       bool use32bitData = true) {
  auto elemType = type.getElementType();
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  int size = type.getNumElements() * bitWidth / 16;
  if (use32bitData) {
    size /= 2;
  }
  std::string str = "v";
  str += std::to_string(size);
  if (!use32bitData) {
    str += "i16";
    elemType = rewriter.getI16Type();
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

template <typename OpType>
class LoadStorePrefetchNdToGenISA : public OpConversionPattern<OpType> {
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
      funcName = rank == 2 ? "llvm.genx.GenISA.LSC2DBlockRead."
                           : "llvm.genx.GenISA.LSCLoadBlock.";
    } else if constexpr (isPrefetch) {
      vecType = VectorType::get({8, 16}, rewriter.getF32Type());
      funcName = rank == 2 ? "llvm.genx.GenISA.LSC2DPrefetch."
                           : "llvm.genx.GenISA.LSCPrefetch";
    } else {
      vecType = cast<VectorType>(op.getValue().getType());
      funcName = rank == 2 ? "llvm.genx.GenISA.LSC2DBlockWrite."
                           : "llvm.genx.GenISA.LSCStoreBlock";
    }
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<spirv::ConstantOp>(loc, type, attr);
    };
    auto i1Type = rewriter.getI1Type();
    auto i8Type = rewriter.getI8Type();
    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();
    auto v4i64 = VectorType::get(4, i64Type);
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
    unsigned dataSize = vecType.getElementType().getIntOrFloatBitWidth();
    auto elemSize = createIntConstant(i8Type, dataSize);
    auto trans = createIntConstant(i1Type, transpose ? 1 : 0);
    auto array_length = op.getTensorDescType().getArrayLength();
    auto nBlks = createIntConstant(i8Type, array_length);
    auto tensorDesc = adaptor.getTensorDesc();
    auto idx0 = createIntConstant(i32Type, 0);
    auto cast = rewriter.create<spirv::BitcastOp>(loc, v4i64, tensorDesc);
    auto base = rewriter.create<spirv::VectorExtractDynamicOp>(loc, cast, idx0);
    auto [typeStr, newType] = encodeGenISAVectorType(rewriter, vecType, false);
    SmallVector<Value> args;
    if (rank == 2) {
      auto blockWidth = tileType.getShape()[1];
      auto blockHeight = tileType.getShape()[0];
      auto blockW = createIntConstant(i32Type, blockWidth);
      auto blockH = createIntConstant(i32Type, blockHeight);
      auto transform = createIntConstant(i1Type, vnni ? 1 : 0);
      // static memref for now
      auto createDescOp =
          op.getTensorDesc().template getDefiningOp<CreateNdDescOp>();
      auto memType = llvm::cast<MemRefType>(createDescOp.getSource().getType());
      unsigned bitWidth = memType.getElementType().getIntOrFloatBitWidth();
      auto surfaceWidth = memType.getShape()[1] * (bitWidth / 8) - 1;
      auto surfaceHeight = memType.getShape()[0] - 1;
      // pitch = width for now
      auto surfacePitch = surfaceWidth;
      auto surfaceW = createIntConstant(i32Type, surfaceWidth);
      auto surfaceH = createIntConstant(i32Type, surfaceHeight);
      auto surfaceP = createIntConstant(i32Type, surfacePitch);
      auto idx5 = createIntConstant(i32Type, 5);
      auto idx6 = createIntConstant(i32Type, 6);
      auto offsetX =
          rewriter.create<spirv::VectorExtractDynamicOp>(loc, tensorDesc, idx5);
      auto offsetY =
          rewriter.create<spirv::VectorExtractDynamicOp>(loc, tensorDesc, idx6);
      args.assign({base, surfaceW, surfaceH, surfaceP, offsetX, offsetY,
                   elemSize, blockW, blockH, nBlks, trans, transform});
      if constexpr (!isLoad && !isPrefetch) {
        args.push_back(adaptor.getValue());
      }
    }
    if constexpr (isLoad)
      funcName += typeStr;
    else if constexpr (!isPrefetch)
      funcName += "isVoid";
    if constexpr (isLoad) {
      auto funcType =
          rewriter.getFunctionType(ValueRange(args).getTypes(), newType);
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType, false);
      auto funcOp =
          rewriter.create<spirv::FunctionCallOp>(loc, newType, funcName, args);
      auto castTy = this->getTypeConverter()->convertType(op.getType());
      auto cast =
          rewriter.create<spirv::BitcastOp>(loc, castTy, funcOp->getResult(0));
      rewriter.replaceOp(op, cast);
    } else {
      auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType, false);
      rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(), funcName, args);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

class DpasToGenISA : public OpConversionPattern<DpasOp> {
public:
  using OpConversionPattern<DpasOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(DpasOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto func = op->getParentOfType<spirv::FuncOp>();
      rewriter.setInsertionPointAfter(func);
      rewriter.create<spirv::ExecutionModeOp>(
          op.getLoc(), func, spirv::ExecutionMode::SubgroupSize, 16);
    }
    auto i32Type = rewriter.getI32Type();
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<spirv::ConstantOp>(loc, type, attr);
    };
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
    auto lType = op.getLhs().getType().cast<VectorType>();
    auto rType = op.getRhs().getType().cast<VectorType>();
    auto resultType = op.getResultType().cast<VectorType>();
    auto [lhsStr, lhsType] = encodeGenISAVectorType(rewriter, lType, false);
    auto [rhsStr, rhsType] = encodeGenISAVectorType(rewriter, rType, false);
    auto [newStr, newType] = encodeGenISAVectorType(rewriter, resultType);
    auto lhs =
        rewriter.create<spirv::BitcastOp>(loc, lhsType, adaptor.getLhs());
    auto rhs =
        rewriter.create<spirv::BitcastOp>(loc, rhsType, adaptor.getRhs());
    uint8_t preca = encodePrecision(lType.getElementType());
    uint8_t precb = encodePrecision(rType.getElementType());
    auto precA = createIntConstant(i32Type, preca);
    auto precB = createIntConstant(i32Type, precb);
    // fixed for now
    auto rc = createIntConstant(i32Type, 8);
    auto sd = createIntConstant(i32Type, 8);
    auto dpasW = createIntConstant(rewriter.getI1Type(), 0);
    Value acc = op.getAcc() ? adaptor.getAcc()
                            : rewriter.create<spirv::UndefOp>(loc, newType);
    SmallVector<Value, 8> args{acc, lhs, rhs, precA, precB, sd, rc, dpasW};
    std::string funcName = "llvm.genx.GenISA.sub.group.dpas.";
    funcName += newStr;
    funcName += ".";
    funcName += newStr;
    funcName += ".";
    funcName += lhsStr;
    funcName += ".";
    funcName += rhsStr;
    auto funcType =
        rewriter.getFunctionType(ValueRange(args).getTypes(), newType);
    Operation *opPtr = op;
    lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType, false);
    auto funcOp =
        rewriter.create<spirv::FunctionCallOp>(loc, newType, funcName, args);
    rewriter.replaceOp(op, funcOp);
    return success();
  }
};

void imex::populateXeGPUToGenISAPatterns(SPIRVTypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
  patterns.add<CreateNdDescToSPIRV, DpasToGenISA,
               LoadStorePrefetchNdToGenISA<LoadNDOp>,
               LoadStorePrefetchNdToGenISA<StoreNDOp>,
               LoadStorePrefetchNdToGenISA<PrefetchNDOp>>(
      typeConverter, patterns.getContext());
}

namespace {
// PVC-specific subgroup size for JointMatrix
constexpr uint64_t jointMatrixSubGroupSize = 16;
// Calculate flattened offsets
// Calculate flattened offsets based on dims and offsets(indices)
Value linearizeOffset(OpBuilder builder, Location loc,
                      SmallVectorImpl<Value> &offsets,
                      SmallVectorImpl<Value> &dims) {
  assert(offsets.size() == dims.size() &&
         "number of offsets & dimensions must be same");
  auto createIntConstant = [&](Type type, unsigned value) {
    auto attr = builder.getIntegerAttr(type, value);
    return builder.create<spirv::ConstantOp>(loc, type, attr);
  };

  auto i64Type = builder.getI64Type();
  auto rank = dims.size();
  Value linearizedOffset = createIntConstant(i64Type, 0);
  for (unsigned i = 0; i < rank; i++) {
    Value perDimstrideMultiplier = createIntConstant(i64Type, 1);
    for (unsigned j = i + 1; j < rank; j++) {
      perDimstrideMultiplier = builder.create<spirv::IMulOp>(
          loc, i64Type, perDimstrideMultiplier, dims[j]);
    }
    perDimstrideMultiplier = builder.create<spirv::IMulOp>(
        loc, i64Type, perDimstrideMultiplier, offsets[i]);

    linearizedOffset = builder.create<spirv::IAddOp>(
        loc, i64Type, linearizedOffset, perDimstrideMultiplier);
  }
  return linearizedOffset;
}

unsigned getElementPerWI(imex::xegpu::TensorDescType tDescType) {
  imex::xegpu::SubGroupMapAttr sgMap;
  auto mapping = tDescType.getMapping();

  sgMap = llvm::dyn_cast<imex::xegpu::SubGroupMapAttr>(mapping);

  auto blockSize = tDescType.getShape();
  auto wiLayout = sgMap.getWiLayout();
  auto wiData = sgMap.getWiData();
  unsigned elemPerWI = 1;
  for (int64_t i = 0; i < wiData.size(); i++) {
    if (wiData[i] != 1)
      llvm_unreachable("wi_data must be 1 for all dimension for "
                       "JointMatrix lowering");
    elemPerWI *= (blockSize[i] / wiLayout[i]);
  }
  return elemPerWI;
}

class CreateNdDescToJointMatrix : public OpConversionPattern<CreateNdDescOp> {
public:
  using OpConversionPattern<CreateNdDescOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CreateNdDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op.getTensorDescType().getBoundaryCheck() == false &&
           "for xegpu to joint matrix lowering boundary_check attribute must "
           "be false");
    auto loc = op.getLoc();
    auto tileType = op.getTensorDesc().getType();
    auto rank = tileType.getRank();

    // Set the SPIR-V Struct to represent the Tensor Descriptor
    // The create_nd_tdesc returns a spirv.struct
    // The elements in the struct contains the following elements
    // element 0 = base address pointer : spirv.ptr
    // element 1 = 1D offset : i64
    // element 2 = X Dim Size : i64
    // element 3 = Y Dim Size : i64
    // [SPIR-V lowering uses 1D flattened addresses passed as kernel parameters]
    SmallVector<Type, 4> memberTypes;
    auto i64Type = rewriter.getI64Type();
    // Default storage class is spirv::StorageClass::CrossWorkgroup
    auto spirvStorageClass = spirv::StorageClass::CrossWorkgroup;
    // For memref use memref spirv storage attribute if available
    auto srcType = op.getSourceType();
    if (llvm::isa<mlir::MemRefType>(srcType)) {
      auto sc = dyn_cast_or_null<spirv::StorageClassAttr>(
          llvm::cast<mlir::MemRefType>(srcType).getMemorySpace());
      if (sc)
        spirvStorageClass = sc.getValue();
    }
    auto spirvBaseAddressType =
        spirv::PointerType::get(op.getElementType(), spirvStorageClass);

    memberTypes.push_back(spirvBaseAddressType);
    memberTypes.push_back(i64Type);
    // For nD descriptor, dimesion=rank, so we need dimSize for all the
    // dimensions
    for (int i = 0; i < rank; i++) {
      memberTypes.push_back(i64Type);
    }

    auto ndDescStruct = spirv::StructType::get(memberTypes);

    Value payLoad = rewriter.create<spirv::UndefOp>(loc, ndDescStruct);
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<spirv::ConstantOp>(loc, type, attr);
    };

    // Insert the base address to the ndDescStruct struct
    Value genericBasePtr;
    // If the base type is memref, add a bitcast op
    // If the base type is not memref type, add a ConvertUToPtr op
    if (llvm::isa<mlir::MemRefType>(srcType)) {
      genericBasePtr = rewriter.create<spirv::BitcastOp>(
          loc, spirvBaseAddressType, adaptor.getSource());
    } else {
      genericBasePtr = rewriter.create<spirv::ConvertUToPtrOp>(
          loc, spirvBaseAddressType, adaptor.getSource());
    }

    payLoad = rewriter.create<spirv::CompositeInsertOp>(
        loc, genericBasePtr, payLoad, llvm::ArrayRef(0));

    // TODO: We should be able to use op.getOffsets() directly with index cast
    // But we need support from XeGPU dialect definition to return i64_t

    auto createOffset = [&](unsigned idx) -> Value {
      Value val;
      OpFoldResult ofr = op.getOffsets()[idx];
      auto v = llvm::dyn_cast_if_present<Value>(ofr);
      if (v) {
        val = ofr.get<Value>();
        // Cast index type to i64
        val = rewriter.create<arith::IndexCastOp>(loc, i64Type, val);
      } else {
        int off = llvm::cast<IntegerAttr>(ofr.get<Attribute>()).getInt();
        val = createIntConstant(i64Type, off);
      }
      return val;
    };

    // TODO: We should be able to use op.getShape() directly with index cast
    // But we need support from XeGPU dialect definition to return i64_t

    auto createShape = [&](unsigned idx) -> Value {
      Value val;
      OpFoldResult ofr = op.getShape()[idx];
      auto v = llvm::dyn_cast_if_present<Value>(ofr);
      if (v) {
        val = ofr.get<Value>();
        // Cast index type to i64
        val = rewriter.create<arith::IndexCastOp>(loc, i64Type, val);
      } else {
        int dim = llvm::cast<IntegerAttr>(ofr.get<Attribute>()).getInt();
        val = createIntConstant(i64Type, dim);
      }
      return val;
    };

    SmallVector<Value, 4> nDOffsets;
    SmallVector<Value, 4> nDDims;
    for (unsigned i = 0; i < rank; i++) {
      nDOffsets.push_back(createOffset(i));
    }

    for (unsigned i = 0; i < rank; i++) {
      nDDims.push_back(createShape(i));
    }

    // Calculate the 1-D offset, since the memrefs are flattened when
    // passed to SPIR-V
    Value linearizedOffset = linearizeOffset(rewriter, loc, nDOffsets, nDDims);
    // Insert the flattened (1D) offset to the ndDescStruct struct

    payLoad = rewriter.create<spirv::CompositeInsertOp>(
        loc, linearizedOffset, payLoad, llvm::ArrayRef(1));
    for (int i = 0; i < rank; i++) {
      payLoad = rewriter.create<spirv::CompositeInsertOp>(loc, nDDims[i],
                                                          payLoad, (i + 2));
    }
    rewriter.replaceOp(op, payLoad);
    return success();
  }
};

class UpdateNDOffsetJointMatrix : public OpConversionPattern<UpdateNDOffsetOp> {
public:
  using OpConversionPattern<UpdateNDOffsetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UpdateNDOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto desc = adaptor.getTensorDesc();
    const int dimStartIdx = 2;
    auto i64Type = rewriter.getI64Type();
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<spirv::ConstantOp>(loc, type, attr);
    };
    // Calculate the 1-D offset, since the memrefs are flattened when
    // passed to SPIR-V
    Value offset1D;
    offset1D = createIntConstant(i64Type, 0);
    auto offsets = adaptor.getOffsets();
    auto rank = op.getTensorDesc().getType().getRank();
    // number of offsets & tensorDescriptor rank must be same
    assert(offsets.size() == (size_t)op.getTensorDesc().getType().getRank() &&
           "number of offsets & tensorDescriptor rank must be same");
    for (unsigned i = 0; i < rank; i++) {
      Value perDimstrideMultiplier;
      perDimstrideMultiplier = createIntConstant(i64Type, 1);
      for (unsigned j = i + 1; j < rank; j++) {
        Value dimSize = rewriter.create<spirv::CompositeExtractOp>(
            loc, desc, (j + dimStartIdx));
        perDimstrideMultiplier = rewriter.create<spirv::IMulOp>(
            loc, i64Type, perDimstrideMultiplier, dimSize);
      }
      // Cast index type to i64
      Value offsetVal =
          rewriter.create<arith::IndexCastOp>(loc, i64Type, offsets[i]);
      perDimstrideMultiplier = rewriter.create<spirv::IMulOp>(
          loc, i64Type, perDimstrideMultiplier, offsetVal);

      offset1D = rewriter.create<spirv::IAddOp>(loc, i64Type, offset1D,
                                                perDimstrideMultiplier);
    }

    // Add the newOffset to previous offset
    Value prev1DOffset = rewriter.create<spirv::CompositeExtractOp>(
        loc, desc, llvm::ArrayRef(1));
    offset1D =
        rewriter.create<spirv::IAddOp>(loc, i64Type, offset1D, prev1DOffset);
    // Update the descriptor with the new offset
    desc = rewriter.create<spirv::CompositeInsertOp>(loc, offset1D, desc,
                                                     llvm::ArrayRef(1));
    rewriter.replaceOp(op, desc);
    return success();
  }
};

class LoadNDJointMatrix : public OpConversionPattern<LoadNDOp> {
public:
  using OpConversionPattern<LoadNDOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LoadNDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getTranspose())
      op.emitError("transpose is not currently supported for XeGPU to "
                   "JointMatrix lowering");
    auto loc = op.getLoc();
    auto tDesc = adaptor.getTensorDesc();
    auto tDescType = op.getTensorDesc().getType();
    int rank = tDescType.getRank();
    assert(rank == 2 && "only support 2d load for now");

    // Get the base address
    Value baseAddress = rewriter.create<spirv::CompositeExtractOp>(
        loc, tDesc, llvm::ArrayRef(0));
    // Get the offset
    Value offset = rewriter.create<spirv::CompositeExtractOp>(
        loc, tDesc, llvm::ArrayRef(1));

    SmallVector<Value, 2> linearizedIndices;
    // Get the load address
    Value loadAddress = rewriter.create<spirv::InBoundsPtrAccessChainOp>(
        loc, baseAddress, offset, linearizedIndices);

    // Stride for jointMatrixLoad = Y Dim size
    // TODO: what do we do for transpose case?
    Value stride = rewriter.create<spirv::CompositeExtractOp>(
        loc, tDesc, llvm::ArrayRef(3));

    // Figure out the Matrix Use type (MatrixA, MatrixB, Accumulator)
    uint32_t matrixUse;
    // Don't expect vnni axis to be set for the Accumulator

    if (auto vnniAxis = adaptor.getVnniAxis())
      // vnniAxis 0 -> MatrixB -> matrixUse = 1
      // vnniAxis 1 -> MatrixA -> matrixUse = 0
      matrixUse = (*vnniAxis + 1) % 2;
    else
      // vnniAxis empty -> Accumulator -> matrixUse = 2
      matrixUse = 2;

    // TODO: Need to discuss how to handle transpose, load then transpose or
    // transposed load?
    auto jointMatrixtype = spirv::JointMatrixINTELType::get(
        tDescType.getElementType(), spirv::Scope::Subgroup,
        tDescType.getDimSize(0), tDescType.getDimSize(1),
        spirv::MatrixLayout::RowMajor, *spirv::symbolizeMatrixUse(matrixUse));

    auto jointMatrixLoaded = rewriter.create<spirv::INTELJointMatrixLoadOp>(
        loc, jointMatrixtype, loadAddress, stride,
        ::mlir::spirv::MatrixLayout::RowMajor, ::mlir::spirv::Scope::Subgroup,
        nullptr, nullptr);

    // TODO: Once architecture-spcific info are in place, add subgroup_size
    // restriction verification
    unsigned elemPerWI = getElementPerWI(tDescType);
    auto elemType = tDescType.getElementType();
    auto perWIVectorType = VectorType::get(elemPerWI, elemType);
    Value payLoad = rewriter.create<spirv::UndefOp>(loc, perWIVectorType);
    llvm::SmallVector<Value, 8> extractedVal;
    for (unsigned i = 0; i < elemPerWI; i++) {
      auto idx = createConstantI32(loc, rewriter, i);
      extractedVal.push_back(rewriter.create<spirv::VectorExtractDynamicOp>(
          loc, jointMatrixLoaded, idx));
    }

    // Putting all the extract and insert operations together, may make it
    // easier for compiler (IGC) to reason about
    for (unsigned i = 0; i < elemPerWI; i++) {
      auto idx = createConstantI32(loc, rewriter, i);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(
          loc, payLoad, extractedVal[i], idx);
    }
    rewriter.replaceOp(op, payLoad);
    return success();
  }
};

class StoreNDJointMatrix : public OpConversionPattern<StoreNDOp> {
public:
  using OpConversionPattern<StoreNDOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(StoreNDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tDesc = adaptor.getTensorDesc();
    auto tDescType = op.getTensorDesc().getType();
    int rank = tDescType.getRank();
    assert(rank == 2 && "only support 2d load for now");

    // Get the base address
    Value baseAddress = rewriter.create<spirv::CompositeExtractOp>(
        loc, tDesc, llvm::ArrayRef(0));
    // Get the offset
    Value offset = rewriter.create<spirv::CompositeExtractOp>(
        loc, tDesc, llvm::ArrayRef(1));

    SmallVector<Value, 2> linearizedIndices;
    // Get the load address
    Value loadAddress = rewriter.create<spirv::InBoundsPtrAccessChainOp>(
        loc, baseAddress, offset, linearizedIndices);

    // Stride for jointMatrixLoad = Y Dim size
    // TODO: what do we do for transpose case?
    Value stride = rewriter.create<spirv::CompositeExtractOp>(
        loc, tDesc, llvm::ArrayRef(3));

    // For Store, we only allow Accumulator type matrix to store.
    // TODO: We need to Add option on the xegpu.store_nd to support storing B
    // matrix for that we need to add vnni_axis attribute to store_nd op as
    // well.
    uint32_t matrixUse = 2;
    // Don't expect vnni axis to be set for the Accumulator
    auto jointMatrixtype = spirv::JointMatrixINTELType::get(
        tDescType.getElementType(), spirv::Scope::Subgroup,
        tDescType.getDimSize(0), tDescType.getDimSize(1),
        spirv::MatrixLayout::RowMajor, *spirv::symbolizeMatrixUse(matrixUse));
    Value matrix = rewriter.create<spirv::UndefOp>(loc, jointMatrixtype);

    // TODO: Once architecture-spcific info are in place, add subgroup_size
    // restriction verification
    unsigned elemPerWI = getElementPerWI(tDescType);
    // auto elemType = tDescType.getElementType();
    // Get the 2D vector
    auto perWIVector = adaptor.getValue();
    llvm::SmallVector<Value, 8> extractedVal;
    for (unsigned i = 0; i < elemPerWI; i++) {
      auto idx = createConstantI32(loc, rewriter, i);
      extractedVal.push_back(rewriter.create<spirv::VectorExtractDynamicOp>(
          loc, perWIVector, idx));
    }

    // Putting all the extract and insert operations together, may make it
    // easier for compiler (IGC) to reason about
    for (unsigned i = 0; i < elemPerWI; i++) {
      auto idx = createConstantI32(loc, rewriter, i);
      matrix = rewriter.create<spirv::VectorInsertDynamicOp>(
          loc, matrix, extractedVal[i], idx);
    }
    auto payLoad = rewriter.create<spirv::INTELJointMatrixStoreOp>(
        loc, loadAddress, matrix, stride, ::mlir::spirv::MatrixLayout::RowMajor,
        ::mlir::spirv::Scope::Subgroup, nullptr, nullptr);
    rewriter.replaceOp(op, payLoad);
    return success();
  }
};

class DpasJointMatrix : public OpConversionPattern<DpasOp> {
public:
  using OpConversionPattern<DpasOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(DpasOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto vectorA = op.getLhs();
    auto vectorB = op.getRhs();
    auto vectorC = op.getAcc();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto func = op->getParentOfType<spirv::FuncOp>();
      rewriter.setInsertionPointAfter(func);
      rewriter.create<spirv::ExecutionModeOp>(
          op.getLoc(), func, spirv::ExecutionMode::SubgroupSize,
          int(jointMatrixSubGroupSize));
    }
    // Matrix row = 1st dim of input vector
    // Matrix colomn = 2nd dim of input vector * jointMatrixSubGroupSize
    auto matrixAType = spirv::JointMatrixINTELType::get(
        vectorA.getType().getElementType(), spirv::Scope::Subgroup,
        vectorA.getType().getShape()[0],
        vectorA.getType().getShape()[1] * jointMatrixSubGroupSize,
        spirv::MatrixLayout::RowMajor, spirv::MatrixUse::MatrixA);

    // B matrix vector is passed VNNI-transformed, so row = dim0 *dim3
    auto matrixBType = spirv::JointMatrixINTELType::get(
        vectorB.getType().getElementType(), spirv::Scope::Subgroup,
        vectorB.getType().getShape()[0] * vectorB.getType().getShape()[2],
        vectorB.getType().getShape()[1] * jointMatrixSubGroupSize,
        spirv::MatrixLayout::RowMajor, spirv::MatrixUse::MatrixB);

    auto matrixCType = spirv::JointMatrixINTELType::get(
        vectorC.getType().getElementType(), spirv::Scope::Subgroup,
        vectorC.getType().getShape()[0],
        vectorC.getType().getShape()[1] * jointMatrixSubGroupSize,
        spirv::MatrixLayout::RowMajor, spirv::MatrixUse::Accumulator);

    Value matrixA = rewriter.create<spirv::UndefOp>(loc, matrixAType);
    Value matrixB = rewriter.create<spirv::UndefOp>(loc, matrixBType);
    Value matrixC = rewriter.create<spirv::UndefOp>(loc, matrixCType);
    // Create Matrices from the vectors
    // Get the flattened vectors through the adaptor, since SPIRV only allows 1D
    // vector
    auto perWIVectorA = adaptor.getLhs();
    auto perWIVectorB = adaptor.getRhs();
    auto perWIVectorC = adaptor.getAcc();

    llvm::SmallVector<Value, 8> extractedValA;
    auto perWIelemsA =
        llvm::cast<mlir::VectorType>(perWIVectorA.getType()).getNumElements();
    for (unsigned i = 0; i < perWIelemsA; i++) {
      auto idx = createConstantI32(loc, rewriter, i);
      extractedValA.push_back(rewriter.create<spirv::VectorExtractDynamicOp>(
          loc, perWIVectorA, idx));
    }
    // Putting all the extract and insert operations together, may make it
    // easier for compiler (IGC) to reason about
    for (unsigned i = 0; i < perWIelemsA; i++) {
      auto idx = createConstantI32(loc, rewriter, i);
      matrixA = rewriter.create<spirv::VectorInsertDynamicOp>(
          loc, matrixA, extractedValA[i], idx);
    }

    llvm::SmallVector<Value, 8> extractedValB;
    auto perWIelemsB =
        llvm::cast<mlir::VectorType>(perWIVectorB.getType()).getNumElements();
    for (unsigned i = 0; i < perWIelemsB; i++) {
      auto idx = createConstantI32(loc, rewriter, i);
      extractedValB.push_back(rewriter.create<spirv::VectorExtractDynamicOp>(
          loc, perWIVectorB, idx));
    }
    // Putting all the extract and insert operations together, may make it
    // easier for compiler (IGC) to reason about
    for (unsigned i = 0; i < perWIelemsB; i++) {
      auto idx = createConstantI32(loc, rewriter, i);
      matrixB = rewriter.create<spirv::VectorInsertDynamicOp>(
          loc, matrixB, extractedValB[i], idx);
    }

    llvm::SmallVector<Value, 8> extractedValC;
    auto perWIelemsC =
        llvm::cast<mlir::VectorType>(perWIVectorC.getType()).getNumElements();
    for (unsigned i = 0; i < perWIelemsC; i++) {
      auto idx = createConstantI32(loc, rewriter, i);
      extractedValC.push_back(rewriter.create<spirv::VectorExtractDynamicOp>(
          loc, perWIVectorC, idx));
    }
    // Putting all the extract and insert operations together, may make it
    // easier for compiler (IGC) to reason about
    for (unsigned i = 0; i < perWIelemsC; i++) {
      auto idx = createConstantI32(loc, rewriter, i);
      matrixC = rewriter.create<spirv::VectorInsertDynamicOp>(
          loc, matrixC, extractedValC[i], idx);
    }

    Value result = rewriter.create<spirv::INTELJointMatrixMadOp>(
        loc, matrixA, matrixB, matrixC, spirv::Scope::Subgroup);

    Value payLoad =
        rewriter.create<spirv::UndefOp>(loc, perWIVectorC.getType());
    llvm::SmallVector<Value, 8> extractedValResult;
    auto perWIelemsResult = perWIelemsC;
    for (unsigned i = 0; i < perWIelemsResult; i++) {
      auto idx = createConstantI32(loc, rewriter, i);
      extractedValResult.push_back(
          rewriter.create<spirv::VectorExtractDynamicOp>(loc, result, idx));
    }
    for (unsigned i = 0; i < perWIelemsResult; i++) {
      auto idx = createConstantI32(loc, rewriter, i);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(
          loc, payLoad, extractedValResult[i], idx);
    }
    rewriter.replaceOp(op, payLoad);
    return success();
  }
};

} // namespace

void imex::populateXeGPUToJointMatrixPatterns(SPIRVTypeConverter &typeConverter,
                                              RewritePatternSet &patterns) {
  patterns.add<CreateNdDescToJointMatrix, UpdateNDOffsetJointMatrix,
               LoadNDJointMatrix, StoreNDJointMatrix, DpasJointMatrix,
               VectorShapeCast, VectorExtract, VectorExtractStridedSlice,
               VectorShuffle>(typeConverter, patterns.getContext());
}
