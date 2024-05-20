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

#include "../PassDetail.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/AddDiscriminators.h"

#include <cassert>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Debug.h>

#include "imex/Utils/XeCommon.h"
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
#include <type_traits>

using namespace imex;
using namespace mlir;
using namespace mlir::xegpu;

namespace {
/// @brief
/// We have to use i32 for intrinsic calls like llvm_genx_raw_send2_*, if we
/// want to get the original element type (e.g., f16) as the result of a load,
/// we have to encode the resulting i32 vector back to it.
VectorType encodeVectorTypeTo(VectorType currentVecType, Type toElemType) {
  auto elemType = currentVecType.getElementType();
  auto currentbitWidth = elemType.getIntOrFloatBitWidth();
  auto newBitwidth = toElemType.getIntOrFloatBitWidth();
  const int size =
      currentVecType.getNumElements() * currentbitWidth / newBitwidth;
  return VectorType::get(size, toElemType);
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
        surfaceW = rewriter.create<spirv::UConvertOp>(loc, i32Type,
                                                      adaptor.getShape()[1]);
        surfaceW = rewriter.create<spirv::IMulOp>(loc, surfaceW, bytesPerElem);
        surfaceW = rewriter.create<spirv::ISubOp>(loc, surfaceW, one);
        // compute surface height
        surfaceH = rewriter.create<spirv::UConvertOp>(loc, i32Type,
                                                      adaptor.getShape()[0]);
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
        OpFoldResult ofr = op.getMixedOffsets()[idx];
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
      int array_length = op.getType().getArrayLength();
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

class UpdateNDOffsetToVCPattern : public OpConversionPattern<UpdateNdOffsetOp> {
public:
  using OpConversionPattern<UpdateNdOffsetOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UpdateNdOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i32Type = rewriter.getI32Type();
    auto offsets = adaptor.getOffsets();
    auto desc = adaptor.getTensorDesc();
    for (size_t i = 0; i < offsets.size(); i++) {
      auto offset = offsets[i];
      if (auto cst = offset.getDefiningOp<arith::ConstantOp>())
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
    constexpr bool isLoad = std::is_same_v<OpType, LoadNdOp>;
    constexpr bool isPrefetch = std::is_same_v<OpType, PrefetchNdOp>;
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
        // Intrinsic accepts and returns i32 type, but we want to return a
        // vector of the original element type
        auto loadResultInOrigType =
            encodeVectorTypeTo(retType, tileType.getElementType());
        if (loadResultInOrigType != funcOp->getResult(0).getType()) {
          auto cast = rewriter.create<spirv::BitcastOp>(
              loc, loadResultInOrigType, funcOp->getResult(0));
          rewriter.replaceOp(op, cast);
        } else {
          rewriter.replaceOp(op, funcOp);
        }
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

#if 0
std::optional<xegpu::CreateNdDescOp> findDescOp(mlir::Value val) {
  if (auto op = val.getDefiningOp()) {
    if (auto descOp = dyn_cast<xegpu::CreateNdDescOp>(op)) {
      return descOp;
    } else if (auto update = dyn_cast<xegpu::UpdateNdOffsetOp>(op)) {
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
#endif

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
    constexpr bool isLoad = std::is_same_v<OpType, LoadNdOp>;
    constexpr bool isPrefetch = std::is_same_v<OpType, PrefetchNdOp>;
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

    // TODO: currently limit transposeBitWidth to 32, it is
    // an architecture feature, and 32 works on PVC but may
    // be not FS. To support other bits, we cannot hardcode
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
        // Intrinsic accepts and returns i32 type, but we want to return a
        // vector of the original element type
        auto loadResultInOrigType = encodeVectorTypeTo(newType, elmType);
        if (loadResultInOrigType != funcOp->getResult(0).getType()) {
          auto cast = rewriter.create<spirv::BitcastOp>(
              loc, loadResultInOrigType, funcOp->getResult(0));
          rewriter.replaceOp(op, cast);
        } else {
          rewriter.replaceOp(op, funcOp);
        }
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
    auto lhsType = mlir::cast<VectorType>(op.getLhs().getType());
    auto rhsType = mlir::cast<VectorType>(op.getRhs().getType());
    auto resultType = mlir::cast<VectorType>(op.getResultType());
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

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    // Intrinsic accepts i32 type, therefore the element type should be casted
    // to i32
    auto [lhsName, lhsNewType] = encodeVectorType(rewriter, lhsType);
    auto [rhsName, rhsNewType] = encodeVectorType(rewriter, rhsType);
    auto [resultName, newResultType] = encodeVectorType(rewriter, resultType);

    if (lhsNewType != adaptor.getLhs().getType()) {
      lhs =
          rewriter.create<spirv::BitcastOp>(loc, lhsNewType, adaptor.getLhs());
    }
    if (rhsNewType != adaptor.getRhs().getType()) {
      rhs =
          rewriter.create<spirv::BitcastOp>(loc, rhsNewType, adaptor.getRhs());
    }
    SmallVector<Value, 4> args{rhs, lhs, info};
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
      args.assign({adaptor.getAcc(), rhs, lhs, prec1Arg, prec2Arg, sdArg, rcArg,
                   signless, signless});
    }
    funcName += resultName;
    funcName += "_";
    funcName += rhsName;
    funcName += "_";
    funcName += lhsName;
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
        op.getNbarrierNum());
    rewriter.eraseOp(op);
    return success();
  }
};

class InitNbarrierToVCPattern : public OpConversionPattern<InitNbarrierOp> {
public:
  using OpConversionPattern<InitNbarrierOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(InitNbarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto i32Type = rewriter.getIntegerType(32);
    auto v8i32Type = mlir::VectorType::get(8, i32Type);

    auto loc = op.getLoc();
    auto nbarrier_id = op.getNbarrierId();

    // a participant is both a producer or a consumer (0)
    auto nbarrier_role = rewriter.create<arith::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(0));
    auto num_participants = zext(i32Type, op.getParticipantThreadNum());

    DenseElementsAttr constantData = DenseElementsAttr::get(
        v8i32Type, ArrayRef<int>(std::vector<int>(1, 0)));
    Value nbarrier_src =
        rewriter.create<spirv::ConstantOp>(loc, v8i32Type, constantData);

    Value payload = zext(i32Type, nbarrier_id);

    Value payload_nbarrier_role =
        logic_shl(i32Type, nbarrier_role, i32_val(14));
    payload = bitwise_or(i32Type, payload, payload_nbarrier_role);

    Value payload_num_producers =
        logic_shl(i32Type, num_participants, i32_val(16));
    payload = bitwise_or(i32Type, payload, payload_num_producers);

    Value payload_num_consumers =
        logic_shl(i32Type, num_participants, i32_val(24));
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
    auto payload = adaptor.getNbarrier();

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
    auto payload = adaptor.getNbarrier();

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

class CompileHintToVCPattern : public OpConversionPattern<CompileHintOp> {
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

class FenceToVCPattern : public OpConversionPattern<FenceOp> {
public:
  using OpConversionPattern<FenceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto pred = i1_val(1);
    uint8_t fence_op, fence_scope, sfid;

    enum lscFenceOp {
      NONE = 0,
      EVICT = 1,
      INVALIDATE = 2,
      DISCARD = 3,
      CLEAN = 4,
      FLUSHL3 = 5
    };
    enum lscFenceScope {
      GROUP = 0,
      LOCAL = 1,
      TILE = 2,
      GPU = 3,
      GPUS = 4,
      SYSTEM = 5,
      SYSACQ = 6
    };
    enum lscSFID { UGM = 0, UGML = 1, TGM = 3, SLM = 4 };

    // the design limits the fence_op to NONE
    fence_op = lscFenceOp::NONE;

    switch (op.getMemoryKind()) {
    case mlir::xegpu::MemoryScope::Global:
      sfid = lscSFID::UGM;
      break;
    case mlir::xegpu::MemoryScope::SLM:
      sfid = lscSFID::TGM;
      break;
    }

    switch (op.getFenceScope()) {
    case mlir::xegpu::FenceScope::Workgroup:
      fence_scope = lscFenceScope::GROUP;
      break;
    case mlir::xegpu::FenceScope::GPU:
      fence_scope = lscFenceScope::GPU;
      break;
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
}

struct VectorExtractStridedSlice final
    : public OpConversionPattern<vector::ExtractStridedSliceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(vector::ExtractStridedSliceOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = getTypeConverter()->convertType(extractOp.getType());
    if (!dstType)
      return failure();

    auto offsets = extractOp.getOffsets().getValue();
    auto sizes = extractOp.getSizes().getValue();
    auto strides = extractOp.getStrides().getValue();

    if (mlir::cast<IntegerAttr>(strides[0]).getInt() != 1)
      return rewriter.notifyMatchFailure(
          extractOp, "Strided slice with stride != 1 is not supported.");

    Value srcVector = adaptor.getOperands().front();

    // Extract vector<1xT> case.
    if (isa<spirv::ScalarType>(dstType)) {
      uint64_t offset = getFirstIntValue(extractOp.getOffsets());
      rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(extractOp,
                                                             srcVector, offset);
      return success();
    }

    // if kD offsets are specified for nd source vector (n > k), the granularity
    // of the extraction is greater than 1. In this case last (n-k) dimensions
    // form the extraction granularity. example : %0 =
    // vector.extract_strided_slice %src { offsets = [0, 0], sizes = [2, 2],
    // strides = [1, 1]} : vector<4x8x8xf32> to vector<2x2x8xf32>
    // here, extraction granularity is 8.
    int64_t extractSliceLen = 1;
    auto n = extractOp.getSourceVectorType().getRank();
    auto k = (int64_t)offsets.size();
    if (n > k) {
      for (unsigned i = 0; i < n - k; i++) {
        extractSliceLen *= extractOp.getSourceVectorType().getShape()[i + k];
      }
    }

    // get total number of extracted slices
    int64_t nExtractedSlices = 1;
    for (auto size : sizes) {
      nExtractedSlices *= mlir::cast<IntegerAttr>(size).getInt();
    }

    // compute the strides of the source vector considering first k dimensions
    SmallVector<int32_t, 4> sourceStrides(k, extractSliceLen);
    for (int i = k - 2; i >= 0; --i) {
      sourceStrides[i] = sourceStrides[i + 1] *
                         extractOp.getSourceVectorType().getShape()[i + 1];
    }
    // final shuffle indices has nExtractedElems * extractSliceLen elements
    SmallVector<int32_t, 4> indices(nExtractedSlices * extractSliceLen);
    // compute the strides of the extracted kD vector
    SmallVector<int32_t, 4> extractedStrides(k, 1);
    // compute extractedStrides
    for (int i = k - 2; i >= 0; --i) {
      extractedStrides[i] = extractedStrides[i + 1] *
                            mlir::cast<IntegerAttr>(sizes[i + 1]).getInt();
    }
    // iterate over all extracted slices from 0 to nExtractedElems-1
    // and compute the multi-dimensional index and the corresponding linearized
    // index within the source vector
    for (int64_t i = 0; i < nExtractedSlices; ++i) {
      int64_t index = i;
      // compute the corresponding multi-dimensional index
      SmallVector<int32_t, 4> multiDimIndex(k, 0);
      for (int64_t j = 0; j < k; ++j) {
        multiDimIndex[j] = (index / extractedStrides[j]);
        index -= multiDimIndex[j] * extractedStrides[j];
      }
      // compute the corresponding linearized index in the source vector
      // i.e. shift the multiDimIndex by the offsets
      int64_t linearizedIndex = 0;
      for (int64_t j = 0; j < k; ++j) {
        linearizedIndex +=
            (mlir::cast<IntegerAttr>(offsets[j]).getInt() + multiDimIndex[j]) *
            sourceStrides[j];
      }
      // fill the indices array form linearizedIndex to linearizedIndex +
      // sliceLen
      for (int64_t j = 0; j < extractSliceLen; ++j) {
        indices[i * extractSliceLen + j] = linearizedIndex + j;
      }
    }
    // perform a shuffle to extract the kD vector
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

template <typename SPIRVOp> std::string getVCIntrinsicName() {
  constexpr bool isFMaxOp = std::is_same_v<SPIRVOp, spirv::CLFMaxOp>;
  constexpr bool isExpOp = std::is_same_v<SPIRVOp, spirv::CLExpOp>;
  if (isFMaxOp)
    return "llvm.genx.fmax.";
  else if (isExpOp)
    return "llvm.genx.exp.";
  else
    assert(0 && "Unsupported SPIRV Op. Add more support!");
}

template <typename SPIRVOp>
struct SPIRVElementwiseToVC : public OpConversionPattern<SPIRVOp> {
  using OpConversionPattern<SPIRVOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SPIRVOp op, typename SPIRVOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = this->getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return failure();

    // This lowering pattern is needed only for spirv ops with large vector
    // lengths.
    assert(
        !imex::isGenericVectorTy(dstType) &&
        "Vector size is considered generic and op does not require lowering to "
        "VC intrinsic. Consider marking this op + vector length as legal.");
    auto vecSize = mlir::dyn_cast<VectorType>(dstType).getNumElements();
    // for larger vector lengths, "llvm.genx.exp" returns the base 2
    // exponentiation of the input. To get the base e exponentiation, we need to
    // scale the input by log2(e)
    bool isExpOp = std::is_same_v<SPIRVOp, spirv::CLExpOp>;
    SmallVector<Value> args{adaptor.getOperands()};
    auto operands = adaptor.getOperands();
    if (isExpOp) {
      SmallVector<float> log2e(vecSize, 1.442695040888963);
      auto log2eConstVec = rewriter.create<spirv::ConstantOp>(
          op.getLoc(), dstType, rewriter.getF32VectorAttr(log2e));
      auto input = operands[0];
      auto scaledInput =
          rewriter.create<spirv::FMulOp>(op.getLoc(), input, log2eConstVec);
      args.clear();
      args.push_back(scaledInput);
    }

    // for large vectors, generate the corresponding VC intrinsic.
    auto funcName = getVCIntrinsicName<SPIRVOp>();
    SmallVector<Type> operandTypes;
    for (auto operand : adaptor.getOperands())
      operandTypes.push_back(operand.getType());
    auto funcType = rewriter.getFunctionType(operandTypes, {dstType});
    funcName +=
        encodeVectorType(rewriter, mlir::dyn_cast<VectorType>(dstType)).first;
    lookupOrInsertIntrinsic(rewriter, op, funcName, funcType);

    rewriter.replaceOpWithNewOp<spirv::FunctionCallOp>(op, dstType, funcName,
                                                       args);
    return success();
  }
};
} // namespace

bool imex::isGenericVectorTy(mlir::Type type) {
  if (isa<spirv::ScalarType>(type))
    return true;
  auto vecSize = mlir::dyn_cast<VectorType>(type).getNumElements();
  return vecSize == 2 || vecSize == 3 || vecSize == 4 || vecSize == 8 ||
         vecSize == 16;
}

void imex::populateXeGPUToVCIntrinsicsPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<CreateNdDescToSPIRV, CreateDescToVCPattern, DpasToVCPattern,
               AllocNbarrierToVCPattern, InitNbarrierToVCPattern,
               NbarrierArriveToVCPattern, NbarrierWaitToVCPattern,
               CompileHintToVCPattern, FenceToVCPattern, VectorShapeCast,
               VectorExtract, VectorExtractStridedSlice, VectorShuffle,
               SPIRVElementwiseToVC<spirv::CLFMaxOp>,
               SPIRVElementwiseToVC<spirv::CLExpOp>,
               GatherScatterToRawSend<LoadGatherOp>,
               GatherScatterToRawSend<StoreScatterOp>, AtomicToLsc,
               UpdateNDOffsetToVCPattern>(typeConverter, patterns.getContext());
  if (getenv("IMEX_NOT_PREFER_RAWSEND"))
    patterns.add<LoadStorePrefetchNdToLsc<LoadNdOp>,
                 LoadStorePrefetchNdToLsc<StoreNdOp>,
                 LoadStorePrefetchNdToLsc<PrefetchNdOp>>(typeConverter,
                                                         patterns.getContext());
  else
    patterns.add<LoadStorePrefetchNdToRawSend<LoadNdOp>,
                 LoadStorePrefetchNdToRawSend<StoreNdOp>,
                 LoadStorePrefetchNdToRawSend<PrefetchNdOp>>(
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
    constexpr bool isLoad = std::is_same_v<OpType, LoadNdOp>;
    constexpr bool isPrefetch = std::is_same_v<OpType, PrefetchNdOp>;
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
    auto lType = mlir::cast<VectorType>(op.getLhs().getType());
    auto rType = mlir::cast<VectorType>(op.getRhs().getType());
    auto resultType = mlir::cast<VectorType>(op.getResultType());
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
               LoadStorePrefetchNdToGenISA<LoadNdOp>,
               LoadStorePrefetchNdToGenISA<StoreNdOp>,
               LoadStorePrefetchNdToGenISA<PrefetchNdOp>>(
      typeConverter, patterns.getContext());
}
