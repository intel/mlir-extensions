//===- LSCPatterns.cpp -  XeGPU to VC Lowering pass  ------------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements patterns used by XeGPUToVC to lower XeGPU ops into
/// function calls to LSC intrinsics.
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/XeGPUToVC/XeGPUToVC.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>

#include "LscIntrinsicEnums.h"
#include "imex/Utils/VCUtils.h"

using namespace mlir;
using mlir::xegpu::AtomicRMWOp;
using mlir::xegpu::FenceOp;
using mlir::xegpu::LoadGatherOp;
using mlir::xegpu::LoadNdOp;
using mlir::xegpu::PrefetchNdOp;
using mlir::xegpu::PrefetchOp;
using mlir::xegpu::StoreNdOp;
using mlir::xegpu::StoreScatterOp;

namespace imex {

namespace LSC {

static int getCacheEncoding(std::optional<xegpu::CachePolicy> hint) {

  if (!hint.has_value())
    return LSC_CACHING_DEFAULT;

  switch (hint.value()) {
  case xegpu::CachePolicy::CACHED:
    return LSC_CACHING_CACHED;

  case xegpu::CachePolicy::UNCACHED:
    return LSC_CACHING_UNCACHED;

  case xegpu::CachePolicy::STREAMING:
    return LSC_CACHING_STREAMING;

  case xegpu::CachePolicy::READ_INVALIDATE:
    return LSC_CACHING_READINVALIDATE;

  case xegpu::CachePolicy::WRITE_BACK:
    return LSC_CACHING_WRITEBACK;

  case xegpu::CachePolicy::WRITE_THROUGH:
    return LSC_CACHING_WRITETHROUGH;
  }

  return LSC_CACHING_DEFAULT;
}

// lsc intrinsic for load and store only works on 32-bit data,
// so we need to extend 8-bit/16-bit data into 32-bit data.
// For example, Vector<16x32xf16> will be Vector<16x32xi32>
// and Vector<8x16xi8> will be Vector<8x16xi32>
// Note: only i32 is the valid way, using f32 is incorrect.
static VectorType getOrigOrI32VectorType(VectorType type) {
  auto elemTy = type.getElementType();
  if (!elemTy.isIntOrFloat())
    return type;

  auto bitWidth = elemTy.getIntOrFloatBitWidth();
  if (bitWidth >= 32)
    return type;

  auto context = type.getContext();
  return VectorType::get(type.getShape(), IntegerType::get(context, 32));
}

// It simply linearizes the N-D vector into 1-D vector, e.g.,
// Vector<8x16xf32> to Vector<128xf32>, and also return a mangled string
// "v128f32".
// TODO: move this into typeConverter, but currently it seems have some
// conflicts with the implementation for RawSend.
static std::pair<std::string, VectorType> convertVectorType(VectorType type) {
  auto elemTy = type.getElementType();
  assert(elemTy.isIntOrFloat() && "unsupported element type");
  auto numElems = type.getNumElements();
  auto bitWidth = elemTy.getIntOrFloatBitWidth();
  auto prefix = elemTy.isInteger() ? "i" : elemTy.isBF16() ? "bf" : "f";
  auto str = llvm::formatv("v{0}{1}{2}", numElems, prefix, bitWidth).str();
  return std::make_pair(str, mlir::VectorType::get(numElems, elemTy));
}

static LogicalResult isValid1DBlockSetup(Type elemTy, int elems, Location &loc,
                                         ConversionPatternRewriter &rewriter) {
  if (!elemTy.isIntOrFloat())
    return failure();

  auto bitWidth = elemTy.getIntOrFloatBitWidth();

  if (bitWidth < 32)
    return rewriter.notifyMatchFailure(loc, "only 32-bit data supported.");

  auto validChunkSizes = getSupportedChunkSizes(1);
  if (!llvm::is_contained(validChunkSizes, elems))
    return rewriter.notifyMatchFailure(
        loc, "invalid number of elements. Supports 1, 2, 3, 4, 8, 16, 32, 64.");

  auto totalSize = elems * bitWidth / 8; // in bytes
  const int constraint = 512;            // up to 512 bytes
  if (totalSize > constraint)
    return rewriter.notifyMatchFailure(loc, "max total size is 512 bytes.");

  return success();
}

static LogicalResult isValidScatterSetup(Type elemTy, int simd_lanes,
                                         int chunk_size, Location &loc,
                                         ConversionPatternRewriter &rewriter) {
  if (!elemTy.isIntOrFloat())
    return failure();

  // TODO: temporarily disable simd_lanes check, it is fine for SIMD pipeline
  // but may be not compatible with SIMT pipeline.
  // if (simd_lanes != 16 && simd_lanes != 32)
  //   return rewriter.notifyMatchFailure(
  //       loc, "A valid simd lane is 16 or 32 for PVC.");

  auto validChunkSizes = getSupportedChunkSizes(simd_lanes);
  if (!llvm::is_contained(validChunkSizes, chunk_size))
    return rewriter.notifyMatchFailure(
        loc, "invalid chunk size. Supports 1, 2, 3, 4, 8"
             "(and 16, 32, 64 if simd_lanes == 1).");

  auto bitWidth = elemTy.getIntOrFloatBitWidth();

  // for 8b and 16b data, the hardware only support vector size of 1
  if ((bitWidth == 8 || bitWidth == 16) && chunk_size != 1)
    return rewriter.notifyMatchFailure(
        loc, "only chunk size 1 supported for 8b/16b data.");

  auto total_size = chunk_size * simd_lanes * bitWidth / 8; // in bytes
  const int constraint = 512;                               // up to 512 bytes
  if (total_size > constraint)
    return rewriter.notifyMatchFailure(loc, "max total size is 512 bytes.");

  return success();
}

// a helper function to get complete instrinsic name for
// lsc.load/store/prefetch. The fullname is in format of
//    1. lsc.load.[slm|stateless].<retType>.<predType>.<offsetType>
//    2. lsc.store.[slm|stateless].<predType>.<offsetType>.<dataType>
//    3. lsc.prefetch.stateless.<predType>.<offsetType>
// All the types are encoded as vN[i/f]M, where N is the number of elements,
// and M is the bit width. So for vector<16xf32>, it will be v16f32, and for
// vector<16xi1>, it will be v16i1. dataTyStr is used for the result of load,
// or the data for store. It is not used for prefetch. prefetch on slm is not
// available.
static std::string getLSCIntrinsicStr(llvm::StringRef opName, int simd_lanes,
                                      xegpu::MemorySpace MemorySpace,
                                      llvm::StringRef dataTyStr = "") {
  auto kind = MemorySpace == xegpu::MemorySpace::SLM ? "slm" : "stateless";
  // using 32bit for slm and 64bit for stateless
  auto addrBits = MemorySpace == xegpu::MemorySpace::SLM ? 32 : 64;
  auto predTyStr = llvm::formatv("v{0}i1", simd_lanes).str();
  auto offsetTyStr = llvm::formatv("v{0}i{1}", simd_lanes, addrBits).str();
  if (opName == "load") {
    return llvm::formatv("llvm.genx.lsc.{0}.{1}.{2}.{3}.{4}", opName, kind,
                         dataTyStr, predTyStr, offsetTyStr)
        .str();
  } else if (opName == "store") {
    return llvm::formatv("llvm.genx.lsc.{0}.{1}.{2}.{3}.{4}", opName, kind,
                         predTyStr, offsetTyStr, dataTyStr)
        .str();
  } else if (opName == "prefetch") {
    kind = "stateless";
    return llvm::formatv("llvm.genx.lsc.{0}.{1}.{2}.{3}", opName, kind,
                         predTyStr, offsetTyStr)
        .str();
  }

  llvm_unreachable("unsupported opName");
  return "";
}

// a helper function to get complete instrinsic name for
// lsc.load/store/prefetch.2d.ugm. The fullname is in format of
//    1. lsc.load.2d.ugm.desc.<transform>.<retType>.<cache_controls>
//    2. lsc.store.2d.ugm.desc.<cacheCtrType>.<dataType>
//    3. lsc.prefetch.2d.ugm.desc.<predType>.<dataType>
// All the types are encoded as vN[i/f]M, where N is the number of elements,
// and M is the bit width. So for vector<16xf32>, it will be v16f32, and for
// vector<16xi1>, it will be v16i1. cacheCtrType is fixed to vNi8, where N is
// number of cache levels, with each cache level setting (cache, writeback,
// etc). is encoded as an i8. dataTyStr is used for the result of load, or the
// data for store. It is not used for prefetch. transform is used for load, it
// can be either "vnni" or "transpose".
static std::string getBlockIntrinsicStr(llvm::StringRef opName,
                                        llvm::StringRef dataTyStr = "",
                                        llvm::StringRef transform = "",
                                        int cache_levels = 2) {
  if (opName == "load") {
    if (transform.empty()) {
      // encode result type and cache controls
      return llvm::formatv("llvm.genx.lsc.load.2d.ugm.desc.{0}.v{1}i8",
                           dataTyStr, cache_levels)
          .str();
    } else {
      // encode transform, result type and cache controls
      return llvm::formatv("llvm.genx.lsc.load.2d.ugm.desc.{0}.{1}.v{2}i8",
                           transform, dataTyStr, cache_levels)
          .str();
    }
  } else if (opName == "store") {
    return llvm::formatv("llvm.genx.lsc.store.2d.ugm.desc.v{0}i8.{1}",
                         cache_levels, dataTyStr)
        .str();
  } else if (opName == "prefetch") {
    return llvm::formatv("llvm.genx.lsc.prefetch.2d.ugm.desc.v{0}i8.{1}",
                         cache_levels, dataTyStr)
        .str();
  }
  llvm_unreachable("unsupported opName");
  return "";
}

// A helper function to create a call to a lsc.load/store/prefetch.* intrinsics.
// These set of intrinsics shares similar arguments, except that the store
// requires an additional data argument. Its arguments are aligned with the lsc
// spec.
static func::CallOp genRawLSCIntrinsicCall(
    ConversionPatternRewriter &rewriter, Location &loc, StringRef intrinsicStr,
    TypeRange resultType, TypedValue<VectorType> pred, enum LSC_OP opCode,
    enum LSC_CACHE_OPT l1, enum LSC_CACHE_OPT l3, short addrScale,
    int immediateOffset, enum LSC_DATA_SIZE dataumSize,
    enum LSC_DATA_ELEMS vectSize, enum LSC_DATA_ORDER transpose,
    int channel_mask, Value addresses, int surface = 0, Value data = {}) {
  // arg0: vNi1 (N = 1, 16 or 32) predicate, use argument directly
  auto predTy = pred.getType();
  auto elemTy = predTy.getElementType();
  assert(predTy.getRank() == 1 && "predicate must be a 1D vector type.");
  assert(elemTy.isInteger(1) && "predicate type must be i1.");
  // TODO: temporarily disable predicate_size check. It is
  // fine for SIMD pipeline but may not match SIMT pipeline.
  //
  // assert(llvm::is_contained({1, 16, 32}, predTy.getNumElements()) &&
  //        "predicate size must be 1, 16 or 32.");

  // arg1: i8 subopcode, LSC_LOAD for load/prefetch, LSC_STORE for store
  assert((opCode == LSC_LOAD || opCode == LSC_STORE) && "unsupported opcode.");
  auto opCodeEncode = i8_val(opCode);

  // arg2: i8, Cache behavior for L1
  auto l1Encode = i8_val(l1);

  // arg3: i8, Cache behavior for L2
  auto l3Encode = i8_val(l3);

  // arg4: i8, Address scale
  auto addrScaleEncode = i16_val(addrScale);

  // arg5: i32, Immediate offset added to each address.
  auto immOffsetEncode = i32_val(immediateOffset);

  // arg6: i8, the dataum size
  assert(dataumSize != LSC_DATA_SIZE_INVALID && "unsupported data type.");
  auto dataumEncode = i8_val(dataumSize);

  // arg7: i8, Number of elements to load/store per address
  // make sure vector size is supported by the lsc instrinsic
  // lsc intrinsic supports limited vector sizes: 1, 2, 3, 4, 8, 16, 32, 64
  assert(vectSize != LSC_DATA_ELEMS_INVALID &&
         "invalid data elems per address.");
  auto vectSizeEncode = i8_val(vectSize);

  // arg8: i8, Indicates if the data is transposed during the transfer.
  assert(transpose != LSC_DATA_ORDER_INVALID && "invalid data order.");
  auto transposeEncode = i8_val(transpose);

  // arg9: i8 Channel mask for quad versions, fixed to 0 for regular op
  auto maskEncode = i8_val(channel_mask);

  // arg10: vNxi{16, 32, 64} the vector register holding offsets.
  // using addresses arg directly.

  // arg11: for store only, the data to write, using the data argument

  // arg12: i32, surface to use for this operation, fixed to 0 now.
  auto surfaceEncode = i32_val(surface);

  llvm::SmallVector<Value> args;

  if (opCode == LSC_LOAD)
    args = {pred,         opCodeEncode,    l1Encode,
            l3Encode,     addrScaleEncode, immOffsetEncode,
            dataumEncode, vectSizeEncode,  transposeEncode,
            maskEncode,   addresses,       surfaceEncode};

  if (opCode == LSC_STORE)
    args = {pred,         opCodeEncode,    l1Encode,
            l3Encode,     addrScaleEncode, immOffsetEncode,
            dataumEncode, vectSizeEncode,  transposeEncode,
            maskEncode,   addresses,       data,
            surfaceEncode};

  return createFuncCall(rewriter, loc, intrinsicStr, resultType, args, false);
}

// Create a call to lsc.load/store/prefetch.* intrinsics. It transforms
// user-friendly arguments into lsc instrinsic required encoding, and
// then calls genRawLSCIntrinsicCall. It masked out some arguments of
// genRawLSCIntrinsicCall with constant settings. And also, for 8/16-bit
// data, it will do u8c32b/u16c32b (C32B, convert to 32b) conversion,
// which means, for load, 8/16-bit data will be read from memory, but
// stored as 32-bit data in register with zero extending, and for store,
// 32-bit data in register will be truncated into 8/16-bit data and saved
// into memory.
static func::CallOp genLSCIntrinsicCallWithEncoding(
    ConversionPatternRewriter &rewriter, Location &loc, StringRef intrinsicStr,
    TypeRange resultType, Value pred, enum LSC_OP opCode,
    std::optional<xegpu::CachePolicy> l1, std::optional<xegpu::CachePolicy> l3,
    Type elemTy, int vectSize, bool transpose, Value addresses,
    Value data = {}) {

  auto getTransposeEncoding = [&](bool transpose) {
    return transpose ? LSC_DATA_ORDER_TRANSPOSE : LSC_DATA_ORDER_NONTRANSPOSE;
  };

  // encode the dataum size according to lsc spec.
  // while 32/64-bit data is encoded as normal,
  // 8-bit data, is encoded as u8c32b,
  // 16-bit data, is encoded as u16c32b
  auto getDataumEncoding = [&](Type elemType) {
    if (!elemType.isIntOrFloat())
      return LSC_DATA_SIZE_INVALID;

    auto bitWidth = elemType.getIntOrFloatBitWidth();
    switch (bitWidth) {
    case 8:
      return LSC_DATA_SIZE_8c32b;
    case 16:
      return LSC_DATA_SIZE_16c32b;
    case 32:
      return LSC_DATA_SIZE_32b;
    case 64:
      return LSC_DATA_SIZE_64b;
    }
    return LSC_DATA_SIZE_INVALID;
  };

  auto getVectorSizeEncoding = [&](int vectorSize) {
    switch (vectorSize) {
    case 1:
      return LSC_DATA_ELEMS_1;
    case 2:
      return LSC_DATA_ELEMS_2;
    case 3:
      return LSC_DATA_ELEMS_3;
    case 4:
      return LSC_DATA_ELEMS_4;
    case 8:
      return LSC_DATA_ELEMS_8;
    case 16:
      return LSC_DATA_ELEMS_16;
    case 32:
      return LSC_DATA_ELEMS_32;
    case 64:
      return LSC_DATA_ELEMS_64;
    default:
      return LSC_DATA_ELEMS_INVALID;
    }
    return LSC_DATA_ELEMS_INVALID;
  };

  assert((opCode == LSC_LOAD || opCode == LSC_STORE) && "unsupported opcode.");

  // arg6: Cache behavior for L1
  auto l1Encode = getCacheEncoding(l1);

  // arg7: Cache behavior for L3
  auto l3Encode = getCacheEncoding(l3);

  // arg8: Address scale, fixed to 1 for regular op
  short addrScale = 1;

  // arg9: Immediate offset added to each address.
  int immOffset = 0;

  // arg10: the dataum size
  auto dataumSizeEncode = getDataumEncoding(elemTy);

  // arg11: Number of elements to load/store per address
  auto vectSizeEncode = getVectorSizeEncoding(vectSize);

  // arg12: Indicates if the data is transposed during the transfer.
  auto transposeEncode = getTransposeEncoding(transpose);

  // arg13: Channel mask for quad versions, fixed to 0 for regular op
  int channel_mask = 0;

  // arg14: vNxi{16, 32, 64} the vector register holding offsets.

  // arg15: i32, surface to use for this operation, fixed to 0 now.
  int surface = 0;

  // arg16: for store only, the data to write, using the data argument.

  // create the call to lsc intrinsic
  return genRawLSCIntrinsicCall(rewriter, loc, intrinsicStr, resultType,
                                TypedValue<VectorType>(pred.getImpl()), opCode,
                                static_cast<LSC_CACHE_OPT>(l1Encode),
                                static_cast<LSC_CACHE_OPT>(l3Encode), addrScale,
                                immOffset,
                                static_cast<LSC_DATA_SIZE>(dataumSizeEncode),
                                static_cast<LSC_DATA_ELEMS>(vectSizeEncode),
                                static_cast<LSC_DATA_ORDER>(transposeEncode),
                                channel_mask, addresses, surface, data);
}

// Generate a call to lsc.load intrinsic using convert to 32bit conversion for
// 8/16-bit data, since the hardware doesn't support them well. 8/16-bit data
// will be read from memory, but stored as 32-bit data in register with zero
// extending. To make it seamless, the result is truncated to match the original
// data type if original data type is 8/16-bit.
static Value genLoadIntrinsicCallWithC32BConversion(
    ConversionPatternRewriter &rewriter, Location &loc, VectorType resultTy,
    int simd_lanes, Value pred, std::optional<xegpu::CachePolicy> l1,
    std::optional<xegpu::CachePolicy> l3, Type elemTy, int chunkSize,
    xegpu::MemorySpace scope, Value addresses) {

  // truncate the value from i32Ty to elemTy.
  auto truncate = [&](Value value, Type elemTy,
                      ConversionPatternRewriter &rewriter) -> Value {
    auto vecTy = dyn_cast<VectorType>(value.getType());
    if (!vecTy)
      return value;

    auto iTy = rewriter.getIntegerType(elemTy.getIntOrFloatBitWidth());
    auto iVecTy = VectorType::get(vecTy.getShape(), iTy);
    value = rewriter.create<arith::TruncIOp>(loc, iVecTy, value);
    if (isa<FloatType>(elemTy)) {
      // need a bitcast from e.g., i16 to f16.
      auto fVecTy = VectorType::get(vecTy.getShape(), elemTy);
      value = rewriter.create<vector::BitCastOp>(loc, fVecTy, value);
    }

    return value;
  };

  // for lsc.load, all 8/16-bit data has to be encoded as i32.
  auto lscTy = getOrigOrI32VectorType(resultTy);
  auto [resTyStr, resTy] = convertVectorType(lscTy);
  auto intrinsicStr = getLSCIntrinsicStr("load", simd_lanes, scope, resTyStr);

  auto callOp = genLSCIntrinsicCallWithEncoding(
      rewriter, loc, intrinsicStr, resTy, pred.getImpl(), LSC_LOAD, l1, l3,
      elemTy, chunkSize, false, addresses);

  if (resTy.getElementType() != elemTy)
    return truncate(callOp->getResult(0), elemTy, rewriter);
  return callOp.getResult(0);
}

// generate a special case of lsc.load for 1D load, which is similar to
// genLoadIntrinsicCallWithC32BConversion, but with simd_lanes = 1,
// and only support 32/64-bit data.
static Value gen1DLoadInstrinsicCall(ConversionPatternRewriter &rewriter,
                                     Location &loc, VectorType resultTy,
                                     std::optional<xegpu::CachePolicy> l1,
                                     std::optional<xegpu::CachePolicy> l3,
                                     Type elemTy, int elems,
                                     xegpu::MemorySpace scope, Value payload) {
  const int simd_lanes = 1;
  auto pred = dense_vector_int_val(1, i1Ty, simd_lanes);
  auto bitWidth = elemTy.getIntOrFloatBitWidth();
  assert((resultTy.getElementType() == elemTy && bitWidth >= 32) &&
         "1D block is only for 32/64-bit data.");

  auto [resTyStr, resTy] = convertVectorType(resultTy);
  auto intrinsicStr = getLSCIntrinsicStr("load", simd_lanes, scope, resTyStr);

  // it uses genLSCIntrinsicCallWithEncoding. it is ensured that the
  // data is 32/64-bit, so the convert-to-32b will be not triggered, and
  // therefore no truncate is needed.
  auto callOp = genLSCIntrinsicCallWithEncoding(
      rewriter, loc, intrinsicStr, resTy, pred, LSC_LOAD, l1, l3, elemTy, elems,
      false /* transpose */, payload);

  return callOp->getResult(0);
}

// generate a call to lsc.prefetch intrinsic. It is built on top of
// genLSCIntrinsicCall.
static func::CallOp
genPrefetchIntrinsicCall(ConversionPatternRewriter &rewriter, Location &loc,
                         int simd_lanes, std::optional<xegpu::CachePolicy> l1,
                         std::optional<xegpu::CachePolicy> l3, Type elemTy,
                         int chunkSize, xegpu::MemorySpace MemorySpace,
                         Value addresses) {
  auto intrinsicStr = getLSCIntrinsicStr("prefetch", simd_lanes, MemorySpace);
  auto pred = dense_vector_int_val(1, i1Ty, simd_lanes);
  return genLSCIntrinsicCallWithEncoding(
      rewriter, loc, intrinsicStr, {} /* null resultType */, pred, LSC_LOAD, l1,
      l3, elemTy, chunkSize, false, addresses);
}

// generate a call to lsc.prefetch intrinsic for 1D prefetch, which is
// a special case of genPrefetchIntrinsicCall, but with simd_lanes = 1.
static func::CallOp gen1DPrefetchIntrinsicCall(
    ConversionPatternRewriter &rewriter, Location &loc,
    std::optional<xegpu::CachePolicy> l1, std::optional<xegpu::CachePolicy> l3,
    Type elemTy, int elems, xegpu::MemorySpace MemorySpace, Value payload) {
  const int simd_lanes = 1;
  auto bitWidth = elemTy.getIntOrFloatBitWidth();
  assert(bitWidth >= 32 && "1D block is only for 32/64-bit data.");
  return genPrefetchIntrinsicCall(rewriter, loc, simd_lanes, l1, l3, elemTy,
                                  elems, MemorySpace, payload);
}

// Generate a call to lsc.store intrinsic, using convert-to-32b conversion
// for 8-bit/16-bit data, since the hardware doesn't support them well.
// The hardware takes 32-bit data as input only. Thus, when to write 8/16-bit
// data, the data to be written has to be extended to 32-bit first, and then
// the hardware will truncate the data to 8/16-bit before write to memory.
// (Yes, it is unfortunate that the hardware can not write 8/16-bit data
// into memory directly)
static func::CallOp genStoreIntrinsicCallWithC32BConversion(
    ConversionPatternRewriter &rewriter, Location &loc, int simd_lanes,
    Value pred, std::optional<xegpu::CachePolicy> l1,
    std::optional<xegpu::CachePolicy> l3, Type elemTy, int chunkSize,
    xegpu::MemorySpace scope, Value addresses, Value data) {

  // lsc store only takes 32-bit data as input and save the least 8-bit,
  // or 16-bit to the memory. So we need to extend the data to 32-bit if
  // it is 8-bit or 16-bit.
  auto extendTo32Bit = [&](Value value) -> Value {
    auto vecTy = dyn_cast<VectorType>(value.getType());
    if (!vecTy)
      return value;

    auto elemTy = vecTy.getElementType();

    if (!elemTy.isIntOrFloat())
      return value;

    auto dstTy = getOrigOrI32VectorType(vecTy);
    if (dstTy == vecTy)
      return value;

    // float16 needs to be casted to i16 then extending to i32
    if (isa<FloatType>(elemTy)) {
      auto intTy = rewriter.getIntegerType(elemTy.getIntOrFloatBitWidth());
      auto intVecTy = VectorType::get(vecTy.getShape(), intTy);
      value = rewriter.create<vector::BitCastOp>(loc, intVecTy, value);
      return rewriter.create<arith::ExtUIOp>(loc, dstTy, value);
    } else if (elemTy.isSignedInteger()) { // signed integer
      return rewriter.create<arith::ExtSIOp>(loc, dstTy, value);
    } else { // signless integer
      return rewriter.create<arith::ExtUIOp>(loc, dstTy, value);
    }
  };

  // extend 8/16-bit to i32 first.
  if (elemTy.getIntOrFloatBitWidth() < 32)
    data = extendTo32Bit(data);

  // get instrinsic name
  auto vecTy = dyn_cast<VectorType>(data.getType());
  auto typeStr = convertVectorType(vecTy).first;
  auto intrinsicStr = getLSCIntrinsicStr("store", simd_lanes, scope, typeStr);

  return genLSCIntrinsicCallWithEncoding(
      rewriter, loc, intrinsicStr, {}, pred, LSC_STORE, l1, l3, elemTy,
      chunkSize, false /* transpose */, addresses, data);
}

// generate a call to lsc.store intrinsic for 1D store, which is a special
// case to genStoreIntrinsicCallWithC32BConversion, but with simd_lanes = 1.
static func::CallOp
gen1DStoreInstrinsicCall(ConversionPatternRewriter &rewriter, Location &loc,
                         std::optional<xegpu::CachePolicy> l1,
                         std::optional<xegpu::CachePolicy> l3, Type elemTy,
                         int elems, xegpu::MemorySpace scope, Value payload,
                         Value data) {
  auto bitWidth = elemTy.getIntOrFloatBitWidth();
  assert(bitWidth >= 32 && "1D block is only for 32/64-bit data.");
  const int simd_lanes = 1;
  auto pred = dense_vector_int_val(1, i1Ty, simd_lanes);

  // get instrinsic name
  auto vecTy = dyn_cast<VectorType>(data.getType());
  auto typeStr = convertVectorType(vecTy).first;
  auto intrinsicStr = getLSCIntrinsicStr("store", simd_lanes, scope, typeStr);

  return genLSCIntrinsicCallWithEncoding(
      rewriter, loc, intrinsicStr, {} /* resultType*/, pred, LSC_STORE, l1, l3,
      elemTy, elems, false /* transpose */, payload, data);
}

// generate a call to lsc.load/store/prefetch.2d.ugm.* intrinsic for block IO.
static func::CallOp gen2DBlockIntrinsicCall(
    ConversionPatternRewriter &rewriter, Location &loc, StringRef intrinsicStr,
    TypeRange resultType, std::optional<xegpu::CachePolicy> l1,
    std::optional<xegpu::CachePolicy> l3, int nblocks,
    llvm::ArrayRef<int64_t> blockShape, Value payload, Value data) {
  assert(blockShape.size() == 2 && "blockShape has to be 2D.");

  // arg0: i1, predicate(true for now)
  auto predicate = i1_val(1);

  // arg1: vNi8, Cache controls
  auto l1Encode = i8_val(getCacheEncoding(l1));
  auto l3Encode = i8_val(getCacheEncoding(l3));
  auto cacheControls = rewriter.create<vector::FromElementsOp>(
      loc, vecTy(2, i8Ty), ValueRange({l1Encode, l3Encode}));

  // arg2: i8, Number of blocks
  auto nBlks = i8_val(nblocks);

  // arg3: i16, block width (in elements)
  auto blkW = i16_val(blockShape[1]);

  // arg4: i16, block height
  auto blkH = i16_val(blockShape[0]);

  // arg5: payload from parameter

  // arg6: i32, memory block X immediate offset (in elements)
  auto offsetX = i32_val(0);

  // arg7: i32, memory block Y immediate offset
  auto offsetY = i32_val(0);

  llvm::SmallVector<Value> args({predicate, cacheControls, nBlks, blkW, blkH,
                                 payload, offsetX, offsetY, data});

  return createFuncCall(rewriter, loc, intrinsicStr, resultType, args, false);
}

// generate a call to lsc.load.2d.ugm.* intrinsic for 2D block load, which is
// built on top of gen2DBlockIntrinsicCall.
static func::CallOp gen2DLoadIntrinsicCall(
    ConversionPatternRewriter &rewriter, Location &loc, StringRef intrinsicStr,
    TypeRange resultType, std::optional<xegpu::CachePolicy> l1,
    std::optional<xegpu::CachePolicy> l3, xegpu::TensorDescType tdescTy,
    Value payload, Value passthru) {
  assert(tdescTy.getRank() == 2 && !tdescTy.isScattered() &&
         "Only works on 2D block TensorDesc.");
  auto nblks = tdescTy.getArrayLength();
  auto shape = tdescTy.getShape();
  return gen2DBlockIntrinsicCall(rewriter, loc, intrinsicStr, resultType, l1,
                                 l3, nblks, shape, payload, passthru);
}

// generate a call to lsc.prefetch.2d.ugm.* intrinsic for 2D block prefetch,
// which is built on top of gen2DBlockIntrinsicCall.
static func::CallOp
gen2DPrefetchIntrinsicCall(ConversionPatternRewriter &rewriter, Location &loc,
                           std::optional<xegpu::CachePolicy> l1,
                           std::optional<xegpu::CachePolicy> l3,
                           xegpu::TensorDescType tdescTy, Value payload) {

  assert(tdescTy.getRank() == 2 && !tdescTy.isScattered() &&
         "Only works on 2D block TensorDesc.");

  auto nblks = tdescTy.getArrayLength();
  auto shape = tdescTy.getShape();
  auto elemTy = tdescTy.getElementType();
  auto noRetTy = TypeRange({});
  auto bitWidth = elemTy.getIntOrFloatBitWidth();
  auto prefix = elemTy.isInteger() ? "i" : elemTy.isBF16() ? "bf" : "f";
  auto typeStr = llvm::formatv("{0}{1}", prefix, bitWidth).str();
  auto intrinsicStr = getBlockIntrinsicStr("prefetch", typeStr);

  // for arg8: dummy value
  auto attr = elemTy.isInteger()
                  ? (TypedAttr)rewriter.getIntegerAttr(elemTy, 0)
                  : (TypedAttr)rewriter.getFloatAttr(elemTy, 0.0);
  auto dummy = constant_val(attr);
  return gen2DBlockIntrinsicCall(rewriter, loc, intrinsicStr, noRetTy, l1, l3,
                                 nblks, shape, payload, dummy);
}

// generate a call to lsc.store.2d.ugm.* intrinsic for 2D block store, which is
// built on top of gen2DBlockIntrinsicCall.
static func::CallOp gen2DStoreIntrinsicCall(
    ConversionPatternRewriter &rewriter, Location &loc,
    std::optional<xegpu::CachePolicy> l1, std::optional<xegpu::CachePolicy> l3,
    xegpu::TensorDescType tdescTy, Value payload, Value data) {
  auto nblks = tdescTy.getArrayLength();
  auto rank = tdescTy.getRank();
  assert(rank == 2 && !tdescTy.isScattered() &&
         "Only works on 2D block TensorDesc.");
  assert(nblks == 1 && "Block store only works on 1 block.");

  auto vecTy = cast<VectorType>(data.getType());
  auto typeStr = convertVectorType(vecTy).first;
  auto intrinsicStr = getBlockIntrinsicStr("store", typeStr);

  auto shape = tdescTy.getShape();
  auto noRetTy = TypeRange({});
  return gen2DBlockIntrinsicCall(rewriter, loc, intrinsicStr, noRetTy, l1, l3,
                                 nblks, shape, payload, data);
}

auto get1DTdescNumTotalElems = [](TensorDescType tdescTy) -> int64_t {
  return tdescTy.getNumElements() * tdescTy.getArrayLength();
};

auto getElemBitWidth = [](TensorDescType tdescTy) -> unsigned {
  return tdescTy.getElementType().getIntOrFloatBitWidth();
};

auto isLowPrecision = [](TensorDescType tdescTy) -> bool {
  auto width = getElemBitWidth(tdescTy);
  return width < 32 && width >= 4;
};

auto getScaled1DTdesc =
    [](TensorDescType tdescTy,
       ConversionPatternRewriter &rewriter) -> TensorDescType {
  // return if not 1D tensor desc
  if (tdescTy.getShape().size() != 1)
    return tdescTy;
  // return if not low precision
  if (!isLowPrecision(tdescTy))
    return tdescTy;

  auto scaledTy = tdescTy.getElementType();
  auto totalBytes =
      get1DTdescNumTotalElems(tdescTy) * getElemBitWidth(tdescTy) / 8;
  switch (totalBytes) {
  // i32 for 4, 8, 12, 16, 32, 64, 128, 256
  // i64 for 24 and 512
  case 4:
  case 8:
  case 12:
  case 16:
  case 32:
  case 64:
  case 128:
  case 256:
    scaledTy = rewriter.getI32Type();
    break;
  case 24:
  case 512:
    scaledTy = rewriter.getI64Type();
    break;
  default:
    break;
  }
  return TensorDescType::get(
      tdescTy.getContext(),
      {totalBytes / (scaledTy.getIntOrFloatBitWidth() / 8)}, scaledTy,
      tdescTy.getEncoding(), /*sg_map*/ nullptr);
};

auto isScaled = [](TensorDescType tdescTy, TensorDescType scaledTy) -> bool {
  return getElemBitWidth(tdescTy) != getElemBitWidth(scaledTy);
};

#define shrui(...) rewriter.createOrFold<arith::ShRUIOp>(loc, __VA_ARGS__)
class LoadNdPattern : public OpConversionPattern<LoadNdOp> {
  using OpConversionPattern<LoadNdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoadNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto tdescTy = op.getTensorDescType();
    auto elemTy = tdescTy.getElementType();
    auto rank = tdescTy.getRank();
    auto scope = tdescTy.getMemorySpace();

    auto l1hint = op.getL1Hint();
    auto l3hint = op.getL3Hint();

    auto vnni = op.getPacked().value_or(false);
    auto transpose =
        op.getTransposeAttr() ? op.getTransposeAttr()[0] == 1 : false;

    if (vnni && transpose)
      return rewriter.notifyMatchFailure(
          op, "vnni and transpose cannot be set at the same time");

    auto transform = vnni ? "vnni" : (transpose ? "transpose" : "");

    if (rank == 1) {
      // for slm and 1D tensor desc, use lsc.load
      if (vnni)
        return rewriter.notifyMatchFailure(
            op, "vnni is not supported for slm and 1D tensor desc");

      if (transpose)
        return rewriter.notifyMatchFailure(
            op, "transpose is not supported for slm and 1D tensor desc");

      auto scaledTdescTy = getScaled1DTdesc(tdescTy, rewriter);
      auto scaledElems = get1DTdescNumTotalElems(scaledTdescTy);
      auto scaledElemTy = scaledTdescTy.getElementType();

      if (failed(
              isValid1DBlockSetup(scaledElemTy, scaledElems, loc, rewriter))) {
        return rewriter.notifyMatchFailure(
            loc, "unsupported 1D/SLM TensorDescType.");
      }
      bool scaled = isScaled(tdescTy, scaledTdescTy);
      auto resTy =
          scaled ? VectorType::get({scaledElems}, scaledElemTy) : op.getType();
      auto newValue = gen1DLoadInstrinsicCall(
          rewriter, loc, resTy, l1hint, l3hint, scaledElemTy, scaledElems,
          tdescTy.getMemorySpace(), adaptor.getTensorDesc());
      if (scaled) {
        newValue =
            rewriter.create<vector::BitCastOp>(loc, op.getType(), newValue);
      }
      rewriter.replaceOp(op, newValue);
      return success();
    } else if (rank == 2) { // 2d.ugm.desc
      if (scope != xegpu::MemorySpace::Global)
        return rewriter.notifyMatchFailure(
            op, "Only global access supported for block load.");
      auto payload = adaptor.getTensorDesc();
      auto retTy = op.getType();

      // TODO: remove this after moving transposeBitWidth into a standalone
      // pass. update the width and pictch of the payload when transposeBitWidth
      // is set, and larger than the element bit width.
      auto transposeBitWidth = op.getTransposeBitWidth().value_or(0);
      auto factor = transposeBitWidth / elemTy.getIntOrFloatBitWidth();
      if (factor > 1) {
        // update the block offset X of the payload, since it is in unit of
        // elements we don't need to update the surface width and pitch, since
        // they are in unit of bytes.
        Value offsetX = rewriter.create<vector::ExtractOp>(loc, payload, 5);
        auto log2 = [&](int val) -> unsigned {
          if (val == 2)
            return 1;
          else if (val == 4)
            return 2;
          else
            assert(false && "invalid vnni Factor!");
        };
        offsetX = shrui(offsetX, i32_val(log2(factor)));
        payload = rewriter.create<vector::InsertOp>(loc, offsetX, payload, 5);

        // update the block width. Here we simply create a new TensorDescType
        // with updated shape only, since it is only consumed by the
        // gen2DBlockIntrinsicCall. which uses the shape and arraylength only to
        // keep the clean interface. This part of the logic will be moved out.
        auto shape = tdescTy.getShape().vec();
        shape[1] = shape[1] / factor;
        tdescTy =
            TensorDescType::get(tdescTy.getContext(), shape, elemTy,
                                tdescTy.getEncoding(), /*sg_map*/ nullptr);

        // update arg7 of the payload
        auto nblks = tdescTy.getArrayLength();
        auto blkW = shape[1];
        auto blkH = shape[0];
        auto block = i32_val((nblks - 1) << 16 | (blkH - 1) << 8 | (blkW - 1));
        payload = rewriter.create<vector::InsertOp>(loc, block, payload, 7);

        // update the retTy
        if (elemTy.isInteger()) {
          elemTy = rewriter.getIntegerType(transposeBitWidth);
        } else {
          if (transposeBitWidth == 16)
            elemTy = rewriter.getF16Type();
          else if (transposeBitWidth == 32)
            elemTy = rewriter.getF32Type();
        }

        retTy = VectorType::get(shape, elemTy);
      }

      auto [resultTyStr, resultTy] = convertVectorType(retTy);
      auto intrinsicStr = getBlockIntrinsicStr("load", resultTyStr, transform);

      // for arg8: value to passthru when predicate is false, using zero now
      auto attr = elemTy.isInteger()
                      ? (TypedAttr)rewriter.getIntegerAttr(elemTy, 0)
                      : (TypedAttr)rewriter.getFloatAttr(elemTy, 0.0);
      auto passthru = dense_vector_val(attr, resultTy);

      auto callOp =
          gen2DLoadIntrinsicCall(rewriter, loc, intrinsicStr, resultTy, l1hint,
                                 l3hint, tdescTy, payload, passthru)
              .getResult(0);

      // TODO: remove this after moving transposeBitWidth into a standalone
      // pass.
      if (retTy != op.getType()) {
        auto targetTy = convertVectorType(op.getType()).second;
        callOp = rewriter.create<vector::BitCastOp>(loc, targetTy, callOp);
      }
      rewriter.replaceOp(op, callOp);
      return success();
    }
    return failure();
  }
};

class PrefetchNdPattern : public OpConversionPattern<PrefetchNdOp> {
  using OpConversionPattern<PrefetchNdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PrefetchNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto tdescTy = op.getTensorDescType();
    auto rank = tdescTy.getRank();
    auto scope = tdescTy.getMemorySpace();

    auto l1hint = op.getL1Hint();
    auto l3hint = op.getL3Hint();

    if (rank == 1) { // for 1D tensor desc, use lsc.load

      if (scope == xegpu::MemorySpace::SLM) {
        // no prefetch for slm.
        rewriter.eraseOp(op);
        return success();
      }

      auto scaledTdescTy = getScaled1DTdesc(tdescTy, rewriter);
      auto scaledElems = get1DTdescNumTotalElems(scaledTdescTy);
      auto scaledElemTy = scaledTdescTy.getElementType();

      if (failed(isValid1DBlockSetup(scaledElemTy, scaledElems, loc, rewriter)))
        return rewriter.notifyMatchFailure(
            loc, "unsupported 1D/SLM TensorDescType.");

      auto callOp = gen1DPrefetchIntrinsicCall(rewriter, loc, l1hint, l3hint,
                                               scaledElemTy, scaledElems, scope,
                                               adaptor.getTensorDesc());
      rewriter.replaceOp(op, callOp);
      return success();
    } else if (rank == 2) { // 2d.ugm.desc
      if (scope != xegpu::MemorySpace::Global)
        return rewriter.notifyMatchFailure(
            op, "Only global access supported for block prefetch.");
      auto callOp = gen2DPrefetchIntrinsicCall(
          rewriter, loc, l1hint, l3hint, tdescTy, adaptor.getTensorDesc());
      rewriter.replaceOp(op, callOp);
      return success();
    }
    return failure();
  }
};

class StoreNdPattern : public OpConversionPattern<StoreNdOp> {
  using OpConversionPattern<StoreNdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StoreNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto tdescTy = op.getTensorDescType();
    auto rank = tdescTy.getRank();
    auto scope = tdescTy.getMemorySpace();

    auto l1hint = op.getL1Hint();
    auto l3hint = op.getL3Hint();

    // for arg8 of block store and arg11 of regular store: the value to write
    auto data = adaptor.getValue();

    if (rank == 1) {
      auto scaledTdescTy = getScaled1DTdesc(tdescTy, rewriter);
      auto scaledElems = get1DTdescNumTotalElems(scaledTdescTy);
      auto scaledElemTy = scaledTdescTy.getElementType();

      if (failed(isValid1DBlockSetup(scaledElemTy, scaledElems, loc, rewriter)))
        return rewriter.notifyMatchFailure(
            loc, "unsupported 1D/SLM TensorDescType.");

      if (isScaled(tdescTy, scaledTdescTy)) {
        auto scaledVecTy = VectorType::get({scaledElems}, scaledElemTy);
        data = rewriter.create<vector::BitCastOp>(loc, scaledVecTy, data);
      }
      auto callOp = gen1DStoreInstrinsicCall(rewriter, loc, l1hint, l3hint,
                                             scaledElemTy, scaledElems, scope,
                                             adaptor.getTensorDesc(), data);

      rewriter.replaceOp(op, callOp);
      return success();

    } else if (rank == 2) { // store.2d.ugm.desc
      if (scope != xegpu::MemorySpace::Global)
        return rewriter.notifyMatchFailure(
            op, "Only global access supported for block store.");

      auto callOp =
          gen2DStoreIntrinsicCall(rewriter, loc, l1hint, l3hint, tdescTy,
                                  adaptor.getTensorDesc(), data);
      rewriter.replaceOp(op, callOp);
      return success();
    }
    return failure();
  }
};

// A pattern lowering xegpu.load_gather to lsc.load.* intrinsic.
// For global memory access, it is lowered to lsc.load.stateless.*
// For shared memory access, it is lowered to lsc.load.slm.*
// Due to the hardware limitation, it only supports Scattered TensorDesc
// without chunk_size, e.g., TensorDesc<16xTy, #scattered> or
// TensorDesc<32xTy, #scattered> where Ty is 8/16-bit data type, e.g.,
// int8, bf16, or fp16. For 32-bit/64-bit data, chunk_size 1, 2, 3, 4, 8
// can be used. The chunk_size is used to control the number of continuous
// elements loaded per address. However the total size is limited to 512bytes
// per load. So for f32, it can support TensorDesc<16x8xf32, #scattered>
// or TensorDesc<32x4xf32, #scattered>, but not TensorDesc<16x6xf32,
// #scattered>. (invalid chunk size) nor TensorDesc<32x8xf32, #scattered>
// (exceeds 512bytes).
class LoadGatherPattern : public OpConversionPattern<LoadGatherOp> {
public:
  using OpConversionPattern<LoadGatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoadGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tdescTy = op.getTensorDescType();
    auto elemTy = tdescTy.getElementType();
    auto chunkSize = tdescTy.getChunkSize();
    auto simd_lanes = tdescTy.getShape()[0];

    // make sure it is a hardware supported TensorDescType
    if (failed(
            isValidScatterSetup(elemTy, simd_lanes, chunkSize, loc, rewriter)))
      return rewriter.notifyMatchFailure(
          loc, "unsupported TensorDescType for lsc load.");

    auto l1hint = op.getL1Hint();
    // auto l2hint = op.getL2Hint();
    auto l3hint = op.getL3Hint();

    auto resultTy = cast<VectorType>(op.getType());
    auto newValue = genLoadIntrinsicCallWithC32BConversion(
        rewriter, loc, resultTy, simd_lanes, op.getMask(), l1hint, l3hint,
        elemTy, chunkSize, tdescTy.getMemorySpace(), adaptor.getTensorDesc());
    rewriter.replaceOp(op, newValue);

    return success();
  }
};

class PrefetchPattern : public OpConversionPattern<PrefetchOp> {
public:
  using OpConversionPattern<PrefetchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PrefetchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tdescTy = op.getTensorDescType();
    auto elemTy = tdescTy.getElementType();
    auto chunkSize = tdescTy.getChunkSize();
    auto simd_lanes = tdescTy.getShape()[0];
    auto scope = tdescTy.getMemorySpace();

    // For SLM, there is not prefetch available, we will simply
    // remove the prefetch op.
    if (scope == xegpu::MemorySpace::SLM) {
      rewriter.eraseOp(op);
      return success();
    }

    // lsc intrinsic has the same constraints for prefetch as load
    if (failed(
            isValidScatterSetup(elemTy, simd_lanes, chunkSize, loc, rewriter)))
      return rewriter.notifyMatchFailure(
          loc, "unsupported TensorDescType for lsc prefetch.");

    auto l1hint = op.getL1Hint();
    // auto l2hint = op.getL2Hint();
    auto l3hint = op.getL3Hint();

    auto callOp = genPrefetchIntrinsicCall(rewriter, loc, simd_lanes, l1hint,
                                           l3hint, elemTy, chunkSize, scope,
                                           adaptor.getTensorDesc());

    rewriter.replaceOp(op, callOp);
    return success();
  }
};

// A pattern lowering xegpu.store_scatter to lsc.store.* intrinsic.
// For global memory access, it is lowered to lsc.store.stateless.*
// For shared memory access, it is lowered to lsc.store.slm.*
// Due to the hardware limitation, it only supports Scattered TensorDesc
// without chunk_size, e.g., TensorDesc<16xTy, #scattered> or
// TensorDesc<32xTy, #scattered> where Ty is 8/16-bit data type, e.g.,
// int8, bf16, or fp16. For 32-bit/64-bit data, chunk_size 1, 2, 3, 4, 8
// can be used. The chunk_size is used to control the number of continuous
// elements loaded per address. However the total size is limited to 512bytes
// per store. So for f32, it can support TensorDesc<16x8xf32, #scattered>
// or TensorDesc<32x4xf32, #scattered>, but not TensorDesc<16x6xf32,
// #scattered>. (invalid chunk size) nor TensorDesc<32x8xf32, #scattered>
// (exceeds 512bytes).
class StoreScatterPattern : public OpConversionPattern<StoreScatterOp> {
public:
  using OpConversionPattern<xegpu::StoreScatterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xegpu::StoreScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto tdescTy = op.getTensorDescType();
    auto elemTy = tdescTy.getElementType();
    auto chunkSize = tdescTy.getChunkSize();
    auto simd_lanes = tdescTy.getShape()[0];

    // make sure it is a hardware supported TensorDescType
    if (failed(
            isValidScatterSetup(elemTy, simd_lanes, chunkSize, loc, rewriter)))
      return rewriter.notifyMatchFailure(
          loc, "unsupported TensorDescType for lsc store.");

    auto l1hint = op.getL1Hint();
    // auto l2hint = op.getL2Hint();
    auto l3hint = op.getL3Hint();
    auto callOp = genStoreIntrinsicCallWithC32BConversion(
        rewriter, loc, simd_lanes, op.getMask(), l1hint, l3hint, elemTy,
        chunkSize, tdescTy.getMemorySpace(), adaptor.getTensorDesc(),
        adaptor.getValue());

    rewriter.replaceOp(op, callOp);
    return success();
  }
};

class AtomicPattern : public OpConversionPattern<AtomicRMWOp> {
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
      return rewriter.create<arith::ConstantOp>(loc, type, attr);
    };

    /// collect common info
    auto i1Type = rewriter.getI1Type();
    auto i8Type = rewriter.getI8Type();
    auto i16Type = rewriter.getI16Type();
    auto i32Type = rewriter.getI32Type();
    VectorType vecType = cast<VectorType>(op.getResult().getType());
    std::string funcName = "llvm.genx.lsc.xatomic.stateless.";
    auto [typeStr, newType] = encodeVectorType(rewriter, vecType, false, true);
    funcName += typeStr;

    /// fill in parameters for lsc
    auto v16i1 = VectorType::get(16, i1Type);
    auto vecAttr = DenseElementsAttr::get(v16i1, true);
    auto pred = rewriter.create<arith::ConstantOp>(loc, v16i1, vecAttr);
    auto subOpcode = createIntConstant(i8Type, encodeOpcode(op.getKind()));
    auto l1CacheHint = createIntConstant(i8Type, 1);
    auto l3CacheHint = createIntConstant(i8Type, 1);
    auto addrScale = createIntConstant(i16Type, 1);
    auto immOffset = createIntConstant(i32Type, 0);
    unsigned dataSize = encodeDataum(vecType.getElementType());
    auto dataumSize = createIntConstant(i8Type, dataSize);
    unsigned numDstVal = (newType.getNumElements() + 16 - 1) / 16;
    auto lscVecSize = 0;
    if (numDstVal <= 4) {
      lscVecSize = numDstVal;
    } else {
      lscVecSize = log2(numDstVal) + 2;
    }

    auto vecSize = createIntConstant(i8Type, lscVecSize);
    auto transposed = createIntConstant(i8Type, 1);
    auto mask = adaptor.getMask();

    // payload
    Value payLoad = adaptor.getTensorDesc();
    // src
    auto v16i32Ty = VectorType::get(16, i32Type);
    Value undef = rewriter.create<mlir::ub::PoisonOp>(loc, v16i32Ty);
    Value src0 = undef;
    if (op.getValue()) {
      src0 = op.getValue();
      if (src0.getType() != newType) {
        src0 = rewriter.create<vector::BitCastOp>(loc, newType, src0);
      }
    }
    Value src1 = undef;
    auto surface = createIntConstant(i32Type, 0);
    SmallVector<Value> args{pred,       subOpcode, l1CacheHint, l3CacheHint,
                            addrScale,  immOffset, dataumSize,  vecSize,
                            transposed, mask,      payLoad,     src0,
                            src1,       surface,   undef};
    funcName += ".v16i1.v16i64";
    auto retType = newType;
    auto newOp = createFuncCall(rewriter, loc, funcName, TypeRange{retType},
                                args, false);
    auto *converter = this->getTypeConverter();
    auto castTy = converter->convertType(op.getType());
    auto cast =
        rewriter.create<vector::BitCastOp>(loc, castTy, newOp->getResult(0));
    rewriter.replaceOp(op, cast);
    return success();
  }
};

class FencePattern : public OpConversionPattern<FenceOp> {
public:
  using OpConversionPattern<FenceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto pred = i1_val(1);
    uint8_t fence_op, sfid, fence_scope;

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
    sfid = lscSFID::UGM;
    fence_scope = lscFenceScope::GROUP;

    switch (op.getMemoryKind()) {
    case xegpu::MemorySpace::Global:
      sfid = lscSFID::UGM;
      break;
    case xegpu::MemorySpace::SLM:
      sfid = lscSFID::TGM;
      break;
    }

    switch (op.getFenceScope()) {
    case xegpu::FenceScope::Workgroup:
      fence_scope = lscFenceScope::GROUP;
      break;
    case xegpu::FenceScope::GPU:
      fence_scope = lscFenceScope::GPU;
      break;
    }

    SmallVector<Value> args{pred, i8_val(sfid), i8_val(fence_op),
                            i8_val(fence_scope)};

    std::string funcName = "llvm.genx.lsc.fence.i1";

    createFuncCall(rewriter, loc, funcName, TypeRange{}, args, false);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace LSC

void populateAtomicAndFenceLSCPatterns(TypeConverter &converter,
                                       RewritePatternSet &patterns) {
  patterns.add<LSC::AtomicPattern, LSC::FencePattern>(converter,
                                                      patterns.getContext());
}

void populateLoadStoreLSCPatterns(TypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<LSC::LoadNdPattern, LSC::StoreNdPattern, LSC::PrefetchNdPattern,
               LSC::LoadGatherPattern, LSC::StoreScatterPattern,
               LSC::PrefetchPattern>(converter, patterns.getContext());
}

} // namespace imex
