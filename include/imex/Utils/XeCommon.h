
//===- XeUtils.h - XeTile/XeGPU Utility Functions --------------------*- C++
//-*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utility functions used by XeTile/XeGPU dialects.
//
//===----------------------------------------------------------------------===//

#ifndef _IMEX_XECOMMON_H_
#define _IMEX_XECOMMON_H_

#include "imex/Dialect/XeTile/IR/XeTileOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/XeGPU/IR/XeGPU.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/OneToNTypeConversion.h>
using namespace mlir::xegpu;
namespace imex {

// Combine vectors vertically while keeping the logical data layout.
// As an example, given two vectors (2x4xf16) p and q, it will merge
// them in to a 4x4xf16 vector.
//  p1, p2, p3, p4            p1, p2, p3, p4
//  p5, p6, p7, p8            p5, p6, p7, p8
//                     ==>    q1, q2, q3, q4
//  q1, q2, q3, q4            q5, q6, q7, q8
//  q5, q6, q7, q8
mlir::TypedValue<mlir::VectorType> stack(mlir::Value vecUp, mlir::Value vecDown,
                                         mlir::Location loc,
                                         mlir::PatternRewriter &rewriter);

// It checks each GPUFuncOp in the module to see
// whether they have arguments and outputs with
// xetile.TileType. They are currently not supported yet.
bool isSupportedModule(mlir::gpu::GPUModuleOp mod);

int getOperandIndex(mlir::Operation *op, mlir::Value operand);

// Obtain the index of the result in the operation. If the result is not found,
// return -1.
int getResultIndex(mlir::Operation *op, mlir::Value result);

mlir::BlockArgument getArgForOperand(mlir::scf::ForOp &op, mlir::Value operand);

mlir::ValueRange buildUnrealizedCast(mlir::OpBuilder &builder,
                                     mlir::TypeRange resultTypes,
                                     mlir::ValueRange inputs);

// An analysis hook used by mlir::getUsageAnalysis for analyzing
// how a tile created by init_tile are used in the program, e.g.,
// is it created for load, store, or prefetch. It also analyzes
// how the result of a load_tile is used, including as A operand
// of tile_mma, B operand of tile_mma or C operand of tile_mma.
// since they need different lowering strategy for in each use
// case.
class TileUsageAnalysis {
public:
  TileUsageAnalysis(mlir::Operation *op) {
    op->walk<mlir::WalkOrder::PreOrder>([&](imex::xetile::InitTileOp op) {
      Usage[op] = (uint)UsageType::None;
      llvm::SmallVector<mlir::Value> q({op});
      while (q.size()) {
        auto curr = q.pop_back_val();
        for (mlir::Operation *user : curr.getUsers()) {
          if (llvm::isa<imex::xetile::LoadTileOp>(user)) {
            Usage[op] |= (uint)UsageType::LOAD;
          } else if (llvm::isa<imex::xetile::PrefetchTileOp>(user)) {
            Usage[op] |= (uint)UsageType::PREFETCH;
          } else if (llvm::isa<imex::xetile::StoreTileOp>(user)) {
            Usage[op] |= (uint)UsageType::STORE;
          } else if (llvm::isa<imex::xetile::UpdateTileOffsetOp>(user)) {
            Usage[op] |= (uint)UsageType::OTHER;
          } else if (auto forOp =
                         llvm::dyn_cast_if_present<mlir::scf::ForOp>(user)) {
            auto arg = getArgForOperand(forOp, curr);
            q.push_back(arg);
          }
        }
      }
    }); // walk on InitTileOp

    op->walk<mlir::WalkOrder::PreOrder>([&](imex::xetile::LoadTileOp op) {
      Usage[op] = (uint)UsageType::None;
      llvm::SmallVector<mlir::Value> q({op});
      while (q.size()) {
        auto curr = q.pop_back_val();
        for (mlir::Operation *user : curr.getUsers()) {
          if (auto mma = llvm::dyn_cast_if_present<xetile::TileMMAOp>(user)) {
            auto idx = getOperandIndex(mma, curr);
            if (idx == 0)
              Usage[op] |= (uint)UsageType::DPAS_A;
            else if (idx == 1)
              Usage[op] |= (uint)UsageType::DPAS_B;
            else if (idx == 2)
              Usage[op] |= (uint)UsageType::DPAS_C;
            else
              op->emitOpError() << "unknown usage: " << idx;
          } else if (auto unpack =
                         llvm::dyn_cast_if_present<xetile::TileUnpackOp>(
                             user)) {
            q.push_back(unpack);
          } else if (auto pack =
                         llvm::dyn_cast_if_present<xetile::TilePackOp>(user)) {
            q.push_back(pack);
          }
        }
      }
    }); // walk on LoadTileOp
  };

  bool isForDPASA(imex::xetile::LoadTileOp op) {
    if (Usage.count(op)) {
      return Usage[op] & UsageType::DPAS_A;
    }
    return false;
  }

  bool isForDPASB(imex::xetile::LoadTileOp op) {
    if (Usage.count(op)) {
      return Usage[op] & UsageType::DPAS_B;
    }
    return false;
  }

  bool isForDPASC(imex::xetile::LoadTileOp op) {
    if (Usage.count(op)) {
      return Usage[op] & UsageType::DPAS_C;
    }
    return false;
  }

  bool isForLoad(imex::xetile::InitTileOp op) {
    if (Usage.count(op)) {
      bool load = Usage[op] & UsageType::LOAD;
      bool store = Usage[op] & UsageType::STORE;
      bool prefetch = Usage[op] & UsageType::PREFETCH;
      return load && !store && !prefetch;
    }
    return false;
  }

  bool isForPrefetch(imex::xetile::InitTileOp op) {
    if (Usage.count(op)) {
      bool load = Usage[op] & UsageType::LOAD;
      bool store = Usage[op] & UsageType::STORE;
      bool prefetch = Usage[op] & UsageType::PREFETCH;
      return !load && !store && prefetch;
    }
    return false;
  }

  //
  bool isForLoadAndPrefetch(imex::xetile::InitTileOp op) {
    if (Usage.count(op)) {
      bool load = Usage[op] & UsageType::LOAD;
      bool store = Usage[op] & UsageType::STORE;
      bool prefetch = Usage[op] & UsageType::PREFETCH;
      return load && !store && prefetch;
    }
    return false;
  }

  bool isForStore(imex::xetile::InitTileOp op) {
    if (Usage.count(op)) {
      bool load = Usage[op] & UsageType::LOAD;
      bool store = Usage[op] & UsageType::STORE;
      bool prefetch = Usage[op] & UsageType::PREFETCH;
      return !load && store && !prefetch;
    }
    return false;
  }

  bool isForLoadAndStore(imex::xetile::InitTileOp op) {
    if (Usage.count(op)) {
      bool load = Usage[op] & UsageType::LOAD;
      bool store = Usage[op] & UsageType::STORE;
      bool prefetch = Usage[op] & UsageType::PREFETCH;
      return load && store && !prefetch;
    }
    return false;
  }

private:
  enum UsageType {
    None = 0,
    LOAD = 1,
    PREFETCH = 2,
    STORE = 4,
    DPAS_A = 8,
    DPAS_B = 16,
    DPAS_C = 32,
    OTHER = 64
  };

  llvm::DenseMap<mlir::Operation *, uint> Usage;
};

// This analysis is used to propagate the inner block size of an operator
// to its uses or users. Current implementation is to propagate the MMA
// size used by an MMA operator to the definition (InitTileOp) for its operands.
// TODO: This analysis can be extended to propagate the block size for other ops
// such that it can be used as a general analysis for other block size
// optimizations.
class PropagateAnalysis {
private:
  llvm::DenseMap<mlir::Value, mlir::DenseI64ArrayAttr> OpAttrMap;

public:
  PropagateAnalysis(mlir::Operation *op) {
    op->walk<mlir::WalkOrder::PostOrder>([&](xetile::TileMMAOp op) {
      mlir::Operation *operation = op.getOperation();
      for (auto value : operation->getOperands()) {
        auto packOp = value.getDefiningOp<xetile::TilePackOp>();
        if (packOp) {
          auto blkSZ = packOp.getInnerBlocksAttr();
          propagate(value, blkSZ);
        }
      }
    });
  }

  bool maybeUpdated(mlir::Operation *op) const {
    assert(op->getNumResults() == 1);
    auto v = op->getResult(0);
    return OpAttrMap.count(v);
  }

  mlir::DenseI64ArrayAttr getValue(mlir::Value value) const {
    auto it = OpAttrMap.find(value);
    if (it != OpAttrMap.end())
      return it->second;
    return {};
  }

  mlir::DenseI64ArrayAttr getValue(mlir::Operation *op) const {
    assert(op->getNumResults() == 1);
    auto v = op->getResult(0);
    auto it = OpAttrMap.find(v);
    if (it != OpAttrMap.end())
      return it->second;
    return {};
  }

private:
  mlir::Operation *getDefineOrParentOp(mlir::Value value) {
    if (llvm::isa<mlir::OpResult>(value))
      return value.getDefiningOp();
    if (auto arg = llvm::dyn_cast_or_null<mlir::BlockArgument>(value))
      return arg.getOwner()->getParentOp();
    return nullptr;
  };

  mlir::Value getOperandForArg(mlir::scf::ForOp &forOp, mlir::Value &value) {
    auto arg = llvm::dyn_cast<mlir::BlockArgument>(value);
    if (arg && arg.getArgNumber() >= forOp.getNumInductionVars()) {
      auto &iterOperand = *forOp.getTiedLoopInit(arg);
      auto numCtrlOperands = forOp.getNumControlOperands();
      auto operandIdx = iterOperand.getOperandNumber();
      return forOp.getInitArgs()[operandIdx - numCtrlOperands];
    }
    return mlir::Value();
  };

  void propagate(mlir::Value start, mlir::DenseI64ArrayAttr attr) {
    llvm::SmallVector<mlir::Value> queue;
    if (bool(start))
      queue.push_back(start);

    while (queue.size()) {
      auto value = queue.pop_back_val();
      if (!bool(value))
        continue;

      auto *op = getDefineOrParentOp(value);

      // stop when meet a function or ops, e.g., arith.truncf.
      // since their source and results could have different bitwidth,
      // in which case the block size cannot be propagated.
      if (!op || llvm::isa<mlir::FunctionOpInterface>(op) ||
          llvm::isa<mlir::CastOpInterface>(op))
        continue;

      OpAttrMap[value] = attr;

      if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
        auto opr = getOperandForArg(forOp, value);
        if (bool(opr))
          queue.push_back(opr);
      } else if (op->getNumOperands() == 1) {
        queue.push_back(op->getOperand(0));
      }
    }
  }
};

std::pair<std::string, mlir::VectorType>
encodeVectorType(mlir::ConversionPatternRewriter &rewriter,
                 mlir::VectorType type, bool use64bitData = false,
                 bool enforceInteger = false, bool keepF16 = false);

mlir::VectorType encodeVectorTypeTo(mlir::VectorType currentVecType,
                                    mlir::Type toElemType);

unsigned encodeDataum(mlir::Type type);

unsigned encodeOpcode(mlir::arith::AtomicRMWKind kind);

// L1 and L3 Cache Policies for Load Operation
//  L1 Cache Policies: Uncached (UC), Cached (C), Cache Streaming (S),
//  Invalidate-After-Read (IAR) L3 Cache Policies: Uncached (UC), Cached (C)
#define L1UC_L3UC 1
#define L1UC_L3C 2
#define L1C_L3UC 3
#define L1C_L3C 4
#define L1S_L3UC 5
#define L1S_L3C 6
#define L1IAR_L3C 7

// L1 and L3 Cache Policies for Store operation
//  L1 Cache Policies: Uncached (UC), Write-Through (WT), Write-Back (WB),
//  Streaming (S) L3 Cache Policies: Uncached (UC), Cached (WB)
#define L1UC_L3WB 2
#define L1WT_L3UC 3
#define L1WT_L3WB 4
#define L1S_L3UC 5
#define L1S_L3WB 6
#define L1WB_L3WB 7

template <typename OpType> unsigned encodeCacheHint(OpType op) {
  auto l1hint = op.getL1Hint();
  auto l3hint = op.getL3Hint();

  constexpr bool isStore = std::is_same_v<OpType, mlir::xegpu::StoreNdOp> ||
                           std::is_same_v<OpType, StoreScatterOp>;
  unsigned cacheHint = L1UC_L3UC;

#define SET_CACHEVALUE(hint, cacheHintVal)                                     \
  hint.has_value() ? hint.value() : cacheHintVal

  if constexpr (!isStore) {

    auto l1CacheValue = SET_CACHEVALUE(l1hint, CachePolicy::UNCACHED);
    auto l3CacheValue = SET_CACHEVALUE(l3hint, CachePolicy::UNCACHED);

// Setting Cache policy override based on L3 Uncached/Cached value for Load
// operation
#define SET_L1L3_CACHEREADHINT(cacheHint, l3CacheValue, uncachedVal,           \
                               cachedVal)                                      \
  if (l3CacheValue == CachePolicy::UNCACHED)                                   \
    cacheHint = uncachedVal;                                                   \
  else if (l3CacheValue == CachePolicy::CACHED)                                \
    cacheHint = cachedVal;

    switch (l1CacheValue) {
    case CachePolicy::UNCACHED:
      SET_L1L3_CACHEREADHINT(cacheHint, l3CacheValue, L1UC_L3UC, L1UC_L3C);
      break;
    case CachePolicy::CACHED:
      SET_L1L3_CACHEREADHINT(cacheHint, l3CacheValue, L1C_L3UC, L1C_L3C);
      break;
    case CachePolicy::STREAMING:
      SET_L1L3_CACHEREADHINT(cacheHint, l3CacheValue, L1S_L3UC, L1S_L3C);
      break;
    case CachePolicy::READ_INVALIDATE:
      if (l3CacheValue == CachePolicy::CACHED)
        cacheHint = L1IAR_L3C;
      break;
    default:
      llvm_unreachable("Invalid Cache Policy for Read.\n");
    }

  } else {
    auto l1CacheValue = SET_CACHEVALUE(l1hint, CachePolicy::UNCACHED);
    auto l3CacheValue = SET_CACHEVALUE(l3hint, CachePolicy::UNCACHED);

// Setting Cache policy override based on L3 Uncached/Write-Back value for Store
// operation
#define SET_L1L3_CACHEWRITEHINT(cacheHint, l3CacheValue, uncachedVal,          \
                                cachedVal)                                     \
  if (l3CacheValue == CachePolicy::UNCACHED)                                   \
    cacheHint = uncachedVal;                                                   \
  else if (l3CacheValue == CachePolicy::WRITE_BACK)                            \
    cacheHint = cachedVal;

    switch (l1CacheValue) {
    case CachePolicy::UNCACHED:
      SET_L1L3_CACHEWRITEHINT(cacheHint, l3CacheValue, L1UC_L3UC, L1UC_L3WB);
      break;
    case CachePolicy::WRITE_THROUGH:
      SET_L1L3_CACHEWRITEHINT(cacheHint, l3CacheValue, L1WT_L3UC, L1WT_L3WB);
      break;
    case CachePolicy::STREAMING:
      SET_L1L3_CACHEWRITEHINT(cacheHint, l3CacheValue, L1S_L3UC, L1S_L3WB);
      break;
    case CachePolicy::WRITE_BACK:
      if (l3CacheValue == CachePolicy::WRITE_BACK)
        cacheHint = L1WB_L3WB;
      break;
    default:
      llvm_unreachable("Invalid Cache Policy for Write.\n");
    }
  }
  return cacheHint;
}
class XeTypeConverter : public mlir::TypeConverter {
public:
  // friend class XeConversionPattern;
  using mlir::TypeConverter::convertType;

  XeTypeConverter(mlir::MLIRContext &context) {
    addConversion([&](xetile::TileType tileTy,
                      llvm::SmallVectorImpl<mlir::Type> &resultTypes)
                      -> std::optional<mlir::LogicalResult> {
      return convertTileType(tileTy, resultTypes);
    });

    addConversion([&](mlir::VectorType vectorTy,
                      llvm::SmallVectorImpl<mlir::Type> &resultTypes)
                      -> std::optional<mlir::LogicalResult> {
      return convertVectorType(vectorTy, resultTypes);
    });
  }

  virtual std::optional<mlir::LogicalResult>
  convertTileType(xetile::TileType tileTy,
                  llvm::SmallVectorImpl<mlir::Type> &resultTypes) {
    llvm_unreachable("Pending Implementation for convertTileType.");
  }

  virtual std::optional<mlir::LogicalResult>
  convertVectorType(mlir::VectorType vectorTy,
                    llvm::SmallVectorImpl<mlir::Type> &resultTypes) {
    llvm_unreachable("Pending Implementation for convertVectorType.");
  }
};

// A simple mlir::RewritePattern wrapper with methods for accessing UsageType
template <typename AnalysisT>
class XeConversionPattern : public mlir::RewritePattern {
public:
  using mlir::RewritePattern::RewritePattern;

  template <typename... Args>
  XeConversionPattern(imex::XeTypeConverter &typeConverter, AnalysisT &analysis,
                      Args &&...args)
      : mlir::RewritePattern(std::forward<Args>(args)...),
        typeConverter(typeConverter), analysis(analysis) {}

  virtual mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    llvm_unreachable("must override matchAndRewrite or a rewrite method");
  };

  imex::XeTypeConverter &getTypeConverter() const { return typeConverter; }

  template <typename ConverterTy>
  std::enable_if_t<std::is_base_of<mlir::TypeConverter, ConverterTy>::value,
                   ConverterTy &>
  getTypeConverter() const {
    return static_cast<ConverterTy &>(typeConverter);
  }

protected:
  imex::XeTypeConverter &typeConverter;
  AnalysisT &analysis;

  template <typename = typename std::enable_if<
                std::is_same_v<AnalysisT, PropagateAnalysis>>>
  mlir::DenseI64ArrayAttr getValue(mlir::Operation *op) const {
    if (op)
      return llvm::cast<PropagateAnalysis>(analysis).getValue(op);
    return {};
  }

  mlir::DenseI64ArrayAttr getValue(mlir::Value value) const {
    return llvm::cast<PropagateAnalysis>(analysis).getValue(value);
  }

  template <typename = typename std::enable_if<
                std::is_same_v<AnalysisT, TileUsageAnalysis>>>
  bool isForDPASA(imex::xetile::LoadTileOp op) const {
    return llvm::cast<TileUsageAnalysis>(analysis).isForDPASA(op);
  }

  template <typename = typename std::enable_if<
                std::is_same_v<AnalysisT, TileUsageAnalysis>>>
  bool isForDPASB(imex::xetile::LoadTileOp op) const {
    return llvm::cast<TileUsageAnalysis>(analysis).isForDPASB(op);
  }

  template <typename = typename std::enable_if<
                std::is_same_v<AnalysisT, TileUsageAnalysis>>>
  bool isForDPASC(imex::xetile::LoadTileOp op) const {
    return llvm::cast<TileUsageAnalysis>(analysis).isForDPASC(op);
  }

  template <typename = typename std::enable_if<
                std::is_same_v<AnalysisT, TileUsageAnalysis>>>
  bool isForLoad(imex::xetile::InitTileOp op) const {
    return llvm::cast<TileUsageAnalysis>(analysis).isForLoad(op);
  }

  template <typename = typename std::enable_if<
                std::is_same_v<AnalysisT, TileUsageAnalysis>>>
  bool isForStore(imex::xetile::InitTileOp op) const {
    return llvm::cast<TileUsageAnalysis>(analysis).isForStore(op);
  }

  template <typename = typename std::enable_if<
                std::is_same_v<AnalysisT, TileUsageAnalysis>>>
  bool isForPrefetch(imex::xetile::InitTileOp op) const {
    return llvm::cast<TileUsageAnalysis>(analysis).isForPrefetch(op);
  }

  template <typename = typename std::enable_if<
                std::is_same_v<AnalysisT, TileUsageAnalysis>>>
  bool isForLoadAndPrefetch(imex::xetile::InitTileOp op) const {
    return llvm::cast<TileUsageAnalysis>(analysis).isForLoadAndPrefetch(op);
  }

  template <typename = typename std::enable_if<
                std::is_same_v<AnalysisT, TileUsageAnalysis>>>
  bool isForLoadAndStore(imex::xetile::InitTileOp op) const {
    return llvm::cast<TileUsageAnalysis>(analysis).isForLoadAndStore(op);
  }
};

/// Clone `shape` with the last two elements swapped.
template <typename T>
llvm::SmallVector<T> swapLastTwoElements(llvm::ArrayRef<T> shape) {
  assert(shape.size() >= 2 && "shape must be at least 2D");
  llvm::SmallVector<T> result(shape.begin(), shape.end());
  auto size = result.size();
  std::swap(result[size - 1], result[size - 2]);
  return result;
}

/// Creates the default strides for the given `shape`. Example:
///   input shape = 2x3x4x5
///   output strides = 60x20x5x1
llvm::SmallVector<int64_t> defaultStrides(llvm::ArrayRef<int64_t> shape);

/// Checks if the given `type` is a 1-D vector type that requires VectorAnyINTEL
/// capability. In other words, the vector size is not supported by SPIR-V.
/// SPIR-V only supports 2, 3, 4, 8, 16 elements (8 and 16 with Vector16
/// capability).
bool isVectorAnyINTELType(mlir::Type type);

} // namespace imex

#endif
