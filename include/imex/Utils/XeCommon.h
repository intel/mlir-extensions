
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

// a helper util to check whether the order is column major.
bool isColMajorOrder(mlir::DenseI32ArrayAttr order);

// a helper util to get the height for SLM block, given the
// block width (which is typically the simd lanes, currently fixed
// to 16), vnni factor, the tile shape, and tile order. The height
// is constrained by the supported chunk sizes, which are 1, 2, 3, 4, 8
// for scattered load/store (used for colMajor), and 16, 32, 64 for 1D
// block load/store. Also shape[0] % height == 0. otherwise, it returns 0.
int getHeightForSLMBlock(llvm::ArrayRef<int64_t> shape, int width,
                         int vnniFactor, bool colMajor);

// a helper util to check whether the tile type is supported
// for optimal SLM access lowering.
bool isSupportedOptimalSLMAccess(xetile::TileType tileTy);

// this method computes the vnni factor for the given element type.
// it returns 1 by default for types does not need vnni transformation.
int getVnniFactor(mlir::Type elemTy);

// a helper function to get the vector type after doing vnni transformation
// e.g., vector<4x4xf16> -> vector<2x4x2xf16>
mlir::VectorType getPackedType(mlir::VectorType vecTy);

// Apply VNNI transformation to the given value, using VectorShuffle
// and shapecast operations. Since it is to add some extra operations
// on the given value. Thus, the function also returns the first
// operation applied to the value for convenience, such that the
// user can replace all uses of current value, except the first
// appended operation.
std::pair<mlir::Value, mlir::Operation *>
applyVnniTransform(mlir::OpBuilder &builder,
                   mlir::TypedValue<mlir::VectorType> value);

// valid chunk sizes are 1, 2, 3, 4, 8 if simdLanes > 1.
// 16, 32, and 64 are only available if simdLanes == 1.
llvm::SmallVector<int> getSupportedChunkSizes(int simdlanes);

using PackFuncTy = std::function<mlir::TypedValue<mlir::VectorType>(
    mlir::Value, mlir::Value, mlir::Location, mlir::OpBuilder &)>;

// A wrapper function to merge small vectors into a big one. It takes a
// range of mlir::Value objects with mlir::VectorType, and merge them
// into a big vector using the provided transformation function.
mlir::Value packVectorsWith(mlir::ValueRange ins, PackFuncTy op,
                            mlir::Location loc, mlir::OpBuilder &builder);

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
                                         mlir::OpBuilder &builder);

// merge vectors horizontally while keep the logical data layout.
// 1 2 3 4   +    10 11 12   =   1 2 3 4 10 11 12
// 5 6 7 8        13 14 15       5 6 7 8 13 14 15
// since there is no direct op in mlir exists, we will
// using ShapeCast and Shuffle to mimic it. It comes with
// cost of complex shuffle masks. the mask for the above one
// will be like this: 0 1 2 3  8  9 10
//                    4 5 6 7 11 12 13
mlir::TypedValue<mlir::VectorType> concat(mlir::Value lhs, mlir::Value rhs,
                                          mlir::Location loc,
                                          mlir::OpBuilder &builder);

// It checks each GPUFuncOp in the module to see
// whether they have arguments and outputs with
// xetile.TileType. They are currently not supported yet.
bool isSupportedModule(mlir::gpu::GPUModuleOp mod);

llvm::SmallVector<int64_t> getOperandIndices(mlir::Operation *op, mlir::Value operand);

// Obtain the index of the result in the operation. If the result is not found,
// return -1.
int getResultIndex(mlir::Operation *op, mlir::Value result);

llvm::SmallVector<mlir::BlockArgument> getArgsForOperand(mlir::scf::ForOp &op, mlir::Value operand);

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
          } else if (llvm::isa<imex::xetile::AtomicRMWOp>(user)) {
            Usage[op] |= (uint)UsageType::ATOMICRMW;
          } else if (llvm::isa<imex::xetile::UpdateTileOffsetOp>(user)) {
            Usage[op] |= (uint)UsageType::OTHER;
          } else if (auto forOp =
                         llvm::dyn_cast_if_present<mlir::scf::ForOp>(user)) {
            // we need to check all ForOp arguments for using initTileOp result
            auto args = getArgsForOperand(forOp, curr);
            q.insert(q.end(),args.begin(), args.end());
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
            auto opIndices = getOperandIndices(mma, curr);
            assert (opIndices.size() == 1 && "Only MMA operations with non-equal ops supported");
            auto idx = opIndices[0];
            if (idx == 0)
              Usage[op] |= (uint)UsageType::DPAS_A;
            else if (idx == 1)
              Usage[op] |= (uint)UsageType::DPAS_B;
            else if (idx == 2)
              Usage[op] |= (uint)UsageType::DPAS_C;
            else
              op->emitOpError() << "unknown usage: " << idx;
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

  bool isForAtomicRMW(imex::xetile::InitTileOp op) {
    if (Usage.count(op)) {
      bool load = Usage[op] & UsageType::LOAD;
      bool store = Usage[op] & UsageType::STORE;
      bool prefetch = Usage[op] & UsageType::PREFETCH;
      bool atomic_rmw = Usage[op] & UsageType::ATOMICRMW;
      return !load && !store && !prefetch && atomic_rmw;
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

  bool isForLoadAndAtomicRMW(imex::xetile::InitTileOp op) {
    if (Usage.count(op)) {
      bool load = Usage[op] & UsageType::LOAD;
      bool store = Usage[op] & UsageType::STORE;
      bool prefetch = Usage[op] & UsageType::PREFETCH;
      bool atomic_rmw = Usage[op] & UsageType::ATOMICRMW;
      return load && !store && !prefetch && atomic_rmw;
    }
    return false;
  }

  bool isForAtomicRMWAndStore(imex::xetile::InitTileOp op) {
    if (Usage.count(op)) {
      bool load = Usage[op] & UsageType::LOAD;
      bool store = Usage[op] & UsageType::STORE;
      bool prefetch = Usage[op] & UsageType::PREFETCH;
      bool atomic_rmw = Usage[op] & UsageType::ATOMICRMW;
      return !load && store && !prefetch && atomic_rmw;
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
    ATOMICRMW = 64,
    OTHER = 128
  };

  llvm::DenseMap<mlir::Operation *, uint> Usage;
};

std::pair<std::string, mlir::VectorType>
encodeVectorType(mlir::ConversionPatternRewriter &rewriter,
                 mlir::VectorType type, bool use64bitData = false,
                 bool enforceInteger = false, bool keepF16 = false);

unsigned encodeDataum(mlir::Type type);

unsigned encodeOpcode(mlir::arith::AtomicRMWKind kind);

// A simple mlir::RewritePattern wrapper with methods for accessing UsageType
class XeConversionPattern : public mlir::RewritePattern {
public:
  using mlir::RewritePattern::RewritePattern;

  template <typename... Args>
  XeConversionPattern(mlir::TypeConverter &typeConverter, Args &&...args)
      : mlir::RewritePattern(std::forward<Args>(args)...),
        typeConverter(typeConverter) {}

  virtual mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    llvm_unreachable("must override matchAndRewrite or a rewrite method");
  };

  mlir::TypeConverter &getTypeConverter() const { return typeConverter; }

  template <typename ConverterTy>
  std::enable_if_t<std::is_base_of<mlir::TypeConverter, ConverterTy>::value,
                   ConverterTy &>
  getTypeConverter() const {
    return static_cast<ConverterTy &>(typeConverter);
  }

protected:
  mlir::TypeConverter &typeConverter;
};

/// Checks if the given `type` is a 1-D vector type that requires VectorAnyINTEL
/// capability. In other words, the vector size is not supported by SPIR-V.
/// SPIR-V only supports 2, 3, 4, 8, 16 elements (8 and 16 with Vector16
/// capability).
bool isVectorAnyINTELType(mlir::Type type);

/// convert OpFoldResult to Value by replacing integer
/// attributes with arith::ConstantOps. It also performs
/// simple type conversions
mlir::Value getValueOrConstantOp(mlir::OpFoldResult ofr, mlir::Location loc,
                                 mlir::PatternRewriter &rewriter,
                                 mlir::Type type = nullptr);

// A universal get method for offsets or shapes or strides (OSS) of
// xetile::InitTileOp and xegpu::CreateNdDescOp op.
// OSS (Offsets, Shapes, Strides) information provided
// to InitTileOp & CreateNdDescOp is multifaceted. In other words oss info
// provided to InitTileOp & CreateNdDescOp in multiple ways, especially the
// shapes and strides:
// 1. For static memrefs: the shapes and strides  info are inherent in the
// memref data type

// 2. For dynamic memrefs and i64/i32 source: the shapes and strides info is
// provided via the operands `sizes` and `strides` repectively, however these
// operands can also take two different types:

// 2.1 Constant type: constant attribute can be passed
// 2.2 Value type: a value type can be passed

// This function collects these info based on different scenarios and returns
// them in Value types.

// One can pass the result of getMixedOffsets(), getMixedSizes(),
// getMixedStrides() to the following utility to get them as Value types.
// Since both xetile::InitTileOp and xegpu::CreateNdDescOp ops implement the
// OffsetSizeAndStrideOpInterface, getMixedOffsets(), getMixedSizes(),
// getMixedStrides() takes care of the different scenarios mentioned above.

llvm::SmallVector<mlir::Value> getStridesOrOffsetsOrShapesInValueType(
    mlir::PatternRewriter &rewriter,
    ::llvm::SmallVector<mlir::OpFoldResult> mixedOSS, mlir::Location loc);

} // namespace imex

#endif
