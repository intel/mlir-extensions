
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

#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/OneToNTypeConversion.h>

namespace imex {

// It checks each GPUFuncOp in the module to see
// whether they have arguments and outputs with
// xetile.TileType. They are currently not supported yet.
bool isSupportedModule(mlir::gpu::GPUModuleOp mod);

int getOperandIndex(mlir::Operation *op, mlir::Value operand);
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
              llvm::dbgs() << "unknown usage: " << idx;
          }

          if (auto unpack =
                  llvm::dyn_cast_if_present<xetile::TileUnpackOp>(user)) {
            q.push_back(unpack);
          }

          if (auto pack = llvm::dyn_cast_if_present<xetile::TilePackOp>(user)) {
            q.push_back(pack);
          }
        }
      }
    }); // walk on LoadTileOp
  };

  uint getUsage(imex::xetile::InitTileOp op) {
    if (Usage.count(op))
      return Usage[op];
    return UsageType::None;
  }

  uint getUsage(imex::xetile::LoadTileOp op) {
    if (Usage.count(op))
      return Usage[op];
    return UsageType::None;
  }

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

class XeTypeConverter : public mlir::OneToNTypeConverter {
public:
  friend class XeConversionPattern;
  using mlir::OneToNTypeConverter::convertType;

  XeTypeConverter(mlir::MLIRContext &context,
                  TileUsageAnalysis *analysis = nullptr)
      : context(context), usageAnalysis(analysis) {
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

private:
  mlir::MLIRContext &context;

protected:
  TileUsageAnalysis *usageAnalysis;
};

// A simple mlir::RewritePattern wrapper with methods for accessing UsageType
class XeConversionPattern : public mlir::RewritePattern {
public:
  using mlir::RewritePattern::RewritePattern;

  template <typename... Args>
  XeConversionPattern(imex::XeTypeConverter &typeConverter, Args &&...args)
      : mlir::RewritePattern(std::forward<Args>(args)...),
        typeConverter(typeConverter) {}

  virtual mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    llvm_unreachable("must override matchAndRewrite or a rewrite method");
  };

  bool isForDPASA(imex::xetile::LoadTileOp op) const {
    return typeConverter.usageAnalysis->isForDPASA(op);
  }

  bool isForDPASB(imex::xetile::LoadTileOp op) const {
    return typeConverter.usageAnalysis->isForDPASB(op);
  }

  bool isForDPASC(imex::xetile::LoadTileOp op) const {
    return typeConverter.usageAnalysis->isForDPASC(op);
  }

  bool isForLoad(imex::xetile::InitTileOp op) const {
    return typeConverter.usageAnalysis->isForLoad(op);
  }

  bool isForStore(imex::xetile::InitTileOp op) const {
    return typeConverter.usageAnalysis->isForStore(op);
  }

  bool isForPrefetch(imex::xetile::InitTileOp op) const {
    return typeConverter.usageAnalysis->isForPrefetch(op);
  }

  bool isForLoadAndPrefetch(imex::xetile::InitTileOp op) const {
    return typeConverter.usageAnalysis->isForLoadAndPrefetch(op);
  }

  bool isForLoadAndStore(imex::xetile::InitTileOp op) const {
    return typeConverter.usageAnalysis->isForLoadAndStore(op);
  }

  imex::XeTypeConverter &getTypeConverter() const { return typeConverter; }

  template <typename ConverterTy>
  std::enable_if_t<std::is_base_of<mlir::TypeConverter, ConverterTy>::value,
                   ConverterTy &>
  getTypeConverter() const {
    return static_cast<ConverterTy &>(typeConverter);
  }

protected:
  imex::XeTypeConverter &typeConverter;
};

} // namespace imex

#endif
