//===- CoalesceShardOps.cpp - CoalesceShardingOps Transform -----*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a transform of Mesh and NDArray dialects.
///
/// This pass tries to minimize the number of mesh::ShardOps.
/// Instead of creating a new copy for each repartition, it tries to combine
/// multiple RePartitionOps into one. For this, it computes the local bounding
/// box of several uses of repartitioned copies of the same base array. It
/// replaces all matched RepartitionOps with one which provides the computed
/// bounding box. Uses of the eliminated RePartitionOps get updated with th
/// appropriate target part as originally used. Right now supported uses are
/// SubviewOps and InsertSliceOps.
///
/// InsertSliceOps are special because they mutate data. Hence they serve as
/// barriers across which no combination of RePartitionOps will happen.
///
/// Additionally, while most other ops do not request a special target part,
/// InsertSliceOps request a target part on the incoming array. This target
/// part gets back-propagated as far as possible, most importantly including
/// EWBinOps.
///
/// Also, as part of this back-propagation, RePartitionOps between two EWBinOps,
/// e.g. those which come from one EWBinOp and have only one use and that in a
/// another EWBinOp get simply erased.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Dialect/NDArray/Utils/Utils.h>
#include <imex/Utils/ArithUtils.h>
#include <imex/Utils/PassUtils.h>

#include <mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Mesh/IR/MeshDialect.h>
#include <mlir/Dialect/Mesh/IR/MeshOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/ShapedOpInterfaces.h>
#include <mlir/Pass/Pass.h>

#include <iostream>
#include <set>
#include <unordered_map>

namespace imex {
#define GEN_PASS_DEF_COALESCESHARDOPS
#include "imex/Dialect/NDArray/Transforms/Passes.h.inc"

namespace {

bool isCreator(::mlir::Operation *op) {
  return ::mlir::isa<::mlir::tensor::EmptyOp, ::mlir::tensor::SplatOp>(op);
}

bool isElementwise(::mlir::Operation *op) {
  return ::mlir::isa<
      ::mlir::tosa::AddOp, ::mlir::tosa::ArithmeticRightShiftOp,
      ::mlir::tosa::BitwiseAndOp, ::mlir::tosa::BitwiseOrOp,
      ::mlir::tosa::BitwiseXorOp, ::mlir::tosa::IntDivOp,
      ::mlir::tosa::LogicalAndOp, ::mlir::tosa::LogicalLeftShiftOp,
      ::mlir::tosa::LogicalRightShiftOp, ::mlir::tosa::LogicalOrOp,
      ::mlir::tosa::LogicalXorOp, ::mlir::tosa::MaximumOp,
      ::mlir::tosa::MinimumOp, ::mlir::tosa::MulOp, ::mlir::tosa::PowOp,
      ::mlir::tosa::SubOp, ::mlir::tosa::SelectOp, ::mlir::tosa::EqualOp,
      ::mlir::tosa::GreaterOp, ::mlir::tosa::GreaterEqualOp,
      ::mlir::tosa::ClampOp, ::mlir::tosa::SigmoidOp, ::mlir::tosa::TanhOp,
      ::mlir::tosa::ErfOp, ::mlir::tosa::AbsOp, ::mlir::tosa::BitwiseNotOp,
      ::mlir::tosa::CeilOp, ::mlir::tosa::ClzOp, ::mlir::tosa::CosOp,
      ::mlir::tosa::ExpOp, ::mlir::tosa::FloorOp, ::mlir::tosa::LogOp,
      ::mlir::tosa::LogicalNotOp, ::mlir::tosa::NegateOp,
      ::mlir::tosa::ReciprocalOp, ::mlir::tosa::RsqrtOp, ::mlir::tosa::SinOp,
      ::mlir::linalg::AbsOp, ::mlir::linalg::AddOp, ::mlir::linalg::CeilOp,
      ::mlir::linalg::Conv3DOp, ::mlir::linalg::CopyOp, ::mlir::linalg::DivOp,
      ::mlir::linalg::DivUnsignedOp, ::mlir::linalg::ErfOp,
      ::mlir::linalg::ExpOp, ::mlir::linalg::FillOp,
      ::mlir::linalg::FillRng2DOp, ::mlir::linalg::FloorOp,
      ::mlir::linalg::LogOp, ::mlir::linalg::MapOp, ::mlir::linalg::MaxOp,
      ::mlir::linalg::MinOp, ::mlir::linalg::MulOp, ::mlir::linalg::NegFOp,
      ::mlir::linalg::PowFOp, ::mlir::linalg::ReciprocalOp,
      ::mlir::linalg::RoundOp, ::mlir::linalg::RsqrtOp, ::mlir::linalg::SqrtOp,
      ::mlir::linalg::SquareOp, ::mlir::linalg::SubOp, ::mlir::linalg::TanhOp>(
      op);
}

// *******************************
// ***** Pass infrastructure *****
// *******************************

struct CoalesceShardOpsPass
    : public imex::impl::CoalesceShardOpsBase<CoalesceShardOpsPass> {

  CoalesceShardOpsPass() = default;

  /// Follow def-chain of given Value until hitting a creation function
  /// or array-returning EWBinOp or EWUnyOp et al
  /// @return defining op
  ::mlir::Operation *getBaseArray(const ::mlir::Value &val) {
    auto defOp = val.getDefiningOp();

    if (!defOp) {
      return nullptr;
    } else if (auto op = ::mlir::dyn_cast<::mlir::mesh::ShardOp>(defOp)) {
      // this is the only place where we expect block args
      return op.getSrc().getDefiningOp() ? getBaseArray(op.getSrc()) : op;
    } else if (isCreator(defOp)) {
      return defOp;
    } else if (auto op = ::mlir::dyn_cast<::mlir::DestinationStyleOpInterface>(
                   defOp)) {
      return op.getNumDpsInits() == 1 ? getBaseArray(op.getDpsInits()[0])
                                      : defOp;
    } else if (auto op = ::mlir::dyn_cast<::imex::ndarray::SubviewOp>(defOp)) {
      return getBaseArray(op.getSource());
    } else if (auto op =
                   ::mlir::dyn_cast<::imex::ndarray::InsertSliceOp>(defOp)) {
      return getBaseArray(op.getDestination());
    } else if (auto op = ::mlir::dyn_cast<::mlir::UnrealizedConversionCastOp>(
                   defOp)) {
      if (op.getInputs().size() == 1) {
        return getBaseArray(op.getInputs().front());
      }
    }
    return defOp;
  }

  /// The actual back propagation of target parts
  /// if meeting a supported op, recursively gets defining ops and back
  /// propagates as it follows only supported ops, all other ops act as
  /// propagation barriers (e.g. InsertSliceOps) on the way it
  /// updates target info on SubviewOps
  bool backPropagateShardSizes(::mlir::IRRewriter &builder,
                               ::mlir::Operation *op,
                               const ::mlir::mesh::MeshSharding &sharding,
                               ::mlir::Operation *&nOp) {
    nOp = nullptr;
    if (op == nullptr)
      return false;

    auto assignSharding = [&](::mlir::Operation *op,
                              const ::mlir::mesh::MeshSharding &sh) -> bool {
      if (auto typedOp = ::mlir::dyn_cast<::mlir::mesh::ShardOp>(op)) {
        ::mlir::mesh::MeshSharding currSharding(typedOp.getSharding());
        if (currSharding.equalSplitAndPartialAxes(sharding)) {
          assert(currSharding.getStaticHaloSizes().empty());
          if (!currSharding.equalHaloAndShardSizes(sharding)) {
            builder.setInsertionPoint(op);
            auto newSharding =
                builder.create<::mlir::mesh::ShardingOp>(op->getLoc(), sh);
            typedOp.getShardingMutable().assign(newSharding.getResult());
            return true;
          }
        }
      }
      return false;
    };

    bool modified = false;
    if (auto typedOp = ::mlir::dyn_cast<::mlir::mesh::ShardOp>(op)) {
      if (typedOp.getAnnotateForUsers()) {
        modified = assignSharding(typedOp, sharding);
        if (modified)
          backPropagateShardSizes(builder, typedOp.getSrc().getDefiningOp(),
                                  sharding, nOp);
      } else {
        auto srcOp = typedOp.getSrc().getDefiningOp();
        if (srcOp &&
            backPropagateShardSizes(builder, typedOp.getSrc().getDefiningOp(),
                                    sharding, nOp)) {
          assignSharding(typedOp, sharding);
        }
      }
    } else if (isElementwise(op) || isCreator(op)) {
      modified = isCreator(op);
      for (auto oprnd : op->getOperands()) {
        if (::mlir::isa<::mlir::RankedTensorType>(oprnd.getType())) {
          modified |= backPropagateShardSizes(builder, oprnd.getDefiningOp(),
                                              sharding, nOp);
        }
      }
    } else if (auto typedOp =
                   ::mlir::dyn_cast<::imex::ndarray::SubviewOp>(op)) {
      modified = true;
      // backPropagateShardSizes(
      //     builder, typedOp.getSource().getDefiningOp(), sharding, nOp);
    }

    assert(nOp == nullptr);
    return modified;
  }

  /// entry point for back propagation of target shardings.
  void backPropagateShardSizes(::mlir::IRRewriter &builder,
                               ::mlir::mesh::ShardOp op) {
    ::mlir::Operation *nOp = nullptr;
    auto sharding = op.getSharding();
    assert(sharding);
    backPropagateShardSizes(builder, op.getSrc().getDefiningOp(), sharding,
                            nOp);
    assert(nOp == nullptr);
    return;
  }

  // return ShardOp that annotates the result of a given op
  ::mlir::mesh::ShardOp getShardOp(::mlir::Operation *op) {
    if (auto typedOp = ::mlir::dyn_cast<::mlir::mesh::ShardOp>(op)) {
      return typedOp;
    }
    if (!op->hasOneUse()) {
      return {};
    }
    op = *op->user_begin();
    if (::mlir::isa<::mlir::UnrealizedConversionCastOp>(op)) {
      assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
      assert(op->hasOneUse());
      op = *op->user_begin();
    }
    return ::mlir::dyn_cast<::mlir::mesh::ShardOp>(op);
  }

  // return ShardOp that annotates the given operand/value
  ::mlir::mesh::ShardOp getShardOpOfOperand(::mlir::Value val) {
    auto op = val.getDefiningOp();
    // FIXME as long as we have NDArrays we might meet casts
    if (::mlir::isa<::mlir::UnrealizedConversionCastOp>(op)) {
      assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
      assert(op->hasOneUse() && op->getNumOperands() == 1);
      op = op->getOperand(0).getDefiningOp();
    }
    assert(op->hasOneUse());
    return ::mlir::dyn_cast<::mlir::mesh::ShardOp>(op);
  }

  void backPropagateBaseSharding(const ::mlir::Value &val,
                                 mlir::Value baseSharding) {
    auto defOp = val.getDefiningOp();

    if (mlir::isa<ndarray::InsertSliceOp, ndarray::SubviewOp>(defOp)) {
      return;
    } else if (auto op = ::mlir::dyn_cast<::mlir::mesh::ShardOp>(defOp)) {
      // this is the only place where we expect block args
      if (op.getSrc().getDefiningOp()) {
        if (op.getAnnotateForUsers()) {
          op.getShardingMutable().assign(baseSharding);
        }
        backPropagateBaseSharding(op.getSrc(), baseSharding);
      }
    } else if (auto op = ::mlir::dyn_cast<::mlir::DestinationStyleOpInterface>(
                   defOp)) {
      if (op.getNumDpsInits() == 1) {
        getShardOp(defOp).getShardingMutable().assign(baseSharding);
        backPropagateBaseSharding(op.getDpsInits()[0], baseSharding);
      }
    }
    return;
  }

  template <typename T>
  static auto getBaseShardDimSize(T shard, T numShards, T extend) {
    return extend / numShards +
           shard.sge(numShards - (extend % numShards)).select(1l, 0l);
  }

  static auto getBaseShardDimSize(int64_t shard, int64_t numShards,
                                  int64_t extend) {
    return extend / numShards +
           (shard >= numShards - (extend % numShards) ? 1 : 0);
  }

  static ::mlir::SmallVector<::imex::EasyI64>
  extendHaloForSliceOp(::mlir::IRRewriter &rewriter, mlir::Operation *op,
                       ::mlir::ArrayRef<int64_t> baseShape,
                       ::mlir::FlatSymbolRefAttr mesh,
                       ::mlir::mesh::MeshAxesArrayAttr splitAxes,
                       const ::mlir::SmallVector<::imex::EasyI64> &dynHaloSizes,
                       ::mlir::ArrayRef<int64_t> staticOffsets,
                       ::mlir::ArrayRef<int64_t> staticSizes,
                       ::mlir::ArrayRef<int64_t> staticStrides,
                       ::mlir::ArrayRef<int64_t> staticTargetOffsets) {

    const ::mlir::Location loc = op->getLoc();
    ::mlir::SymbolTableCollection symbolTable;
    auto meshOp = ::mlir::mesh::getMesh(op, mesh, symbolTable);
    assert(meshOp);

    // compute number of shards along split axes
    // compute sharded dims extends (element count per sharded dim of base
    // array)
    ::mlir::SmallVector<int64_t> numShards, shardedDims;
    for (auto dim = 0; dim < (int64_t)splitAxes.size(); ++dim) {
      auto axes = splitAxes.getAxes()[dim];
      if (!axes.empty()) {
        numShards.emplace_back(::mlir::mesh::collectiveProcessGroupSize(
            axes.asArrayRef(), meshOp));
        assert(!::mlir::ShapedType::isDynamic(numShards.back()));
        shardedDims.emplace_back(dim);
      }
    }

    // init halo sizes either from input or to 0
    ::mlir::SmallVector<::imex::EasyI64> haloSizes = dynHaloSizes;
    auto zero = easyI64(loc, rewriter, 0);
    auto one = easyI64(loc, rewriter, 1);
    if (haloSizes.empty()) {
      haloSizes.resize(numShards.size() * 2, zero);
    }
    assert(haloSizes.size() == numShards.size() * 2);

    // iterate split axes and compute lower/upper halo bounds for each dim
    int64_t curr = 0;
    for (size_t dim = 0; dim < numShards.size(); ++dim) {
      auto num = numShards[dim];
      auto tensorDim = shardedDims[dim];
      auto baseOff = zero;
      auto baseEnd = easyI64(loc, rewriter,
                             getBaseShardDimSize(0, num, baseShape[tensorDim]));
      auto targetOff = easyI64(loc, rewriter, staticOffsets[tensorDim]);
      auto stride = easyI64(loc, rewriter, staticStrides[tensorDim]);
      assert(staticTargetOffsets[curr] == 0);
      auto sz = staticTargetOffsets[++curr]; // first offset is always 0
      auto targetEnd =
          targetOff + zero.max(stride * easyI64(loc, rewriter, sz - 1) + one);

      // FIXME what about staticSizes?
      for (auto i = 0; i < num; ++i, ++curr) {
        auto targetSz = easyI64(loc, rewriter, sz);
        if (sz != 0) {
          haloSizes[dim * 2] = targetSz.sgt(zero).select(
              haloSizes[dim * 2].max(zero.max(baseOff - targetOff)),
              haloSizes[dim * 2]);
          haloSizes[dim * 2 + 1] = targetSz.sgt(zero).select(
              haloSizes[dim * 2 + 1].max(zero.max(targetEnd - baseEnd)),
              haloSizes[dim * 2 + 1]);
        }
        if (i + 1 >= num) {
          ++curr;
          break;
        }
        if (sz != 0) {
          targetOff = targetOff + stride * targetSz;
        }
        sz = staticTargetOffsets[curr + 1] - staticTargetOffsets[curr];
        if (sz != 0) {
          targetEnd = targetOff + stride * easyI64(loc, rewriter, sz - 1) + one;
        } else {
          targetEnd = targetOff;
        }
        baseOff = baseEnd;
        baseEnd = baseOff + easyI64(loc, rewriter,
                                    getBaseShardDimSize(i + 1, num,
                                                        baseShape[tensorDim]));
      }
    }

    return haloSizes;
  }

  // This pass tries to combine multiple ShardOps into one.
  // It does not actually erase any ops, but rather annotates some so that
  // later passes will not create actual resharding/communicating ops.
  //
  // Dependent operations (like SubviewOp) get adequately annotated.
  //
  // The basic idea is to compute a the bounding box of several SubViews
  // and use it for a combined ShardOp. Dependent SubviewOps can then
  // extract the appropriate part from that bounding box without further
  // communication/repartitioning.
  //
  // Right now we only support subviews with static indices (offs, sizes,
  // strides).
  //
  // 1. back-propagation of explicit target-parts
  // 2. group SubviewOps
  // 3. create base ShardOp and update dependent SubviewOps
  void runOnOperation() override {

    auto root = this->getOperation();
    ::mlir::IRRewriter builder(&getContext());
    ::mlir::SymbolTableCollection symbolTableCollection;

    // back-propagate targets from RePartitionOps

    // find InsertSliceOp and SubviewOp operating on the same base pointer
    // opsGroups holds independent partial operation sequences operating on a
    // specific base pointer
    // on the way compute and back-propagate target parts for InsertSliceOps

    std::unordered_map<::mlir::Operation *,
                       ::mlir::SmallVector<::mlir::Operation *>>
        opsGroups;

    auto wRes = root->walk([&](::mlir::Operation *op) -> mlir::WalkResult {
      ::mlir::Value val;
      if (auto typedOp = ::mlir::dyn_cast<::imex::ndarray::InsertSliceOp>(op)) {
        val = typedOp.getDestination();
      } else if (auto typedOp =
                     ::mlir::dyn_cast<::imex::ndarray::SubviewOp>(op)) {
        val = typedOp.getSource();
      }
      if (val) {
        auto base = getBaseArray(val);
        if (!base) {
          return mlir::WalkResult::interrupt();
        }
        auto baseShardOp = getShardOp(base);
        if (!baseShardOp) {
          return mlir::WalkResult::interrupt();
        }
        opsGroups[base].emplace_back(op);

        // for InsertSliceOps compute and propagate target parts
        if (auto typedOp =
                ::mlir::dyn_cast<::imex::ndarray::InsertSliceOp>(op)) {
          builder.setInsertionPointAfter(baseShardOp);
          auto srcShardOp =
              typedOp.getSource().getDefiningOp<::mlir::mesh::ShardOp>();
          assert(srcShardOp && "InsertSliceOp must have a ShardOp as source");
          assert(srcShardOp.getAnnotateForUsers());
          backPropagateShardSizes(builder, srcShardOp);
        }
      }
      return mlir::WalkResult::advance();
    });
    if (wRes.wasInterrupted()) {
      std::cerr << "Non-existing or incomplete sharding. Skipping coalescing "
                   "shard ops.\n";
      return;
    }

    ::mlir::SymbolTableCollection symbolTable;
    // outer loop iterates base over base pointers
    for (auto grpP : opsGroups) {
      if (grpP.second.empty())
        continue;

      auto &base = grpP.first;
      auto baseShape =
          ::mlir::cast<::mlir::ShapedType>(base->getResult(0).getType());
      assert(baseShape.hasStaticShape() && "Base array must have static shape");

      auto shardOp = getShardOp(base);
      ::mlir::SmallVector<::mlir::mesh::ShardOp> shardOps;
      ::mlir::SmallVector<::imex::EasyI64> halos;
      int numHalos = 0;
      for (auto axes : shardOp.getSharding()
                           .getDefiningOp<::mlir::mesh::ShardingOp>()
                           .getSplitAxes()) {
        if (!axes.empty()) {
          ++numHalos;
        }
      }
      ::mlir::SmallVector<::mlir::Type> haloResultTypes(numHalos * 2,
                                                        builder.getI64Type());

      for (auto currOp : grpP.second) {
        // collect SubviewOps until we meet a InsertSliceOp
        if (auto subviewOp =
                ::mlir::dyn_cast<::imex::ndarray::SubviewOp>(*currOp)) {
          if (!subviewOp->hasAttr("final")) {
            auto sOffs = subviewOp.getStaticOffsets();
            auto sSizes = subviewOp.getStaticSizes();
            auto sStrides = subviewOp.getStaticStrides();
            assert(!(::mlir::ShapedType::isDynamicShape(sSizes) ||
                     ::mlir::ShapedType::isDynamicShape(sOffs) ||
                     ::mlir::ShapedType::isDynamicShape(sStrides)) &&
                   "SubviewOp must have static offsets, sizes and strides");
            if (auto svShardOp = getShardOp(subviewOp)) {
              auto svShardingOp =
                  svShardOp.getSharding()
                      .getDefiningOp<::mlir::mesh::ShardingOp>();
              assert(svShardingOp);
              auto target = svShardingOp.getStaticShardedDimsOffsets();
              assert(!::mlir::ShapedType::isDynamicShape(target) &&
                     "ShardOp of Subview must have static sharded dims sizes");
              builder.setInsertionPoint(shardOp);
              halos = extendHaloForSliceOp(
                  builder, subviewOp, baseShape.getShape(),
                  svShardingOp.getMeshAttr(), svShardingOp.getSplitAxes(),
                  halos, sOffs, sSizes, sStrides, target);
              shardOps.emplace_back(getShardOpOfOperand(subviewOp.getSource()));
              // subviewOps.emplace_back(subviewOp);
            }
          }
        } else if (auto insertSlcOp =
                       ::mlir::dyn_cast<::imex::ndarray::InsertSliceOp>(
                           *currOp)) {
          shardOps.emplace_back(shardOps.emplace_back(
              getShardOpOfOperand(insertSlcOp.getDestination())));
        }
      }

      // Update base sharding with halo sizes
      ::imex::ValVec haloVals;
      for (auto sz : halos) {
        haloVals.emplace_back(sz.get());
      }
      auto orgSharding =
          shardOp.getSharding().getDefiningOp<::mlir::mesh::ShardingOp>();
      bool mutateBaseSharding = shardOp.getSrc().getDefiningOp();
      if (mutateBaseSharding) {
        builder.setInsertionPoint(shardOp);
      } else {
        builder.setInsertionPointAfter(shardOp);
      }

      auto newSharding = builder.create<::mlir::mesh::ShardingOp>(
          shardOp->getLoc(),
          ::mlir::mesh::ShardingType::get(shardOp->getContext()),
          orgSharding.getMeshAttr(), orgSharding.getSplitAxesAttr(),
          orgSharding.getPartialAxesAttr(), orgSharding.getPartialTypeAttr(),
          ::mlir::DenseI64ArrayAttr::get(shardOp->getContext(), {}),
          ::mlir::ValueRange{},
          ::mlir::DenseI64ArrayAttr::get(
              shardOp->getContext(),
              ::mlir::SmallVector<int64_t>(haloVals.size(),
                                           ::mlir::ShapedType::kDynamic)),
          haloVals);

      auto newShardOp = shardOp;
      if (mutateBaseSharding) {
        // FIXME: what about a second visit of the same base with different
        // sharding
        newShardOp.getShardingMutable().assign(newSharding.getResult());
      } else { // block arg
        newShardOp = builder.create<::mlir::mesh::ShardOp>(
            shardOp->getLoc(), shardOp, newSharding.getResult());
      }

      // update shardOps of dependent Subview/InsertSliceOps
      for (auto svShardOp : shardOps) {
        backPropagateBaseSharding(svShardOp, newSharding.getResult());
        // svShardOp.getShardingMutable().assign(newSharding);
        // assert(svShardOp->hasOneUse());
        if (auto typedOp = mlir::dyn_cast<::imex::ndarray::InsertSliceOp>(
                *svShardOp->user_begin())) {
          getShardOp(typedOp).getShardingMutable().assign(newSharding);
        }
      }
      // barriers/halo-updates get inserted when InsertSliceOps (or other write
      // ops) get spmdized
    } // for (auto grpP : opsGroups)

    mlir::SmallVector<mlir::mesh::ShardingOp> prevShardings;
    root->walk([&](::mlir::mesh::ShardingOp op) {
      for (auto prev : prevShardings) {
        if (mlir::mesh::MeshSharding(op.getResult()) ==
            mlir::mesh::MeshSharding(prev.getResult())) {
          builder.replaceOp(op, prev.getResult());
          op = nullptr;
          break;
        }
      }
      if (op) {
        prevShardings.emplace_back(op);
      }
    });
  } // runOnOperation
};
} // namespace

std::unique_ptr<::mlir::Pass> createCoalesceShardOpsPass() {
  return std::make_unique<::imex::CoalesceShardOpsPass>();
}

} // namespace imex
