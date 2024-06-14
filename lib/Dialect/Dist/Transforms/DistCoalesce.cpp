//===- DistCoalesce.cpp - NDArrayToDist Transform  -----*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements transforms of the Dist dialect.
///
/// This pass tries to minimize the number of mesh::ShardOps.
/// Instead of creating a new copy for each repartition, it tries to combine
/// multiple RePartitionOps into one. For this, it computes the local bounding
/// box of several uses of repartitioned copies of the same base araay. It
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

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/Dist/Transforms/Passes.h>
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Dialect/NDArray/Utils/Utils.h>
#include <imex/Utils/ArithUtils.h>
#include <imex/Utils/PassUtils.h>

#include <mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Mesh/IR/MeshDialect.h>
#include <mlir/Dialect/Mesh/IR/MeshOps.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/ShapedOpInterfaces.h>

#include <iostream>
#include <set>
#include <unordered_map>

namespace imex {
#define GEN_PASS_DEF_DISTCOALESCE
#include "imex/Dialect/Dist/Transforms/Passes.h.inc"
} // namespace imex

namespace imex {
namespace dist {

namespace {

bool isCreator(::mlir::Operation *op) {
  return ::mlir::isa<::mlir::tensor::EmptyOp>(op);
}

bool isElementwise(::mlir::Operation *op) {
  return ::mlir::isa<::mlir::linalg::AbsOp,
                    ::mlir::linalg::AddOp,
                    ::mlir::linalg::CeilOp,
                    ::mlir::linalg::Conv3DOp,
                    ::mlir::linalg::CopyOp,
                    ::mlir::linalg::DivOp,
                    ::mlir::linalg::DivUnsignedOp,
                    ::mlir::linalg::ElemwiseBinaryOp,
                    ::mlir::linalg::ElemwiseUnaryOp,
                    ::mlir::linalg::ErfOp,
                    ::mlir::linalg::ExpOp,
                    ::mlir::linalg::FillOp,
                    ::mlir::linalg::FillRng2DOp,
                    ::mlir::linalg::FloorOp,
                    ::mlir::linalg::LogOp,
                    ::mlir::linalg::MapOp,
                    ::mlir::linalg::MaxOp,
                    ::mlir::linalg::MinOp,
                    ::mlir::linalg::MulOp,
                    ::mlir::linalg::NegFOp,
                    ::mlir::linalg::PowFOp,
                    ::mlir::linalg::ReciprocalOp,
                    ::mlir::linalg::RoundOp,
                    ::mlir::linalg::RsqrtOp,
                    ::mlir::linalg::SqrtOp,
                    ::mlir::linalg::SquareOp,
                    ::mlir::linalg::SubOp,
                    ::mlir::linalg::TanhOp>(op);
}

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Lowering dist dialect by no-ops
struct DistCoalescePass
    : public imex::impl::DistCoalesceBase<DistCoalescePass> {

  DistCoalescePass() = default;

#if 0
  // returns true if a Value is defined by any of the given operation types
  template <typename T, typename... Ts>
  static ::mlir::Operation *isDefByAnyOf(const ::mlir::Value &val) {
    if (auto res = val.getDefiningOp<T>())
      return res;
    if constexpr (sizeof...(Ts))
      return isDefByAnyOf<Ts...>(val);
    else if constexpr (!sizeof...(Ts))
      return nullptr;
  }

  // returns true if an operation is of any of the given types
  template <typename T, typename... Ts>
  static bool isAnyOf(const ::mlir::Operation *op) {
    if (::mlir::dyn_cast<T>(op))
      return true;
    if constexpr (sizeof...(Ts))
      return isAnyOf<Ts...>(op);
    else if constexpr (!sizeof...(Ts))
      return false;
  }

  static bool isCreator(::mlir::Operation *op) {
    return op &&
           isAnyOf<::imex::ndarray::LinSpaceOp, ::imex::ndarray::CreateOp>(op);
  }

  /// return true if given op comes from a EWOp and has another EWOp
  /// as its single user.
  bool is_temp(::mlir::mesh::ShardOp &op) {
    if (!op->hasAttr("target") && op->hasOneUse() &&
        ::mlir::isa<::imex::dist::EWBinOp, ::imex::dist::EWUnyOp>(
            *op->user_begin()) &&
        ::mlir::isa<::imex::dist::EWBinOp, ::imex::dist::EWUnyOp>(
            op.getSrc().getDefiningOp())) {
      return true;
    }
    return false;
  }

  /// update a SubviewOp with a target part
  /// create and return a new op if the SubviewOp has more than one use.
  ::mlir::Operation *updateTargetPart(::mlir::IRRewriter &builder,
                                      ::imex::dist::SubviewOp op,
                                      const ::mlir::ValueRange &tOffs,
                                      const ::mlir::ValueRange &tSizes) {

    // check if an existing target is the same as ours
    auto offs = op.getTargetOffsets();
    auto szs = op.getTargetSizes();
    if (offs.size() > 0) {
      assert(offs.size() == szs.size());
      ::mlir::SmallVector<::mlir::Operation *> toBeMoved;
      for (size_t i = 0; i < offs.size(); ++i) {
        if ((tOffs[i] != offs[i] || tSizes[i] != szs[i]) && !op->hasOneUse()) {
          // existing but different target -> need a new repartition for our
          // back-propagation
          auto val = op.getSource();
          builder.setInsertionPointAfter(op);

          auto tmp = tOffs[0].getDefiningOp();
          auto &dom = this->getAnalysis<::mlir::DominanceInfo>();
          if (!dom.dominates(tmp, op)) {
            toBeMoved.resize(0);
            if (canMoveAfter(dom, tmp, op, toBeMoved)) {
              ::mlir::Operation *curr = op;
              for (auto dop : toBeMoved) {
                dop->moveAfter(curr);
                curr = dop;
              }
              builder.setInsertionPointAfter(curr);
            } else {
              assert(false && "Not implemented");
            }
          }
          assert(tOffs.size() == tSizes.size());
          auto dynPtType = cloneWithDynEnv(
              mlir::cast<::imex::ndarray::NDArrayType>(val.getType()));
          return builder.create<::imex::mesh::ShardOp>(
              op->getLoc(), dynPtType, val, tOffs, tSizes);
        }
      }
      // if same existing target -> nothing to be done
    } else {
      const int32_t rank = static_cast<int32_t>(tOffs.size());
      const int32_t svRank = op.getStaticSizes().size();
      const bool hasUnitSize =
          mlir::cast<::imex::ndarray::NDArrayType>(op.getResult().getType())
              .hasUnitSize();

      if (svRank == rank || hasUnitSize) {
        if (hasUnitSize) {
          // Here the subview can have a different rank than the target.
          // The target can be empty (all dims have size zero) for example when
          // the source insert_slice is unit-sized and happens on a different
          // prank. In such cases we need to have all zeros in our target (of
          // rank svRank). Otherwise the target size is 1.
          mlir::OpBuilder::InsertionGuard guard(builder);
          if (rank) {
            builder.setInsertionPointAfter(tSizes[0].getDefiningOp());
          } else {
            builder.setInsertionPoint(op);
          }

          // first compute total size of target
          auto loc = op->getLoc();
          auto zero = easyIdx(loc, builder, 0);
          auto one = easyIdx(loc, builder, 1);
          auto sz = one;
          for (auto r = 0; r < rank; ++r) {
            sz = sz * easyIdx(loc, builder, tSizes[r]);
          }
          // check if the target has total size 0
          sz = sz.eq(zero).select(zero, one);
          op->insertOperands(op->getNumOperands(),
                             ::imex::ValVec(svRank, zero.get()));
          op->insertOperands(op->getNumOperands(),
                             ::imex::ValVec(svRank, sz.get()));
        } else {
          // no existing target -> use ours
          op->insertOperands(op->getNumOperands(), tOffs);
          op->insertOperands(op->getNumOperands(), tSizes);
        }

        const auto sSzsName = op.getOperandSegmentSizesAttrName();
        const auto oa = op->getAttrOfType<::mlir::DenseI32ArrayAttr>(sSzsName);
        ::std::array<int32_t, 6> sSzs{oa[0], oa[1],  oa[2],
                                      oa[3], svRank, svRank};
        op->setAttr(sSzsName, builder.getDenseI32ArrayAttr(sSzs));
      } else {
        assert(false && "found dependent operation with different rank, needs "
                        "broadcasting support?");
      }
    }
    return nullptr;
  }

  /// clone subviewops which are returned and mark them "final"
  /// Needed to protect them from being "redirected" to a reparitioned copy
  void backPropagateReturn(::mlir::IRRewriter &builder,
                           ::mlir::func::ReturnOp retOp) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(retOp);
    bool altered = false;
    ::imex::ValVec oprnds;
    ::mlir::SmallVector<::mlir::Operation *> toErase;
    for (auto val : retOp->getOperands()) {
      if (isDist(val)) {
        bool oneUse = true;
        // "skip" casts and observe if this is a single-use chain
        auto castOp = val.getDefiningOp<::mlir::UnrealizedConversionCastOp>();
        while (castOp && castOp.getInputs().size() == 1) {
          if (!castOp->hasOneUse()) {
            oneUse = false;
          }
          val = castOp.getInputs().front();
          castOp = val.getDefiningOp<::mlir::UnrealizedConversionCastOp>();
        }

        if (auto typedOp = val.getDefiningOp<::imex::dist::SubviewOp>()) {
          auto iOp = builder.clone(*typedOp);
          iOp->setAttr("final", builder.getUnitAttr());
          if (oneUse && typedOp->hasOneUse()) {
            toErase.emplace_back(typedOp);
          }
          oprnds.emplace_back(iOp->getResult(0));
          altered = true;
          continue;
        }
      }
      oprnds.emplace_back(val);
    }
    if (altered) {
      retOp->setOperands(oprnds);
      for (auto op : toErase) {
        op->erase();
      }
    }
  }
#endif

  /// Follow def-chain of given Value until hitting a creation function
  /// or array-returning EWBinOp or EWUnyOp et al
  /// @return defining op
  ::mlir::Operation *getBaseArray(const ::mlir::Value &val) {
    auto defOp = val.getDefiningOp();
    assert(defOp || !"Cannot get base array for a null value");

    if (isElementwise(defOp) || isCreator(defOp)) {
      return defOp;
    } else if (auto op = ::mlir::dyn_cast<::imex::ndarray::SubviewOp>(defOp)) {
      return getBaseArray(op.getSource());
    } else if (auto op =
                  ::mlir::dyn_cast<::imex::ndarray::InsertSliceOp>(defOp)) {
      return getBaseArray(op.getDestination());
    } else if (auto op = ::mlir::dyn_cast<::mlir::mesh::ShardOp>(defOp)) {
      // this is the only place where we expect block args
      return op.getSrc().getDefiningOp() ? getBaseArray(op.getSrc()) : op;
    } else if (auto op = ::mlir::dyn_cast<::mlir::UnrealizedConversionCastOp>(
                  defOp)) {
      if (op.getInputs().size() == 1) {
        return getBaseArray(op.getInputs().front());
      }
      return defOp;
    } else {
      std::cerr << "oops. Unexpected op found: ";
      const_cast<::mlir::Value &>(val).dump();
      assert(false);
    }

    return nullptr;
  }
  

  /// The actual back propagation of target parts
  /// if meeting a supported op, recursively gets defining ops and back
  /// propagates as it follows only supported ops, all other ops act as
  /// propagation barriers (e.g. InsertSliceOps) on the way it updates target
  /// info on SubviewOps and marks shardOps for elimination
  bool backPropagateShardSizes(::mlir::IRRewriter &builder, ::mlir::Operation *op,
                               const ::mlir::mesh::MeshSharding &sharding,
                               ::mlir::Operation *&nOp) {
    nOp = nullptr;
    if (op == nullptr) return false;

    auto assignSharding = [&](::mlir::Operation *op, const ::mlir::mesh::MeshSharding& sh) -> bool {
      if (auto typedOp = ::mlir::dyn_cast<::mlir::mesh::ShardOp>(op)) {
        ::mlir::mesh::MeshSharding currSharding(typedOp.getSharding());
        if(currSharding.equalSplitAndPartialAxes(sharding)) {
          assert(currSharding.getStaticHaloSizes().empty());
          if(!currSharding.equalHaloAndShardSizes(sharding)) {
            builder.setInsertionPoint(op);
            auto newSharding = builder.create<::mlir::mesh::ShardingOp>(op->getLoc(), sh);
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
        if (modified) backPropagateShardSizes(builder, typedOp.getSrc().getDefiningOp(), sharding, nOp);
      } else {
        auto srcOp = typedOp.getSrc().getDefiningOp();
        if (srcOp && backPropagateShardSizes(builder, typedOp.getSrc().getDefiningOp(), sharding, nOp)) {
          assignSharding(typedOp, sharding);
        }
      }
    } else if (isElementwise(op) || isCreator(op)) {
      modified = isCreator(op);
      for (auto oprnd : op->getOperands()) {
        if (::mlir::isa<::mlir::RankedTensorType>(oprnd.getType())) {
          modified |= backPropagateShardSizes(builder, oprnd.getDefiningOp(), sharding, nOp);
        }
      }
    } else if(auto typedOp = ::mlir::dyn_cast<::imex::ndarray::SubviewOp>(op)) {
      modified = backPropagateShardSizes(builder, typedOp.getSource().getDefiningOp(), sharding, nOp);
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
    backPropagateShardSizes(builder, op.getSrc().getDefiningOp(), sharding, nOp);
    assert(nOp == nullptr);
    return;
  }

  // return ShardOp that annotates the result of a given op
  ::mlir::mesh::ShardOp getShardOp(::mlir::Operation *op) {
    if (auto typedOp = ::mlir::dyn_cast<::mlir::mesh::ShardOp>(op)) {
      return typedOp;
    }
    if(op->hasOneUse()) {
      op = *op->user_begin();
      if (::mlir::isa<::mlir::UnrealizedConversionCastOp>(op)) {
        assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
        assert(op->hasOneUse());
        op = *op->user_begin();
      }
    } else {
      assert(op->use_empty());
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
#if 0

  /// compute target part for a given InsertSliceOp
  ::imex::dist::TargetOfSliceOp computeTarget(::mlir::IRRewriter &builder,
                                              ::imex::ndarray::InsertSliceOp op,
                                              ::mlir::Value sharding) {
    auto shardingOp =
        ::mlir::cast<::mlir::mesh::ShardingOp>(sharding.getDefiningOp());
    auto sOffs = op.getStaticOffsets();
    auto sSizes = op.getStaticSizes();
    auto sStrides = op.getStaticStrides();
    assert(!(::mlir::ShapedType::isDynamicShape(sSizes) ||
             ::mlir::ShapedType::isDynamicShape(sOffs) ||
             ::mlir::ShapedType::isDynamicShape(sStrides)) ||
           (false && "SubviewOp must have dynamic offsets, sizes and strides"));

    auto src = getShardOpOfOperand(op.getDestination()).getSrc();
    return builder.create<::imex::dist::TargetOfSliceOp>(
        op->getLoc(), src, sOffs, sSizes, sStrides, shardingOp.getMeshAttr(),
        shardingOp.getSplitAxes());
  }
#endif // 0

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
    std::unordered_map<::mlir::Operation *, ::mlir::Operation *> baseIPts;

    root->walk([&](::mlir::Operation *op) {
      ::mlir::Value val;
      if (auto typedOp = ::mlir::dyn_cast<::imex::ndarray::InsertSliceOp>(op)) {
        val = typedOp.getDestination();
      } else if (auto typedOp =
                     ::mlir::dyn_cast<::imex::ndarray::SubviewOp>(op)) {
        val = typedOp.getSource();
      }
      if (val) {
        auto base = getBaseArray(val);
        baseIPts.emplace(base, getShardOp(base));
        opsGroups[base].emplace_back(op);

        // for InsertSliceOps compute and propagate target parts
        if (auto typedOp =
                ::mlir::dyn_cast<::imex::ndarray::InsertSliceOp>(op)) {
          builder.setInsertionPointAfter(baseIPts[base]);
          auto srcop =
              typedOp.getSource().getDefiningOp<::mlir::mesh::ShardOp>();
          assert(srcop && "InsertSliceOp must have a ShardOp as source");
          assert(srcop.getAnnotateForUsers());
          backPropagateShardSizes(builder, srcop);
        }
      }
    });

    ::mlir::SymbolTableCollection symbolTable;
    // outer loop iterates base over base pointers
    for (auto grpP : opsGroups) {
      if (grpP.second.empty())
        continue;

      auto &base = grpP.first;
      auto baseShape = ::mlir::cast<::mlir::ShapedType>(base->getResult(0).getType());
      assert(baseShape.hasStaticShape() && "Base array must have static shape");

      auto shardOp = getShardOp(base);
      ::mlir::SmallVector<::mlir::mesh::ShardOp> shardOps;
      ::mlir::ValueRange halos;
      int numHalos = 0;
      for (auto axes : shardOp.getSharding().getDefiningOp<::mlir::mesh::ShardingOp>().getSplitAxes()) {
        if (!axes.empty()) {
          ++numHalos;
        }
      }
      ::mlir::SmallVector<::mlir::Type> haloResultTypes(numHalos*2, builder.getI64Type());

      for (auto currOp : grpP.second) {
        // collect SubviewOps until we meet a InsertSliceOp
        if (auto subviewOp = ::mlir::dyn_cast<::imex::ndarray::SubviewOp>(*currOp)) {
          if (!subviewOp->hasAttr("final")) {
            auto sOffs = subviewOp.getStaticOffsets();
            auto sSizes = subviewOp.getStaticSizes();
            auto sStrides = subviewOp.getStaticStrides();
            assert(
                !(::mlir::ShapedType::isDynamicShape(sSizes) ||
                  ::mlir::ShapedType::isDynamicShape(sOffs) ||
                  ::mlir::ShapedType::isDynamicShape(sStrides))
                && "SubviewOp must have static offsets, sizes and strides");
            if(auto svShardOp = getShardOp(subviewOp)) {
              auto svShardingOp = svShardOp.getSharding().getDefiningOp<::mlir::mesh::ShardingOp>();
              assert(svShardingOp);
              auto target = svShardingOp.getStaticShardedDimsOffsets();
              assert(!::mlir::ShapedType::isDynamicShape(target) && "ShardOp of Subview must have static sharded dims sizes");
              auto mesh = svShardingOp.getMeshAttr().getValue();
              builder.setInsertionPoint(shardOp);
              halos = builder.create<::imex::dist::ExtendHaloForSliceOp>(subviewOp->getLoc(), haloResultTypes, baseShape.getShape(), mesh, svShardingOp.getSplitAxes(), halos, sOffs, sSizes, sStrides, target).getResult();
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
      auto orgSharding = shardOp.getSharding().getDefiningOp<::mlir::mesh::ShardingOp>();
      builder.setInsertionPointAfter(shardOp);
      auto newSharding = builder.create<::mlir::mesh::ShardingOp>(
        shardOp->getLoc(), ::mlir::mesh::ShardingType::get(shardOp->getContext()),
        orgSharding.getMeshAttr(), orgSharding.getSplitAxesAttr(), orgSharding.getPartialAxesAttr(), orgSharding.getPartialTypeAttr(),
        ::mlir::DenseI64ArrayAttr::get(shardOp->getContext(), {}), ::mlir::ValueRange{},
        ::mlir::DenseI64ArrayAttr::get(shardOp->getContext(), ::mlir::SmallVector<int64_t>(halos.size(), ::mlir::ShapedType::kDynamic)), halos);
      auto newShardOp = builder.create<::mlir::mesh::ShardOp>(
          shardOp->getLoc(), shardOp, newSharding.getResult());

      // update shardOps of dependent Subview/InsertSliceOps
      for (auto svShardOp : shardOps) {
        svShardOp.getSrcMutable().assign(newShardOp.getResult());
        svShardOp.getShardingMutable().assign(newSharding);
      }
      // barriers/halo-updates get inserted when InsertSliceOps (or other write ops) get spmdized
    } // for (auto grpP : opsGroups)

    mlir::SmallVector<mlir::mesh::ShardingOp> prevShardings;
    root->walk([&](::mlir::mesh::ShardingOp op) {
      for(auto prev : prevShardings) {
        if(mlir::mesh::MeshSharding(op.getResult()) == mlir::mesh::MeshSharding(prev.getResult())) {
          builder.replaceOp(op, prev.getResult());
          op = nullptr;
          break;
        }
      }
      if(op) {
        prevShardings.emplace_back(op);
      }
    });
  } // runOnOperation
}; // DistCoalescePass
} // namespace
} // namespace dist

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::Pass> createDistCoalescePass() {
  return std::make_unique<::imex::dist::DistCoalescePass>();
}

} // namespace imex
