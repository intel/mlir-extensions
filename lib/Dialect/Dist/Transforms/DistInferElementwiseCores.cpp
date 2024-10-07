//===- DistInferEWCores.cpp - DistInferEWCores Transform  ------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements inferring core loops for elementwise operations.
//===----------------------------------------------------------------------===//

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/Dist/Transforms/Passes.h>
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Dialect/NDArray/Utils/Utils.h>
#include <imex/Utils/PassUtils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <algorithm>
#include <cassert>
#include <set>
#include <unordered_map>
#include <vector>

namespace imex {
#define GEN_PASS_DEF_DISTINFEREWCORES
#include "imex/Dialect/Dist/Transforms/Passes.h.inc"
} // namespace imex

namespace imex {
namespace dist {

namespace {

struct DistInferEWCoresPass
    : public imex::impl::DistInferEWCoresBase<DistInferEWCoresPass> {

  DistInferEWCoresPass() = default;

  static bool isEW(::mlir::Operation *op) { // FIXME use interface or such
    return (::mlir::isa<::imex::dist::EWBinOp>(op) ||
            ::mlir::isa<::imex::dist::EWUnyOp>(op));
  };

  static bool hasCore(::mlir::Operation *op) {
    assert(isEW(op));
    return op->getNumOperands() > 2; // FIXME use interface or such
  }

  std::tuple<::imex::ValVec, ::imex::ValVec, ::imex::ValVec> static getCore(
      ::mlir::Operation *op) {
    if (auto typedOp = ::mlir::dyn_cast<::imex::dist::EWBinOp>(op)) {
      return {typedOp.getCoreOffsets(), typedOp.getCoreSizes(),
              typedOp.getTargetOffsets()};
    } else if (auto typedOp = ::mlir::dyn_cast<::imex::dist::EWUnyOp>(op)) {
      return {typedOp.getCoreOffsets(), typedOp.getCoreSizes(),
              typedOp.getTargetOffsets()};
    }
    assert("Expected ewop");
    return {};
  }

  // Adds local core to all dependent ewops of given ewop.
  // Dependent ewops are the ewop itself, operands and users.
  // Stops when visiting non-ewop.
  // Adds visited ewops to visited.
  // Adds ewops to alien if the ewop has a different core.
  void propagateAddLocalCore(
      ::mlir::IRRewriter &builder, ::mlir::Operation *op,
      ::mlir::Operation *lcOp, ::imex::ValVec &coreOffs,
      ::imex::ValVec &coreSzs, ::imex::ValVec &targetOffs,
      ::std::set<::mlir::Operation *> &visited,
      ::std::set<::mlir::Operation *, ::imex::opOrderCmp> &alien) {
    auto &dom = this->getAnalysis<::mlir::DominanceInfo>();
    if (!dom.dominates(lcOp, op)) {
      return;
    }

    // add core to all ewops that we visited
    if (hasCore(op)) {
      auto core = getCore(op);
      auto aCoreOffs = std::get<0>(core);
      auto aCoreSzs = std::get<1>(core);
      auto aTargetOffs = std::get<2>(core);
      if (coreOffs != aCoreOffs || coreSzs != aCoreSzs ||
          targetOffs != aTargetOffs) {
        alien.emplace(op);
      }
    } else {
      op->insertOperands(op->getNumOperands(), coreOffs);
      op->insertOperands(op->getNumOperands(), coreSzs);
      op->insertOperands(op->getNumOperands(), targetOffs);
    }
    visited.emplace(op);

    // we need to back-propagate to operands
    for (int i = 0; i < (mlir::isa<imex::dist::EWBinOp>(op) ? 2 : 1); ++i) {
      auto oprnd = op->getOperand(i).getDefiningOp();
      if (oprnd && isEW(oprnd) && visited.find(oprnd) == visited.end()) {
        propagateAddLocalCore(builder, oprnd, lcOp, coreOffs, coreSzs,
                              targetOffs, visited, alien);
      }
    }

    // forward to dependent uses
    for (auto user : op->getUsers()) {
      if (isEW(user) && visited.find(user) == visited.end()) {
        propagateAddLocalCore(builder, user, lcOp, coreOffs, coreSzs,
                              targetOffs, visited, alien);
      }
    }
  }

  // pull operation up to last defining op/producer
  void pullOp(::mlir::Operation *op, ::mlir::Operation *barrier = nullptr) {
    std::vector<::mlir::Operation *> deps;
    for (auto o : op->getOperands()) {
      auto defOp = o.getDefiningOp();
      if (defOp) {
        deps.emplace_back(defOp);
      }
    }
    if (barrier) {
      deps.emplace_back(barrier);
    }
    assert(!deps.empty());
    auto &dom = this->getAnalysis<::mlir::DominanceInfo>();
    std::sort(deps.begin(), deps.end(), ::imex::opOrderCmp(dom));
    op->moveAfter(deps.back());
  }

  void runOnOperation() override {
    auto root = this->getOperation();
    // first run canonicalizer
    ::mlir::PassManager pm(&getContext());
    // Add the canonicalization pass.
    pm.addPass(::mlir::createCanonicalizerPass());
    // Run the PassManager.
    if (::mlir::failed(pm.run(root))) {
      signalPassFailure();
    }

    ::mlir::IRRewriter builder(&getContext());
    ::mlir::SmallVector<::mlir::Operation *> rpOps, bbOps;

    // find all ewops
    root->walk([&](::mlir::Operation *op) {
      if (auto typedOp = ::mlir::dyn_cast<::imex::dist::RePartitionOp>(op)) {
        auto arTyp = mlir::cast<::imex::ndarray::NDArrayType>(
            typedOp.getResult().getType());
        if (!arTyp.hasUnitSize() && !arTyp.hasZeroSize()) {
          rpOps.emplace_back(op);
        }
      } else if (::mlir::isa<::imex::dist::LocalTargetOfSliceOp,
                             ::imex::dist::LocalBoundingBoxOp>(op)) {
        bbOps.emplace_back(op);
      }
    });

    // pull up all LocalTargetOfSliceOps and BoundingBoxOps as much as possible.
    for (auto bbOp : bbOps) {
      pullOp(bbOp);
    }
    bbOps.clear();

    // recursively traverse all ewops and insert localcoreop and update ewbinops
    // accordingly
    for (auto rpOp : rpOps) {
      ::std::unordered_multimap<::mlir::Operation *, ::imex::dist::LocalCoreOp>
          lcOps;
      // find all subviews on the repartitioned array to add localcoreops
      std::vector<::mlir::Operation *> users(rpOp->getUsers().begin(),
                                             rpOp->getUsers().end());
      std::sort(users.begin(), users.end(),
                opOrderCmp(this->getAnalysis<::mlir::DominanceInfo>()));
      auto base = ::mlir::cast<::imex::dist::RePartitionOp>(rpOp).getArray();
      for (auto user : users) {
        if (auto svOp = ::mlir::dyn_cast<::imex::dist::SubviewOp>(user)) {
          auto arTyp = mlir::cast<::imex::ndarray::NDArrayType>(
              svOp.getResult().getType());

          if (!arTyp.hasUnitSize() && !arTyp.hasZeroSize()) {
            ::imex::dist::LocalCoreOp nlcOp;
            // check if view has a ewop as user
            for (auto svUse : svOp->getUsers()) {
              if (isEW(svUse)) {
                auto loc = svOp->getLoc();
                auto lcOp = lcOps.find(svUse);
                ::imex::ValVec offsets, sizes, strides;
                // constants should go to the beginning of the block
                builder.setInsertionPointToStart(svOp->getBlock());
                offsets = ::imex::getMixedAsValues(
                    loc, builder, svOp.getOffsets(), svOp.getStaticOffsets());
                sizes = ::imex::getMixedAsValues(loc, builder, svOp.getSizes(),
                                                 svOp.getStaticSizes());
                strides = ::imex::getMixedAsValues(
                    loc, builder, svOp.getStrides(), svOp.getStaticStrides());

                auto resOffsets =
                    lcOp == lcOps.end()
                        ? ::mlir::ValueRange{}
                        : ::mlir::ValueRange(lcOp->second.getResultOffsets());
                auto resSizes =
                    lcOp == lcOps.end()
                        ? ::mlir::ValueRange{}
                        : ::mlir::ValueRange(lcOp->second.getResultSizes());
                // we start with localcoreop right after subviewop...
                builder.setInsertionPointAfter(svOp);
                nlcOp = builder.create<::imex::dist::LocalCoreOp>(
                    loc, base, svOp.getTargetOffsets(), svOp.getTargetSizes(),
                    offsets, sizes, strides, resOffsets, resSizes);
                // ...and pull it up as much as possible
                pullOp(nlcOp);
                lcOps.emplace(svUse, nlcOp);
              }
            }
          }
        }
      }

      // for each "initial" ewop add core to dependent ewops and update for
      // alien ops
      for (auto lcOp : lcOps) {
        ::std::set<::mlir::Operation *> visited;
        auto &dom = this->getAnalysis<::mlir::DominanceInfo>();
        ::std::set<::mlir::Operation *, ::imex::opOrderCmp> alien{
            ::imex::opOrderCmp(dom)};

        auto ewHasCore = hasCore(lcOp.first);

        ::imex::ValVec coreOffs = lcOp.second.getResultOffsets();
        ::imex::ValVec coreSzs = lcOp.second.getResultSizes();
        ::imex::ValVec targetOffs = lcOp.second.getTargetOffsets();

        propagateAddLocalCore(builder, lcOp.first, lcOp.second, coreOffs,
                              coreSzs, targetOffs, visited, alien);

        // update local cores if we found "alien" ewops
        if (!alien.empty()) {
          auto loc = rpOp->getLoc();
          auto currCore = getCore(lcOp.first);
          coreOffs = std::get<0>(currCore);
          coreSzs = std::get<1>(currCore);

          ::std::set<::mlir::Operation *> lcOpsDone;

          for (auto a : alien) {
            auto core =
                getCore(a); // should be the same as currCore if ewHasCore
            auto aCoreOffs = std::get<0>(core);
            auto defOp =
                ewHasCore
                    ? lcOp.second
                    : aCoreOffs[0].getDefiningOp<::imex::dist::LocalCoreOp>();

            if (lcOpsDone.find(defOp) == lcOpsDone.end()) {
              lcOpsDone.emplace(defOp);
              auto aCoreSzs = std::get<1>(core);
              auto aTargetOffs = std::get<2>(core);
              auto nlcOp = builder.create<::imex::dist::LocalCoreOp>(
                  loc, defOp.getArray(), defOp.getTargetOffsets(),
                  defOp.getTargetSizes(), defOp.getSliceOffsets(),
                  defOp.getSliceSizes(), defOp.getSliceStrides(), coreOffs,
                  coreSzs);
              ::mlir::SmallVector<::mlir::Operation *> deps;
              for (auto o : nlcOp->getOperands()) {
                if (auto tmp = o.getDefiningOp()) {
                  deps.emplace_back(tmp);
                }
              }
              std::sort(deps.begin(), deps.end(), ::imex::opOrderCmp(dom));
              nlcOp->moveAfter(deps.back());

              coreOffs = nlcOp.getResultOffsets();
              coreSzs = nlcOp.getResultSizes();

              if (ewHasCore) {
                // if there was already a core then we had updated all deps
                // already
                // -> all deps should have the same core -> no need to handle
                // other aliens
                continue;
              }
            }
          }

          // update full chain of ewops with new core
          auto rank = coreOffs.size();
          for (auto vop : visited) {
            auto cStart = ::mlir::isa<::imex::dist::EWBinOp>(vop) ? 2 : 1;
            vop->setOperands(cStart, rank, coreOffs);
            vop->setOperands(cStart + rank, rank, coreSzs);
          }
        }
      }
    }
  }; // runOnOperation()
};   // DistInferEWCoresPass

} // namespace
} // namespace dist

/// Create DistInferEWBinopPass
std::unique_ptr<::mlir::Pass> createDistInferEWCoresPass() {
  return std::make_unique<::imex::dist::DistInferEWCoresPass>();
}

} // namespace imex
