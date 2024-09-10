//===- OverlapCommAndCompute.cpp - OverlapCommAndCompute Transform *- C++ -*-//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file implements overlapping communication and computation.
/// The pass tries to pull up ewops which are users of other ewops so that
/// similar, e.g. dependent, ewops are close together.
/// The main purpose of this is to allow separate groups of ewops from each
/// other so that asynchronous operations can effectively operate in the
/// background. For this, the pass pushes down WaitOps to the first use of the
/// data they protect. It is also necessary to push down SubViewOps to their
/// first use.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/DistRuntime/IR/DistRuntimeOps.h>
#include <imex/Dialect/DistRuntime/Transforms/Passes.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Dialect/NDArray/Transforms/Utils.h>
#include <imex/Utils/PassUtils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <algorithm>
#include <vector>

namespace imex {
#define GEN_PASS_DEF_OVERLAPCOMMANDCOMPUTE
#include <imex/Dialect/DistRuntime/Transforms/Passes.h.inc>
} // namespace imex

namespace imex {
namespace ndarray {
std::vector<::mlir::Operation *> getSortedUsers(::mlir::DominanceInfo &dom,
                                                ::mlir::Operation *op) {
  std::vector<::mlir::Operation *> users(op->getUsers().begin(),
                                         op->getUsers().end());
  std::sort(users.begin(), users.end(), ::imex::opOrderCmp(dom));
  return users;
}

namespace {

struct OverlapCommAndComputePass
    : public impl::OverlapCommAndComputeBase<OverlapCommAndComputePass> {

  OverlapCommAndComputePass() = default;

  // pull up dependent ewops if possible
  void pullDescendents(::std::vector<::mlir::Operation *> &grp) {
    auto &dom = this->getAnalysis<::mlir::DominanceInfo>();

    for (auto op : grp) {
      ::std::vector<::mlir::Operation *> allOps;
      ::mlir::Operation *curr = op;

      if (std::find(allOps.begin(), allOps.end(), op) == allOps.end()) {
        allOps.emplace_back(op);
      }

      // pull all users
      for (auto user = op->getUsers().begin(); user != op->getUsers().end();
           ++user) {
        if (::mlir::isa<::imex::ndarray::EWBinOp, ::imex::ndarray::EWUnyOp>(
                *user) &&
            std::find(grp.begin(), grp.end(), *user) != grp.end() &&
            std::find(allOps.begin(), allOps.end(), *user) == allOps.end()) {
          ::mlir::SmallVector<::mlir::Operation *> toBeMoved;
          // store users and dependences for later sorting and move
          if (canMoveAfter(dom, *user, curr, toBeMoved)) {
            for (auto dop : toBeMoved) {
              if (::mlir::isa<::imex::distruntime::GetHaloOp>(dop)) {
                toBeMoved.clear();
                break; // FIXME can we do this less restrictive?
              }
            }
            for (auto dop : toBeMoved) {
              if (std::find(allOps.begin(), allOps.end(), dop) ==
                  allOps.end()) {
                allOps.emplace_back(dop);
              }
            }
            curr = allOps.back();
          }
        }
      }

      // sort all deps and move to right after ewop
      if (!allOps.empty()) {
        std::sort(allOps.begin(), allOps.end(), ::imex::opOrderCmp(dom));
        curr = allOps.front();
        for (auto dop : allOps) {
          if (dop != curr) {
            assert(!::mlir::isa<::imex::distruntime::GetHaloOp>(dop));
            dop->moveAfter(curr);
            curr = dop;
          }
        }
      }
    }
  }

  // push operation down to first use
  void pushDefiningOp(::mlir::Operation *op) {
    auto &dom = this->getAnalysis<::mlir::DominanceInfo>();
    auto users = getSortedUsers(dom, op);
    if (!users.empty()) {
      op->moveBefore(users.front());
    }
  }

  // collect all users of given value, excluding wait ops
  static void appendUsers(::mlir::Value val,
                          ::mlir::SmallVector<::mlir::Operation *> &users) {
    for (auto it = val.user_begin(); it != val.user_end(); ++it) {
      auto op = *it;
      if (!::mlir::isa<::imex::distruntime::WaitOp>(op)) {
        // A cast op can be ignored as it is basically a no-op
        // but the result needs to be tracked
        if (auto castOp = ::mlir::dyn_cast<::imex::ndarray::CastOp>(op)) {
          appendUsers(castOp.getDestination(), users);
        } else {
          users.push_back(op);
        }
      }
    }
  };

  // From the WaitOps defining GetHaloOp get resulting halos.
  // Move WaitOp to first use of any of the halos.
  void pushWaitOp(::mlir::Operation *op) {
    auto waitOp = ::mlir::cast<::imex::distruntime::WaitOp>(op);
    auto asyncOp = waitOp.getHandle().getDefiningOp<::mlir::AsyncOpInterface>();
    assert(asyncOp);

    ::mlir::SmallVector<::mlir::Operation *> users;
    for (auto d : asyncOp.getDependent()) {
      appendUsers(d, users);
    }

    // sort
    auto &dom = this->getAnalysis<::mlir::DominanceInfo>();
    std::sort(users.begin(), users.end(), ::imex::opOrderCmp(dom));

    // push WaitOp down
    if (!users.empty()) {
      op->moveBefore(users.front());
    }
    // if there is no user, we still need the wait call.
  }

  /// @brief group ewops, push out SubviewOps and WaitOps as much as possible.
  /// Do not pull ewops over InsertSliceOps
  void runOnOperation() override {
    auto root = this->getOperation();
    ::std::vector<::mlir::Operation *> ewops, svops, waitops;
    ::std::vector<::std::vector<::mlir::Operation *>> ewgroups;

    // find all ewops, WaitOps and Subviewops
    // create groups of ewops separated by InsertSliceOps
    root->walk([&](::mlir::Operation *op) {
      if (::mlir::isa<::imex::ndarray::EWBinOp, ::imex::ndarray::EWUnyOp>(op)) {
        ewops.emplace_back(op);
      } else if (::mlir::isa<::imex::ndarray::SubviewOp>(op)) {
        svops.emplace_back(op);
      } else if (::mlir::isa<::imex::distruntime::WaitOp>(op)) {
        waitops.emplace_back(op);
      } else if (::mlir::isa<::imex::ndarray::InsertSliceOp>(op) &&
                 !ewops.empty()) {
        ewgroups.push_back(std::move(ewops));
        assert(ewops.empty());
      }
    });
    if (!ewops.empty()) {
      ewgroups.emplace_back(std::move(ewops));
    }

    // within each group, pull up dependent ewops
    for (auto grp : ewgroups) {
      pullDescendents(grp);
      grp.clear();
    }
    ewgroups.clear();

    // push down SubviewOPs to their first use
    for (auto op : svops) {
      pushDefiningOp(op);
    }
    svops.clear();

    // push down WaitOps to the first use of the data they protect
    for (auto op : waitops) {
      pushWaitOp(op);
    }
    waitops.clear();
  }
};
} // namespace
} // namespace ndarray

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::Pass> createOverlapCommAndComputePass() {
  return std::make_unique<::imex::ndarray::OverlapCommAndComputePass>();
}

} // namespace imex
