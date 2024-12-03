//===- RemoveTemporaries.cpp ------------------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the RemoveTemporaries transform.
///
//===----------------------------------------------------------------------===//

#include "imex/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include <imex/Utils/PassUtils.h>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace imex {
#define GEN_PASS_DEF_REMOVETEMPORARIES
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

#define DEBUG_TYPE "imex-remove-temporaries"

#ifndef NDEBUG
#define DEBUG_MSG(PREFIX, MSG)                                                 \
  LLVM_DEBUG(llvm::dbgs() << PREFIX << ": " << MSG << "\n");
#define DEBUG_OP(PREFIX, MSG, OP)                                              \
  LLVM_DEBUG(llvm::dbgs() << PREFIX << ": " << MSG << " '" << OP->getName()    \
                          << "' " << OP->getLoc() << "\n");
#define DEBUG_OP_VEC(PREFIX, MSG, OPVEC)                                       \
  LLVM_DEBUG(llvm::dbgs() << PREFIX << ": " << MSG << " (" << OPVEC.size()     \
                          << ")\n");                                           \
  for (auto op : OPVEC) {                                                      \
    DEBUG_OP(PREFIX, "  ", op)                                                 \
  }
#endif

using namespace imex;

namespace {

/// Returns allocation operation or nullptr
::mlir::Operation *findAllocOp(::mlir::Value value) {
  if (::mlir::Operation *op = value.getDefiningOp()) {
    if (auto effects = ::mlir::dyn_cast<::mlir::MemoryEffectOpInterface>(op)) {
      if (effects.hasEffect<::mlir::MemoryEffects::Allocate>()) {
        return op;
      }
    }
  }
  return nullptr;
}

/// Returns deallocation operation or nullptr
::mlir::Operation *findDeallocOp(::mlir::Value value) {
  auto users = value.getUsers();
  auto it = llvm::find_if(users, [&](::mlir::Operation *op) {
    if (auto effects = ::mlir::dyn_cast<::mlir::MemoryEffectOpInterface>(op)) {
      return effects.hasEffect<::mlir::MemoryEffects::Free>();
    }
    return false;
  });
  return (it == users.end() ? nullptr : *it);
}

/// Finds all subviews of an operation
/// Follows chained subview and cast ops
void findAllSubviews(::mlir::Operation *op,
                     ::mlir::SmallVector<::mlir::Operation *> &foundOps) {
  for (auto user : op->getUsers()) {
    if (::mlir::isa<::mlir::memref::SubViewOp>(user)) {
      foundOps.push_back(user);
    }
    if (::mlir::isa<::mlir::memref::SubViewOp, ::mlir::memref::CastOp>(user)) {
      findAllSubviews(user, foundOps);
    }
  }
}

/// Find if val is being returned
/// Follows chained subview and cast ops
bool findReturn(::mlir::Value val) {
  for (auto user : val.getUsers()) {
    DEBUG_OP("findReturn", "  user", user)
    if (::mlir::isa<::mlir::func::ReturnOp>(user)) {
      return true;
    }
    if (::mlir::isa<::mlir::memref::SubViewOp, ::mlir::memref::CastOp>(user)) {
      if (findReturn(user->getResult(0))) {
        return true;
      }
    }
  }
  return false;
}

/// Returns the root value in a chain of memref.subview/cast ops
/// Appends all found subview ops to `subviewOps` vector.
::mlir::Value
findSubviewRootValue(::mlir::Value val,
                     ::mlir::SmallVector<::mlir::Operation *> &subviewOps) {
  auto op = val.getDefiningOp();
  if (op &&
      ::mlir::isa<::mlir::memref::SubViewOp, ::mlir::memref::CastOp>(op)) {
    DEBUG_OP("findSubviewRootValue", "  found op", op)
    auto src = op->getOperand(0);
    if (::mlir::isa<::mlir::memref::SubViewOp>(op)) {
      subviewOps.push_back(op);
    }
    return findSubviewRootValue(src, subviewOps);
  }
  return val;
}

/// Check whether `op` can have a write effect on value `val`
static bool opHasWriteEffect(::mlir::Value val, ::mlir::Operation *op) {
  // Check whether the operation `op` has write effect on the memory.
  if (!llvm::is_contained(val.getUsers(), op))
    return false;
  if (auto memEffect = ::mlir::dyn_cast<::mlir::MemoryEffectOpInterface>(op)) {
    ::mlir::SmallVector<::mlir::MemoryEffects::EffectInstance, 1> effects;
    memEffect.getEffects(effects);
    return llvm::any_of(
        effects, [](::mlir::MemoryEffects::EffectInstance effect) {
          return ::mlir::isa<::mlir::MemoryEffects::Write>(effect.getEffect());
        });
  }
  // Op does not implement the interface, assume effect is present
  return true;
}

/// Collect all operations between `startOp` and `endOp` that satisfies
/// `opHasProperty`. If `endOp` is `nullptr` travers until the end of region.
/// @return true if property checking succeeded
bool collectMatchingOps(
    ::mlir::Operation *startOp, ::mlir::Operation *endOp,
    ::mlir::SmallVector<::mlir::Operation *> &foundOps,
    std::function<bool(::mlir::Operation *)> opHasProperty) {

  DEBUG_OP("collectMatchingOps", "  start", startOp)
  auto startOpRegion = startOp->getParentRegion();
  auto startOpBlock = startOp->getBlock();
  bool checkAllBlocks = true;
  if (endOp) {
    DEBUG_OP("collectMatchingOps", "   end", endOp)
    auto endOpRegion = endOp->getParentRegion();
    auto endOpBlock = endOp->getBlock();
    // Ops in different regions is not supported
    if (startOpRegion != endOpRegion) {
      return false;
    }
    checkAllBlocks = endOpBlock != startOpBlock;
  }

  ::mlir::SmallVector<::mlir::Block *, 2> blocksToCheck;
  {
    // Check ops in start block after startOp
    for (auto iter = ++startOp->getIterator(), end = startOpBlock->end();
         iter != end && (!endOp || (&*iter != endOp)); ++iter) {
      if (opHasProperty(&*iter)) {
        DEBUG_OP("collectMatchingOps", "  found op", (&*iter))
        foundOps.emplace_back(&*iter);
      }
    }
    // If endOp is not in start block, add successor blocks to check list.
    if (checkAllBlocks) {
      for (::mlir::Block *succ : startOpBlock->getSuccessors()) {
        blocksToCheck.push_back(succ);
      }
    }
  }

  // Keep track of blocks already checked
  ::mlir::SmallPtrSet<::mlir::Block *, 4> checkedBlocks;
  // Traverse the graph until reaching `endOp`.
  while (!blocksToCheck.empty()) {
    ::mlir::Block *blk = blocksToCheck.pop_back_val();
    if (checkedBlocks.insert(blk).second) {
      continue;
    }
    for (::mlir::Operation &op : *blk) {
      if (endOp && (&op == endOp)) {
        break;
      }
      if (opHasProperty(&op)) {
        foundOps.emplace_back(&op);
      }
      if (&op == blk->getTerminator()) {
        for (::mlir::Block *succ : blk->getSuccessors()) {
          blocksToCheck.push_back(succ);
        }
      }
    }
  }
  return true;
}

/// Collect all operations between `startOp` and `endOp` that have a write
/// effect on `val`.
/// @return true if property checking succeeded
bool collectWriteEffectOps(::mlir::Operation *startOp, ::mlir::Operation *endOp,
                           ::mlir::SmallVector<::mlir::Operation *> &foundOps,
                           ::mlir::Value val) {
  auto writesToVal = [&val](::mlir::Operation *op) {
    return opHasWriteEffect(val, op);
  };
  return collectMatchingOps(startOp, endOp, foundOps, writesToVal);
}

// Compares write/read memref types and infers possible RAW conflicts
// Requires that strides and offsets are static.
bool safeToWrite(::mlir::MemRefType write, ::mlir::MemRefType read) {
  auto wStrOff = ::mlir::getStridesAndOffset(write);
  auto rStrOff = ::mlir::getStridesAndOffset(read);
  auto wStrides = wStrOff.first;
  auto wOffset = wStrOff.second;
  auto rStrides = rStrOff.first;
  auto rOffset = rStrOff.second;

  if (write.getRank() != read.getRank()) {
    // rank does not match
    return false;
  }

  for (auto i = 0; i < write.getRank(); i++) {
    if (::mlir::ShapedType::isDynamic(wStrides[i]) ||
        ::mlir::ShapedType::isDynamic(rStrides[i]) ||
        wStrides[i] > rStrides[i]) {
      // stride does not match
      return false;
    }
  }
  if (::mlir::ShapedType::isDynamic(wOffset) ||
      ::mlir::ShapedType::isDynamic(rOffset) || wOffset > rOffset) {
    // write offset is larger; potential read-after-write conflict
    // assuming positive loop iteration order
    return false;
  }
  return true;
}

/// Return corresponding memref type with zero offset and unit strides
mlir::MemRefType getMemRefTypeWithIdentityLayout(mlir::MemRefType srcType) {
  mlir::MemRefLayoutAttrInterface layout = {};
  return mlir::MemRefType::get(srcType.getShape(), srcType.getElementType(),
                               layout, nullptr);
}

/// Apply subview ops in reverse order and infer final memref type
mlir::MemRefType
inferSubviewChainResultType(mlir::MemRefType srcType,
                            mlir::SmallVector<::mlir::Operation *> subviewOps) {
  // force static identity data layout
  auto newType = getMemRefTypeWithIdentityLayout(srcType);
  for (auto it = subviewOps.rbegin(); it != subviewOps.rend(); ++it) {
    auto svOp = mlir::cast<mlir::memref::SubViewOp>(*it);
    auto off = svOp.getStaticOffsets();
    auto size = svOp.getStaticSizes();
    auto stride = svOp.getStaticStrides();
    newType = mlir::cast<::mlir::MemRefType>(
        svOp.inferResultType(newType, off, size, stride));
  }
  return newType;
}

/// Check whether replacing write destination `writeVal` by `newVal` in `op`
/// would result in a RAW conflict
/// @return true if potential conflict is found
bool opHasRAWConflict(mlir::Operation *op, mlir::Value writeVal,
                      mlir::Value newVal, mlir::AliasAnalysis &mAlias) {
  for (auto readVal : op->getOperands()) {
    auto aliasRes = mAlias.alias(readVal, newVal);
    if (aliasRes.isPartial() || aliasRes.isMust()) {
      // After replacement we would read and write to the same memref
      // NOTE we accept MayAlias e.g. in the case of input args
      bool compatibleMemrefs = false;
      ::mlir::SmallVector<::mlir::Operation *> newSVOps, readSVOps;
      auto newRoot = findSubviewRootValue(newVal, newSVOps);
      auto readRoot = findSubviewRootValue(readVal, readSVOps);
      auto readType = mlir::dyn_cast<::mlir::MemRefType>(readVal.getType());
      auto newType = mlir::dyn_cast<::mlir::MemRefType>(newVal.getType());
      if (newRoot == readRoot && readType && newType) {
        // check memref types of write and read operands
        // 1) infer static-layout memref type for readVal by applying subviews
        auto readType2 = inferSubviewChainResultType(readType, readSVOps);
        // append all writeVal subviews to newVal subview chain
        findSubviewRootValue(writeVal, newSVOps);
        // 2) infer what write operand memref type would be after replacement
        auto writeType = inferSubviewChainResultType(newType, newSVOps);
        DEBUG_MSG("checkReadWriteConflict", "   readType: " << readType2)
        DEBUG_MSG("checkReadWriteConflict", "  writeType: " << writeType)
        compatibleMemrefs = safeToWrite(writeType, readType2);
      }
      if (!compatibleMemrefs) {
        DEBUG_OP("checkReadWriteConflict", "  read/write conflict in", op)
        return true;
      } else {
        DEBUG_OP("checkReadWriteConflict", "  subviews are compatible in", op)
      }
    }
  }
  return false;
}

/// Checks whether replacing `scrAllocOp` with `dstDefOp` would result in a
/// read/write conflict in some operation that writes into `scrAllocOp` value.
/// @return true if no conflict is found
bool checkReadWriteConflict(mlir::Operation *op, mlir::Operation *srcAllocOp,
                            mlir::Operation *dstDefOp,
                            mlir::AliasAnalysis &mAlias) {
  // find all ops that write to srcAlloc, traverse through all blocks
  auto srcVal = srcAllocOp->getResult(0);
  auto newVal = dstDefOp->getResult(0);
  ::mlir::SmallVector<::mlir::Operation *> srcAllocWriteOps;
  if (!collectWriteEffectOps(srcAllocOp, nullptr, srcAllocWriteOps, srcVal)) {
    return false;
  }
  DEBUG_OP_VEC("checkReadWriteConflict", "ops that write to src alloc",
               srcAllocWriteOps)
  // check if op has potential write conflict
  for (auto cop : srcAllocWriteOps) {
    if (cop != op && opHasRAWConflict(cop, srcVal, newVal, mAlias))
      return false;
  }

  // repeat to all ops that write to a view of srcAlloc
  ::mlir::SmallVector<::mlir::Operation *> srcSubviewOps;
  findAllSubviews(srcAllocOp, srcSubviewOps);
  DEBUG_OP_VEC("checkReadWriteConflict", "src subview ops", srcSubviewOps)
  for (auto sview : srcSubviewOps) {
    ::mlir::SmallVector<::mlir::Operation *> writeOps;
    auto svVal = sview->getResult(0);
    if (!collectWriteEffectOps(srcAllocOp, nullptr, writeOps, svVal)) {
      return false;
    }
    DEBUG_OP("checkReadWriteConflict", "inspecting src alloc view", sview)
    DEBUG_OP_VEC("checkReadWriteConflict", "ops that write to a src alloc view",
                 writeOps)
    for (auto cop : writeOps) {
      if (cop != op && opHasRAWConflict(cop, svVal, newVal, mAlias))
        return false;
    }
  }
  return true;
}

/// Replace the uses of `oldOp` with the given `val` and for subview uses
/// propagate the type change. Changing the memref type may require
/// propagating it through subview ops so we cannot just do a replaceAllUse
/// but need to propagate the type change and erase old subview ops. Ported
/// from mlir memref MultiBuffer.cpp
static void replaceUsesAndPropagateType(
    mlir::RewriterBase &rewriter, mlir::Operation *oldOp, mlir::Value val,
    ::mlir::SmallVector<mlir::Operation *> &opsToDelete) {
  mlir::SmallVector<mlir::OpOperand *> operandsToReplace;

  // Save the operand to replace / delete later (avoid iterator invalidation).
  // TODO: can we use an early_inc iterator?
  for (mlir::OpOperand &use : oldOp->getUses()) {
    // Check for dealloc ops
    if (auto effects =
            ::mlir::dyn_cast<::mlir::MemoryEffectOpInterface>(use.getOwner())) {
      if (effects.hasEffect<::mlir::MemoryEffects::Free>()) {
        continue;
      }
    }
    // Non-subview ops will be replaced by `val`.
    auto subviewUse = mlir::dyn_cast<mlir::memref::SubViewOp>(use.getOwner());
    if (!subviewUse) {
      operandsToReplace.push_back(&use);
      continue;
    }

    // `subview(old_op)` is replaced by a new `subview(val)`.
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(subviewUse);
    mlir::Type newType = mlir::memref::SubViewOp::inferRankReducedResultType(
        subviewUse.getType().getShape(),
        mlir::cast<mlir::MemRefType>(val.getType()),
        subviewUse.getStaticOffsets(), subviewUse.getStaticSizes(),
        subviewUse.getStaticStrides());
    mlir::Value newSubview = rewriter.create<mlir::memref::SubViewOp>(
        subviewUse->getLoc(), mlir::cast<mlir::MemRefType>(newType), val,
        subviewUse.getMixedOffsets(), subviewUse.getMixedSizes(),
        subviewUse.getMixedStrides());

    // Ouch recursion ... is this really necessary?
    replaceUsesAndPropagateType(rewriter, subviewUse, newSubview, opsToDelete);

    opsToDelete.push_back(use.getOwner());
  }

  // Perform late replacement.
  for (mlir::OpOperand *operand : operandsToReplace) {
    mlir::Operation *op = operand->getOwner();
    rewriter.startOpModification(op);
    operand->set(val);
    rewriter.finalizeOpModification(op);
  }
}

// Moves op after markerOp if possible. Returns true if successful.
// If move is not possible, applies no changes and returns false.
bool moveAfterIfPossible(::mlir::Operation *op, ::mlir::Operation *markerOp,
                         ::mlir::Operation *anchorOp,
                         ::mlir::DominanceInfo &dom) {
  ::mlir::SmallVector<::mlir::Operation *> toBeMoved;
  ::mlir::SmallVector<::mlir::Operation *> writeEffOps;
  if (!dom.dominates(op, markerOp)) {
    if (canMoveAfter(dom, op, markerOp, toBeMoved)) {
      DEBUG_OP_VEC("RemoveTemporaries", "ops to be moved", toBeMoved)
      // check for write effects: get toBeMoved op's return value,
      // check if any op has a write effect on it between op and anchorOp
      for (auto cop : toBeMoved) {
        for (auto res : cop->getResults()) {
          if (!collectWriteEffectOps(cop, anchorOp, writeEffOps, res)) {
            return false;
          }
        }
      }
      // check if write effect ops can move
      for (auto cop : writeEffOps) {
        if (!canMoveAfter(dom, cop, markerOp, toBeMoved)) {
          return false;
        }
      }
      // sort ops, also remove possible duplicates
      auto opIsIncluded = [&toBeMoved](::mlir::Operation *op) {
        for (auto cop : toBeMoved) {
          if (cop == op) {
            return true;
          }
        }
        return false;
      };
      ::mlir::SmallVector<::mlir::Operation *> sortedOps;
      if (!collectMatchingOps(markerOp, anchorOp, sortedOps, opIsIncluded)) {
        return false;
      }
      DEBUG_OP_VEC("RemoveTemporaries", "ops to be moved after sort", sortedOps)

      // move all ops
      for (auto it = sortedOps.rbegin(); it != sortedOps.rend(); ++it) {
        auto cop = *it;
        DEBUG_OP("RemoveTemporaries", "  moving", cop)
        DEBUG_OP("RemoveTemporaries", "    after", markerOp)
        cop->moveAfter(markerOp);
      }
      return true;
    }
  }
  return false;
}

struct RemoveTemporaries
    : public imex::impl::RemoveTemporariesBase<RemoveTemporaries> {
  void runOnOperation() override {
    ::mlir::SmallVector<mlir::Operation *> opsToRemove;
    getOperation()->walk([&](::mlir::CopyOpInterface copyOp) {
      transform(copyOp, opsToRemove);
    });
    for (::mlir::Operation *op : opsToRemove) {
      if (!op->use_empty()) {
        DEBUG_OP("RemoveTemporaries", "cannot remove op", op)
        for (auto user : op->getResult(0).getUsers()) {
          DEBUG_OP("RemoveTemporaries", "    used by", user)
        }
        assert(false && "Cannot remove op: it still has users");
      }
      op->erase();
    }
    return;
  }

private:
  void transform(::mlir::CopyOpInterface opi,
                 ::mlir::SmallVector<mlir::Operation *> &opsToRemove) {

    auto op = opi.getOperation();
    auto dst = opi.getTarget();
    auto src = opi.getSource();
    mlir::IRRewriter rewriter(op->getContext());
    DEBUG_MSG("RemoveTemporaries", "------------------------------------------")
    DEBUG_OP("RemoveTemporaries", "inspecting", op)

    auto srcAllocOp = findAllocOp(src);
    auto srcDeallocOp = findDeallocOp(src);
    auto dstDeallocOp = findDeallocOp(dst);
    auto dstDefOp = dst.getDefiningOp();
    if (!srcAllocOp) {
      // src is not associated with a temp array allocation
      DEBUG_MSG("RemoveTemporaries",
                "src is not associated with an alloc, skipping")
      return;
    }
    auto allocOpParentReg = srcAllocOp->getParentRegion();
    auto copyOpParentReg = op->getParentRegion();
    DEBUG_OP("RemoveTemporaries", "  src alloc op", srcAllocOp)

    bool srcIsReturned = findReturn(srcAllocOp->getResult(0));

    if (copyOpParentReg != allocOpParentReg) {
      DEBUG_MSG("RemoveTemporaries",
                "alloc and copy are in different regions, skipping")
      return;
    }
    if (dstDefOp) {
      // There is a dst defining op
      DEBUG_OP("RemoveTemporaries", "  defining op", dstDefOp)
      auto tmpOps = ::mlir::SmallVector<::mlir::Operation *>();
      auto dstRootValue = findSubviewRootValue(dst, tmpOps);
      bool dstIsReturned = findReturn(dstRootValue);
      if (srcIsReturned && dstIsReturned) {
        // removing src alloc would potentially result in returning dst and
        // its subview which changes function semantics
        // TODO need to check for a subview as well?
        DEBUG_MSG("RemoveTemporaries",
                  "  both src and dst are returned, aborting.")
        return;
      }
      auto &memrefAlias = getAnalysis<mlir::AliasAnalysis>();
      memrefAlias.alias(src, dst);
      if (!checkReadWriteConflict(op, srcAllocOp, dstDefOp, memrefAlias)) {
        DEBUG_MSG("RemoveTemporaries",
                  "found read after write conflict, skipping")
        return;
      }
      // Move copy target right after src allocation
      // unless target is defined earlier
      auto &dom = getAnalysis<::mlir::DominanceInfo>();
      if (!dom.dominates(dstDefOp, srcAllocOp) &&
          !moveAfterIfPossible(dstDefOp, srcAllocOp, op, dom)) {
        DEBUG_MSG("RemoveTemporaries", "cannot move dst defining op, skipping")
        return;
      }
      // Replace src alloc uses by dst defining op
      DEBUG_OP("RemoveTemporaries", "  replacing src alloc", srcAllocOp)
      DEBUG_OP("RemoveTemporaries", "   with", dstDefOp)
      replaceUsesAndPropagateType(rewriter, srcAllocOp, dst, opsToRemove);
    } else {
      if (srcIsReturned) {
        // no defining op, dst is function argument, after removing scr allow
        // we potentially end up returning a view to dst
        DEBUG_MSG("RemoveTemporaries",
                  "  no dst defining op and src is returned, aborting.")
        return;
      }
      // no defining op, replace src with dst mlir::Value
      DEBUG_OP("RemoveTemporaries", "  replacing src alloc", srcAllocOp)
      DEBUG_MSG("RemoveTemporaries", "    with copy op dst value")
      replaceUsesAndPropagateType(rewriter, srcAllocOp, dst, opsToRemove);
    }
    DEBUG_OP("RemoveTemporaries", "  removing op", op)
    opsToRemove.push_back(op);
    if (srcDeallocOp) {
      DEBUG_OP("RemoveTemporaries", "  removing src dealloc op", srcDeallocOp)
      opsToRemove.push_back(srcDeallocOp);
    } else if (dstDeallocOp) {
      // src alloc is not deallocated, after replacement it would be
      // remove dealloc in case this memref should be kept alive
      DEBUG_OP("RemoveTemporaries", "  removing dst dealloc op", dstDeallocOp)
      opsToRemove.push_back(dstDeallocOp);
    }
    DEBUG_OP("RemoveTemporaries", "  removing src alloc op", srcAllocOp)
    opsToRemove.push_back(srcAllocOp);
  }
};

} // end anonymous namespace

namespace imex {
std::unique_ptr<mlir::Pass> createRemoveTemporariesPass() {
  return std::make_unique<RemoveTemporaries>();
}
} // namespace imex
