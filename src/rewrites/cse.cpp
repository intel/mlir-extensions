#include "plier/rewrites/cse.hpp"

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/RecyclingAllocator.h>

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Operation.h>

namespace
{
struct SimpleOperationInfo : public llvm::DenseMapInfo<mlir::Operation *> {
    static unsigned getHashValue(const mlir::Operation *opC) {
        return static_cast<unsigned>(mlir::OperationEquivalence::computeHash(const_cast<mlir::Operation *>(opC)));
    }
    static bool isEqual(const mlir::Operation *lhsC, const mlir::Operation *rhsC) {
        auto *lhs = const_cast<mlir::Operation *>(lhsC);
        auto *rhs = const_cast<mlir::Operation *>(rhsC);
        if (lhs == rhs)
          return true;
        if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
            rhs == getTombstoneKey() || rhs == getEmptyKey())
          return false;
        return mlir::OperationEquivalence::isEquivalentTo(const_cast<mlir::Operation *>(lhsC),
                                                          const_cast<mlir::Operation *>(rhsC));
    }
};

using AllocatorTy = llvm::RecyclingAllocator<
  llvm::BumpPtrAllocator,
  llvm::ScopedHashTableVal<mlir::Operation *, mlir::Operation *>>;
using ScopedMapTy = llvm::ScopedHashTable<mlir::Operation *, mlir::Operation *,
                                          SimpleOperationInfo, AllocatorTy>;

mlir::LogicalResult simplifyRegion(ScopedMapTy& map, mlir::Region& region, mlir::PatternRewriter& rewriter)
{
    if (region.empty() || std::next(region.begin()) != region.end())
    {
        return mlir::failure();
    }

    bool success = false;
    for (auto &inst : llvm::make_early_inc_range(region.front()))
    {
        if (inst.isKnownTerminator())
        {
            break;
        }
        if (!mlir::MemoryEffectOpInterface::hasNoEffect(&inst))
        {
            continue;
        }
        if (!inst.getRegions().empty())
        {
            for (auto& reg : inst.getRegions())
            {
                ScopedMapTy::ScopeTy scope(map);
                if (mlir::succeeded(simplifyRegion(map, reg, rewriter)))
                {
                    success = true;
                }
            }
            continue;
        }

        auto* previous_op = map.lookup(&inst);
        if (previous_op != nullptr)
        {
            rewriter.replaceOp(&inst, previous_op->getResults());
            success = true;
        }
        else
        {
            map.insert(&inst, &inst);
        }
    }
    return mlir::success(success);
}
}

mlir::LogicalResult plier::detail::applyCSE(mlir::Region& region, mlir::PatternRewriter& rewriter)
{
    ScopedMapTy map;
    ScopedMapTy::ScopeTy scope(map);
    return simplifyRegion(map, region, rewriter);
}
