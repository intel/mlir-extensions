//===- BF16ToGPU.cpp - bf16 to Intel GPU pass -------------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This pass iterates gpu.func and gpu.launch_func starting from top module
/// and
///     replace bf16 dtype with bitwidth equal i16 type
///     rewrite bf16 compute as bf16 extended, f32 compute, f32 truncated
///
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include <mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

using namespace mlir;
using namespace imex;

namespace {
struct BF16ToGPUPass : public BF16ToGPUBase<BF16ToGPUPass> {

public:
  void runOnOperation() override {
    auto mod = getOperation();
    SymbolTable symbolTable(mod);
    mlir::OpBuilder builder(mod);
    auto &aliases = getAnalysis<mlir::BufferViewFlowAnalysis>();
    // Part 1: gpu::GPUFuncOp
    WalkResult result1 =
        mod.walk<WalkOrder::PreOrder>([&](gpu::GPUFuncOp op) -> WalkResult {
          // 1-1: Create new FunctionType and replace old FunctionType
          auto oftype = op.getFunctionType();
          llvm::SmallVector<mlir::Type, 4> argTypes;
          ArrayRef<Type> inputTypes;
          ArrayRef<Type> resultTypes;
          for (Type t : oftype.getInputs()) {
            MemRefType m = t.dyn_cast<MemRefType>();
            if (m) {
              Type et = m.getElementType();
              if (et.isBF16()) {
                if (m.hasStaticShape()) {
                  llvm::ArrayRef<int64_t> s = m.getShape();
                  auto i = MemRefType::get(s, builder.getI16Type());
                  argTypes.push_back(i);
                } else {
                  // TODO: Support dynamic shape
                  op.emitError(
                      "Non static shape bf16 MemRefType in GPUFuncOp inputs");
                }
              } else {
                argTypes.push_back(t);
              }
            } else if (t.isBF16()) {
              argTypes.push_back(builder.getI16Type());
            } else {
              argTypes.push_back(t);
            }
          }
          auto nftype =
              dyn_cast<FunctionType>(op.cloneTypeWith(argTypes, resultTypes));
          op.setFunctionType(nftype);

          // 1-2: Collect ops that need bf16 widening and widen those ops
          // Most ops in arith and math dialect that has bf16 operand will
          // be widened to use f32 operand
          SmallVector<Operation *, 8> widenOps;
          WalkResult result1_1 = op.getRegion().walk<WalkOrder::PreOrder>(
              [&](Operation *lop) -> WalkResult {
                auto oname = lop->getName().getStringRef();
                if (oname.startswith("arith.") || oname.startswith("math.")) {
                  // Skip bitcast operation as we cannot change width of operand
                  if (!oname.startswith("arith.bitcast")) {
                    bool needWidening = false;
                    for (const auto &oper : lop->getOperands()) {
                      if (oper.getType().isBF16()) {
                        needWidening = true;
                      }
                    }
                    if (needWidening) {
                      widenOps.push_back(lop);
                    }
                  }
                }
                return WalkResult::advance();
              });
          for (Operation *o : widenOps) {
            builder.setInsertionPoint(o);
            unsigned int idx = 0;
            for (const auto &oper : o->getOperands()) {
              if (oper.getType().isBF16()) {
                auto newOp = builder.create<arith::ExtFOp>(
                    o->getLoc(), builder.getF32Type(), oper);
                o->setOperand(idx, newOp);
              }
              idx++;
            }
            for (mlir::OpResult res : o->getResults()) {
              if (res.getType().isBF16()) {
                res.setType(builder.getF32Type());
                builder.setInsertionPointAfter(o);
                auto newRes = builder.create<arith::TruncFOp>(
                    o->getLoc(), builder.getBF16Type(), res);
                res.replaceAllUsesExcept(newRes, newRes);
              }
            }
          }
          //  1-3: Change element type of entry block arguments
          Block &eblock = op.getBlocks().front();
          for (mlir::BlockArgument arg : eblock.getArguments()) {
            Type argt = arg.getType();
            MemRefType mt = dyn_cast<MemRefType>(argt);
            if (mt) {
              if (mt.getElementType().isBF16()) {
                MemRefType newMt = dyn_cast<MemRefType>(
                    mt.cloneWith(mt.getShape(), builder.getI16Type()));
                arg.setType(newMt);
              }
            } else if (argt.isBF16()) {
              arg.setType(builder.getI16Type());
            }
          }
          WalkResult result1_2 = op.getRegion().walk<WalkOrder::PreOrder>(
              [&](Operation *lop) -> WalkResult {
                if (dyn_cast<arith::ExtFOp>(lop)) {
                  // if extf i16 -> f32 : "i16" is not a typo
                  if (lop->getOperand(0).getType().isInteger(16)) {
                    if (lop->getResult(0).getType().isF32()) {
                      builder.setInsertionPoint(lop);
                      auto bcast = builder.create<arith::BitcastOp>(
                          lop->getLoc(), builder.getBF16Type(),
                          lop->getOperand(0));
                      lop->setOperand(0, bcast);
                    }
                  }
                } else if (dyn_cast<arith::TruncFOp>(lop)) {
                  // if truncf f32 -> bf16
                  if (lop->getOperand(0).getType().isF32()) {
                    if (lop->getResult(0).getType().isBF16()) {
                      builder.setInsertionPointAfter(lop);
                      auto bcast = builder.create<arith::BitcastOp>(
                          lop->getLoc(), builder.getI16Type(),
                          lop->getResult(0));
                      lop->getResult(0).replaceAllUsesExcept(bcast, bcast);
                    }
                  }
                } else {
                  if (lop->getNumResults() > 0) {
                    if (lop->getResultTypes().front().isBF16()) {
                      lop->getResult(0).setType(builder.getI16Type());
                    }
                  }
                }
                return WalkResult::advance();
              });
          return WalkResult::advance();
        });
    // Part 2: gpu::LaunchFuncOp and gpu::AllocOp
    SmallVector<Operation *, 8> replacedAllocOps;
    WalkResult result2 = mod.walk<WalkOrder::PreOrder>([&](gpu::LaunchFuncOp op)
                                                           -> WalkResult {
      for (const auto &kop : op.getKernelOperands()) {
        auto mem = kop;
        Type memt = mem.getType();
        MemRefType mft = dyn_cast<MemRefType>(memt);
        if (mft) {
          if (!mft.getElementType().isBF16()) {
            continue;
          }
        } else {
          if (!memt.isBF16()) {
            continue;
          }
        }
        SmallVector<ViewLikeOpInterface, 4> parentViews;
        while (auto parentView = mem.getDefiningOp<ViewLikeOpInterface>()) {
          parentViews.push_back(parentView);
          mem = parentView.getViewSource();
        }
        auto dop = mem.getDefiningOp();
        // op is the defining Operation*
        if (isa<gpu::AllocOp>(dop)) {
          auto alloc = dyn_cast<gpu::AllocOp>(dop);
          // TODO: Support dynamic size
          if (alloc.getDynamicSizes().size() != 0) {
            op->emitError("gpu::AllocOp with Dynamic size is not supported!");
          }
          auto t = alloc.getType();
          // TODO: does the check above make this check redundant?
          if (!t.hasStaticShape()) {
            op->emitError("gpu::AllocOp with Dynamic shape is not supported!");
          }
          auto et = t.getElementType();
          builder.setInsertionPoint(dop);
          auto zero = builder.create<arith::ConstantIndexOp>(alloc.getLoc(), 0);
          // get shape of kernel operand and construct a same shape
          // type with i16
          MemRefType m = kop.getType().dyn_cast<MemRefType>();
          llvm::ArrayRef<int64_t> s = m.getShape();
          auto itype = MemRefType::get(s, builder.getI16Type());
          ValueRange sizes;
          memref::ViewOp i16View = nullptr;
          // Different cases of root alloc
          // 1) bf16: create flat i8 alloc
          // 2) i8 and flat: create flat i8 alloc
          // 3) other cases: cannot happen as root alloc needs to already in
          //    bf16 or can be casted to bf16 (flat i8)
          Operation *oldChainOp = nullptr;
          Operation *newChainOp = nullptr;
          if (et.isBF16()) {
            // Collect uses to replace.
            llvm::SmallVector<OpOperand *, 1> deallocUse;
            llvm::SmallVector<OpOperand *, 1> funcArgUse;
            llvm::SmallVector<OpOperand *, 8> otherUse;
            for (OpOperand &u : alloc.getResult(0).getUses()) {
              auto owner = u.getOwner();
              if (isa<gpu::DeallocOp>(owner)) {
                deallocUse.push_back(&u);
              } else if (isa<gpu::LaunchFuncOp>(owner)) {
                funcArgUse.push_back(&u);
              } else {
                otherUse.push_back(&u);
              }
            }
            int64_t bsize = 1;
            for (int64_t d : s) {
              bsize *= d;
            }
            bsize *= 2; // bf16 is twice the bit length of i8
            llvm::SmallVector<int64_t, 1> fshape;
            fshape.push_back(bsize);
            auto ftype = MemRefType::get(fshape, builder.getI8Type());
            // Collect allocs to be removed later
            replacedAllocOps.push_back(dop);
            // Create flat gpu alloc
            auto flatAlloc = builder.create<gpu::AllocOp>(
                alloc.getLoc(), ftype,
                alloc.getAsyncToken() ? alloc.getAsyncToken().getType()
                                      : nullptr,
                alloc.getAsyncDependencies(), alloc.getDynamicSizes(),
                alloc.getSymbolOperands(), alloc.getHostShared());
            // Create two views
            // 1) bf16
            // 2) i16
            auto bf16View = builder.create<memref::ViewOp>(
                alloc.getLoc(), alloc.getType(), flatAlloc.getResult(0), zero,
                sizes);
            i16View = builder.create<memref::ViewOp>(
                alloc.getLoc(), itype, flatAlloc.getResult(0), zero, sizes);
            // Replace old uses of "alloc" with proper Value
            for (OpOperand *u : deallocUse) {
              u->set(flatAlloc.getResult(0));
            }
            for (OpOperand *u : funcArgUse) {
              u->set(i16View.getResult());
            }
            for (OpOperand *u : otherUse) {
              u->set(bf16View.getResult());
            }
            oldChainOp = alloc;
            newChainOp = i16View;
          } else {
            // root alloc is already flat i8, insert after
            builder.setInsertionPointAfter(dop);
            oldChainOp = dop;
            newChainOp = dop;
          }
          // replicate view chain (if any) leading to root alloc with i16
          // instead of bf16
          for (auto it = parentViews.rbegin(); it != parentViews.rend(); it++) {
            Operation *bf16ViewOp = (*it).getOperation();
            Type t = bf16ViewOp->getResultTypes().front();
            MemRefType mt = dyn_cast<MemRefType>(t);
            if (!mt.hasStaticShape()) {
              op->emitError(
                  "Parent views with dynamic shape is not supported.");
            }
            auto i16Mt = mt.cloneWith(mt.getShape(), builder.getI16Type());
            auto operands = bf16ViewOp->getOperands();
            SmallVector<Value, 4> newOperands;
            for (Value operand : operands) {
              if (operand.getDefiningOp() == oldChainOp) {
                newOperands.push_back(newChainOp->getResult(0));
              } else {
                newOperands.push_back(operand);
              }
            }
            newChainOp = builder.create(
                bf16ViewOp->getLoc(), bf16ViewOp->getName().getIdentifier(),
                newOperands, {i16Mt}, bf16ViewOp->getAttrs());
            oldChainOp = bf16ViewOp;
          }
          // replace launch func args using old view chain with new view chain
          for (OpOperand &u : oldChainOp->getResult(0).getUses()) {
            auto owner = u.getOwner();
            if (isa<gpu::LaunchFuncOp>(owner)) {
              u.set(newChainOp->getResult(0));
            }
          }
        } else if (isa<arith::ConstantOp>(dop)) {
          if (!isa<arith::ConstantOp>(mem.getDefiningOp())) {
            op->emitError(
                "Only arith::ConstantOp is supported for bf16 scalar arg.");
          }
          if (dop->getResultTypes().front().isBF16()) {
            llvm::SmallVector<OpOperand *, 4> bf16Use;
            auto uses = dop->getResult(0).getUses();
            for (OpOperand &u : uses) {
              if (isa<gpu::LaunchFuncOp>(u.getOwner())) {
                bf16Use.push_back(&u);
              }
            }
            builder.setInsertionPointAfter(dop);
            auto bcast = builder.create<arith::BitcastOp>(
                dop->getLoc(), builder.getI16Type(), dop->getResult(0));
            for (OpOperand *u : bf16Use) {
              u->set(bcast.getResult());
            }
          }
        } else {
          // This can be arith.constant for scalar parameters.
          op->emitError("Expected gpu.alloc memref producer for bf16 memref "
                        "arg or arith.ConstantOp for bf16 scalar arg.");
        }
        //}
      }
      return WalkResult::advance();
    });
    for (auto alloc : replacedAllocOps) {
      alloc->erase();
    }
  }
};
} // namespace

namespace imex {
std::unique_ptr<mlir::Pass> createBF16ToGPUPass() {
  return std::make_unique<BF16ToGPUPass>();
}
} // namespace imex
