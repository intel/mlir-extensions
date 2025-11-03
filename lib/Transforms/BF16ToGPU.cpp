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

#include "imex/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"
#include <mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

namespace imex {
#define GEN_PASS_DEF_BF16TOGPU
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

using namespace mlir;
using namespace imex;

namespace {
struct BF16ToGPUPass : public imex::impl::BF16ToGPUBase<BF16ToGPUPass> {

public:
  void runOnOperation() override {
    auto mod = getOperation();
    SymbolTable symbolTable(mod);
    mlir::OpBuilder builder(mod);
    // Part 1: gpu::GPUFuncOp
    (void)mod.walk<WalkOrder::PreOrder>([&](gpu::GPUFuncOp op) -> WalkResult {
      // 1-1: Create new FunctionType and replace old FunctionType
      auto oftype = op.getFunctionType();
      llvm::SmallVector<mlir::Type, 4> argTypes;
      ArrayRef<Type> resultTypes;
      for (Type t : oftype.getInputs()) {
        MemRefType m = mlir::dyn_cast<MemRefType>(t);
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
      (void)op.getRegion().walk<WalkOrder::PreOrder>(
          [&](Operation *lop) -> WalkResult {
            auto oname = lop->getName().getStringRef();
            if (oname.starts_with("arith.") || oname.starts_with("math.") ||
                oname.starts_with("scf.for") ||
                oname.starts_with("scf.yield")) {
              // Skip bitcast operation as we cannot change width of operand
              if (!(oname.starts_with("arith.bitcast") ||
                    oname.starts_with("arith.extf"))) {
                bool needWidening = false;
                for (const auto &oper : lop->getOperands()) {
                  if (auto vecTy = mlir::dyn_cast<VectorType>(oper.getType())) {
                    if (vecTy.getElementType().isBF16()) {
                      needWidening = true;
                    }
                  } else if (oper.getType().isBF16()) {
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
          if (auto vecTy = mlir::dyn_cast<VectorType>(oper.getType())) {
            if (vecTy.getElementType().isBF16()) {
              auto newTy =
                  VectorType::get(vecTy.getShape(), builder.getF32Type());
              auto newOp =
                  arith::ExtFOp::create(builder, o->getLoc(), newTy, oper);
              o->setOperand(idx, newOp);
            }
          } else if (oper.getType().isBF16()) {
            auto newOp = arith::ExtFOp::create(builder,
                o->getLoc(), builder.getF32Type(), oper);
            o->setOperand(idx, newOp);
          }
          idx++;
        }

        // handle conversion of bf16 loop iter args
        if (auto forOp = dyn_cast<mlir::scf::ForOp>(o)) {
          for (auto arg : forOp.getRegionIterArgs()) {
            Type argt = arg.getType();

            // Change bf16 iter arg types to f32 type
            if (argt.isBF16()) {
              arg.setType(builder.getF32Type());
            } else if (llvm::isa<Float8E5M2Type>(argt) ||
                       llvm::isa<Float8E4M3FNType>(argt)) {
              // TODO: Handle loop fp8 type iter args
              llvm_unreachable(
                  "Unhandled case when loop iter arg is of f8 type");
            }
          }
        }

        for (mlir::OpResult res : o->getResults()) {
          if (auto vecTy = mlir::dyn_cast<VectorType>(res.getType())) {
            if (vecTy.getElementType().isBF16()) {
              auto resTy =
                  VectorType::get(vecTy.getShape(), builder.getF32Type());
              res.setType(resTy);
              builder.setInsertionPointAfter(o);
              auto newTy =
                  VectorType::get(vecTy.getShape(), builder.getBF16Type());
              auto newRes =
                  arith::TruncFOp::create(builder, o->getLoc(), newTy, res);
              res.replaceAllUsesExcept(newRes, newRes);
            }
          } else if (res.getType().isBF16()) {
            res.setType(builder.getF32Type());
            builder.setInsertionPointAfter(o);
            auto newRes = arith::TruncFOp::create(builder,
                o->getLoc(), builder.getBF16Type(), res);
            res.replaceAllUsesExcept(newRes, newRes);
          }
        }
      }
      //  1-3: Change element type of entry block arguments
      //  This step replaces all external sources of bf16 with i16
      //  and the effect is propagated to the entire gpu.func
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
      // arith.constant is another source of bf16 values.
      // replace them with same bit i16 values.
      SmallVector<Operation *, 8> replacedConstantOps;
      (void)op.getRegion().walk<WalkOrder::PreOrder>(
          [&](Operation *lop) -> WalkResult {
            if (auto constOp = dyn_cast<arith::ConstantOp>(lop)) {
              if (auto vecTy = mlir::dyn_cast<VectorType>(constOp.getType())) {
                if (vecTy.getElementType().isBF16()) {
                  if (auto fval =
                          dyn_cast<DenseElementsAttr>(constOp.getValue())) {
                    auto ival = fval.bitcast(builder.getI16Type());
                    builder.setInsertionPoint(lop);
                    auto newTy =
                        VectorType::get(vecTy.getShape(), builder.getI16Type());
                    auto newConst = arith::ConstantOp::create(builder,
                        lop->getLoc(), newTy, ival);
                    lop->replaceAllUsesWith(newConst);
                    replacedConstantOps.push_back(lop);
                  } else {
                    lop->emitError(
                        "Expected DenseElementsAttr for vector bf16 constant.");
                  }
                }
              } else if (constOp.getType().isBF16()) {
                if (auto fval = dyn_cast<FloatAttr>(constOp.getValue())) {
                  auto bf16val = fval.getValue();
                  auto ival = bf16val.bitcastToAPInt();
                  int64_t i64val = ival.getSExtValue();
                  int16_t i16val = static_cast<int16_t>(i64val);
                  builder.setInsertionPoint(lop);
                  auto newConst = arith::ConstantOp::create(builder,
                      lop->getLoc(), builder.getI16Type(),
                      builder.getI16IntegerAttr(i16val));
                  lop->replaceAllUsesWith(newConst);
                  replacedConstantOps.push_back(lop);
                }
              }
            }
            return WalkResult::advance();
          });
      for (auto cop : replacedConstantOps) {
        cop->erase();
      }
      // Now that all primary bf16 are replaced with i16,
      // some ops are invalid and need to be updated.
      // 1) extf and truncf now needs an additional bitcast operation
      // 2) function calls need callee function signature update.
      // 3) propagate i16 type by changing bf16 result types of ops
      //    to i16. skip arith.constant as it is a value source.
      (void)op.getRegion().walk<WalkOrder::PreOrder>([&](Operation *lop)
                                                         -> WalkResult {
        if (dyn_cast<arith::ExtFOp>(lop)) {
          auto src = lop->getOperand(0);
          auto res = lop->getResult(0);
          // if extf i16 -> f32 : "i16" is not a typo
          auto srcTy = dyn_cast<VectorType>(src.getType());
          auto resTy = dyn_cast<VectorType>(res.getType());
          if (srcTy && resTy) {
            if (srcTy.getElementType().isInteger(16) &&
                resTy.getElementType().isF32()) {
              builder.setInsertionPoint(lop);
              auto newTy =
                  VectorType::get(srcTy.getShape(), builder.getBF16Type());
              auto bcast =
                  arith::BitcastOp::create(builder, lop->getLoc(), newTy, src);
              lop->setOperand(0, bcast);
            }
          } else if (src.getType().isInteger(16) && res.getType().isF32()) {
            builder.setInsertionPoint(lop);
            auto bcast = arith::BitcastOp::create(builder,
                lop->getLoc(), builder.getBF16Type(), src);
            lop->setOperand(0, bcast);
          }
        } else if (dyn_cast<arith::TruncFOp>(lop)) {
          auto src = lop->getOperand(0);
          auto res = lop->getResult(0);
          // if truncf f32 -> bf16
          auto srcTy = dyn_cast<VectorType>(src.getType());
          auto resTy = dyn_cast<VectorType>(res.getType());
          if (srcTy && resTy) {
            if (srcTy.getElementType().isF32() &&
                resTy.getElementType().isBF16()) {
              builder.setInsertionPointAfter(lop);
              auto newTy =
                  VectorType::get(resTy.getShape(), builder.getI16Type());
              auto bcast =
                  arith::BitcastOp::create(builder, lop->getLoc(), newTy, res);
              res.replaceAllUsesExcept(bcast, bcast);
            }
          } else if (src.getType().isF32() && res.getType().isBF16()) {
            builder.setInsertionPointAfter(lop);
            auto bcast = arith::BitcastOp::create(builder,
                lop->getLoc(), builder.getI16Type(), res);
            res.replaceAllUsesExcept(bcast, bcast);
          }
        } else {
          if (auto callOp = dyn_cast<func::CallOp>(lop)) {
            auto name = callOp.getCallee();
            auto module = lop->getParentOfType<gpu::GPUModuleOp>();
            if (!module)
              op.emitError("Parent gpu module not found!");
            auto result = SymbolRefAttr::get(module.getContext(), name);
            auto func = module.lookupSymbol<func::FuncOp>(result.getAttr());
            if (!func)
              op.emitError("Callee not found!");
            auto ftype = func.getFunctionType();
            bool needFuncUpdate = false;

            auto convertBF16ToI16 = [&](TypeRange types) {
              SmallVector<Type, 8> newTypes;
              for (Type t : types) {
                if (auto vecTy = dyn_cast<VectorType>(t)) {
                  if (vecTy.getElementType().isBF16()) {
                    auto newTy =
                        vecTy.cloneWith(vecTy.getShape(), builder.getI16Type());
                    newTypes.push_back(newTy);
                    needFuncUpdate = true;
                  } else {
                    newTypes.push_back(t);
                  }
                } else if (t.isBF16()) {
                  newTypes.push_back(builder.getI16Type());
                  needFuncUpdate = true;
                } else {
                  // TODO: Can callee arg type be bf16 memref?
                  newTypes.push_back(t);
                }
              }
              return newTypes;
            };

            auto newArgTypes = convertBF16ToI16(ftype.getInputs());
            auto newRetTypes = convertBF16ToI16(ftype.getResults());

            if (needFuncUpdate) {
              auto nftype = dyn_cast<FunctionType>(
                  func.cloneTypeWith(newArgTypes, newRetTypes));
              func.setFunctionType(nftype);
            }
          }
          if (lop->getNumResults() > 0 && !dyn_cast<arith::ConstantOp>(lop)) {
            // Foreach result
            //   if elemType is bf16, change it to i16
            int i = 0;
            for (Type t : lop->getResultTypes()) {
              if (mlir::isa<mlir::VectorType>(t)) {
                VectorType vt = mlir::cast<mlir::VectorType>(t);
                if (vt.getElementType().isBF16()) {
                  vt.get(vt.getShape(), builder.getI16Type());
                  lop->getResult(i).setType(
                      vt.get(vt.getShape(), builder.getI16Type()));
                }
              } else if (t.isBF16()) {
                lop->getResult(i).setType(builder.getI16Type());
              }
              i++;
            }
          }
        }
        return WalkResult::advance();
      });
      return WalkResult::advance();
    });
    // Part 2: gpu::LaunchFuncOp and gpu::AllocOp
    SmallVector<Operation *, 8> replacedAllocOps;
    (void)mod.walk<WalkOrder::PreOrder>([&](gpu::LaunchFuncOp op)
                                            -> WalkResult {
      bool hasPassthroughArgs = false;
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
        if (!dop) {
          hasPassthroughArgs = true;
        }
        // op is the defining Operation*
        else if (isa<gpu::AllocOp>(dop)) {
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
          auto zero = arith::ConstantIndexOp::create(builder, alloc.getLoc(), 0);
          // get shape of kernel operand and construct a same shape
          // type with i16
          MemRefType m = mlir::dyn_cast<MemRefType>(kop.getType());
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
            auto flatAlloc = gpu::AllocOp::create(builder,
                alloc.getLoc(), ftype,
                alloc.getAsyncToken() ? alloc.getAsyncToken().getType()
                                      : nullptr,
                alloc.getAsyncDependencies(), alloc.getDynamicSizes(),
                alloc.getSymbolOperands(), alloc.getHostShared());
            // Create two views
            // 1) bf16
            // 2) i16
            auto bf16View = memref::ViewOp::create(builder,
                alloc.getLoc(), alloc.getType(), flatAlloc.getResult(0), zero,
                sizes);
            i16View = memref::ViewOp::create(builder,
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
            auto bcast = arith::BitcastOp::create(builder,
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
      }
      // Passthrough arguments are gpu.func arguments directly forwarded
      // from caller function arguments.
      // In such case, caller function argument type is updated to i16
      // This requries updating the FunctionType of caller function and
      // and the block argument type of the entry block.
      if (hasPassthroughArgs) {
        if (auto caller =
                mlir::dyn_cast<mlir::func::FuncOp>(op->getParentOp())) {
          auto oftype = caller.getFunctionType();
          llvm::SmallVector<mlir::Type, 4> argTypes;
          ArrayRef<Type> resultTypes;
          Block &eblock = caller.getBlocks().front();
          unsigned int arg_idx = 0;
          for (mlir::BlockArgument arg : eblock.getArguments()) {
            bool isPassThroughArg = true;
            Type t = oftype.getInput(arg_idx);
            for (auto use : arg.getUsers()) {
              auto callOp = dyn_cast<gpu::LaunchFuncOp>(use);
              if (!callOp) {
                isPassThroughArg = false;
                break;
              }
            }
            if (isPassThroughArg) {
              // Update block arg element type to i16
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
              MemRefType m = mlir::dyn_cast<MemRefType>(t);
              if (m) {
                Type et = m.getElementType();
                if (et.isBF16()) {
                  if (m.hasStaticShape()) {
                    llvm::ArrayRef<int64_t> s = m.getShape();
                    auto i = MemRefType::get(s, builder.getI16Type());
                    argTypes.push_back(i);
                  } else {
                    // TODO: Support dynamic shape
                    caller.emitError(
                        "Non static shape bf16 MemRefType in FuncOp inputs");
                  }
                } else {
                  argTypes.push_back(t);
                }
              } else if (t.isBF16()) {
                argTypes.push_back(builder.getI16Type());
              } else {
                argTypes.push_back(t);
              }
            } else {
              argTypes.push_back(t);
            }
            arg_idx++;
          }
          auto nftype = dyn_cast<FunctionType>(
              caller.cloneTypeWith(argTypes, resultTypes));
          caller.setFunctionType(nftype);
        }
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
