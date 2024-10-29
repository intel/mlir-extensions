//===- CastIndex.cpp - Cast Indexpass ---------------------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This pass iterates gpu.func and replaces compute intensive arith ops using
/// index dtype with i32 type. Index type is casted to and from i32 type.
///
//===----------------------------------------------------------------------===//

#include "imex/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeUtilities.h"

namespace imex {
#define GEN_PASS_DEF_CASTINDEX
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

using namespace mlir;
using namespace imex;

namespace {
struct CastIndexPass : public imex::impl::CastIndexBase<CastIndexPass> {

public:
  void runOnOperation() override {
    auto mod = getOperation();
    SymbolTable symbolTable(mod);
    mlir::OpBuilder builder(mod);
    // Visit gpu::GPUFuncOp
    (void)mod.walk<WalkOrder::PreOrder>([&](gpu::GPUFuncOp op) -> WalkResult {
      // Collect ops that need casting index to/from i32
      SmallVector<Operation *, 8> opsToCast;
      (void)op.getRegion().walk<WalkOrder::PreOrder>(
          [&](Operation *lop) -> WalkResult {
            auto oname = lop->getName().getStringRef();
            if (oname.starts_with("arith.")) {
              // Target ops to cast are hardcoded for now.
              // Current focus is on slow compute intensive ops.
              if (oname.starts_with("arith.div") ||
                  oname.starts_with("arith.rem") ||
                  oname.starts_with("arith.mul")) {
                bool needCasting = false;
                // Op must have an index type operand.
                for (const auto &oper : lop->getOperands()) {
                  if (oper.getType().isIndex()) {
                    needCasting = true;
                  }
                }
                if (needCasting) {
                  opsToCast.push_back(lop);
                }
              }
            }
            return WalkResult::advance();
          });
      for (Operation *o : opsToCast) {
        builder.setInsertionPoint(o);
        for (auto [idx, oper] : llvm::enumerate(o->getOperands())) {
          // Replace index type operands with cast op from
          // index to i32 type.
          if (oper.getType().isIndex()) {
            auto newOp = builder.create<index::CastSOp>(
                o->getLoc(), builder.getI32Type(), oper);
            o->setOperand(idx, newOp);
          }
        }
        for (mlir::OpResult res : o->getResults()) {
          if (res.getType().isIndex()) {
            // Replace index result type with i32 type
            res.setType(builder.getI32Type());
            builder.setInsertionPointAfter(o);
            // Cast i32 type back to index type
            auto newRes = builder.create<index::CastSOp>(
                o->getLoc(), builder.getIndexType(), res);
            // Replace all uase of result with new cast op
            res.replaceAllUsesExcept(newRes, newRes);
          }
        }
      }
      return WalkResult::advance();
    });
  }
};
} // namespace

namespace imex {
std::unique_ptr<mlir::Pass> createCastIndexPass() {
  return std::make_unique<CastIndexPass>();
}
} // namespace imex
