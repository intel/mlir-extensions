//===- EmulateNonNativeBF16.cpp -
// Emulate bf16 for ops that doesn't support native bf16 data type  pass
// -------------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This pass iterates gpu.func starting from top module
/// and
/// Emulate bf16 ops by extending them to f32 and truncate the result back to
/// bf16 whose SPIR-V counterpart is not natively supported
///
//===----------------------------------------------------------------------===//

#include "imex/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/TypeUtilities.h"
#include <mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include <unordered_set>

namespace imex {
#define GEN_PASS_DEF_EMULATENONNATIVEBF16
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

using namespace mlir;
using namespace imex;

namespace {
struct EmulateNonNativeBF16Pass
    : public imex::impl::EmulateNonNativeBF16Base<EmulateNonNativeBF16Pass> {

public:
  void runOnOperation() override {
    auto mod = getOperation();
    SymbolTable symbolTable(mod);
    mlir::OpBuilder builder(mod);
    // gpu::GPUFuncOp
    (void)mod.walk<WalkOrder::PreOrder>([&](gpu::GPUFuncOp op) -> WalkResult {
      // 1: Collect ops that need bf16 widening and widen those ops
      // Most ops in arith and math dialect that has bf16 operand will
      // be widened to use f32 operand

      // ATTENTION: Please be aware this pass is specifically intended for the
      // OpenCL kernels

      // Skip widening of ops, whose lowered SPIR-V counterpart is natively
      // supported.
      // One thing to keep in mind is that, the bf16 natively supported ops are
      // based on SPIR-V ops, not arith or math ops.
      // As a result, the skipped ops are identified based on their current
      // upstream lowering to SPIR-V ops, so if in the future the lowering
      // changes to other ops, this list may also need to be updated.

      // @TODO: Make this an arch-specific list and move it to XeArch.h/cpp
      std::unordered_set<std::string> nativelySupportedOps{
          "arith.bitcast", "arith.extf",    "arith.truncf", "arith.addf",
          "arith.mulf",    "arith.subf",    "arith.divf",   "arith.maximumf",
          "arith.minnumf", "arith.maxnumf", "arith.uitofp", "arith.sitofp",
          "arith.fptoui",  "arith.fptosi",  "math.absf",    "math.fma",
          "math.tanh"};
      SmallVector<Operation *, 8> widenOps;
      (void)op.getRegion().walk<WalkOrder::PreOrder>(
          [&](Operation *lop) -> WalkResult {
            auto oname = lop->getName().getStringRef();
            // Skip the natively supported operations
            if (auto nativeop = nativelySupportedOps.find(oname.str());
                nativeop != nativelySupportedOps.end())
              return WalkResult::skip();

            // For arith and math ops whose lowered SPIR-V counterpart is not
            // natively supported, emulate them with f32 upconvert and bf16
            // downconvert
            auto needWidening = false;
            if (oname.starts_with("arith.") || oname.starts_with("math.")) {
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

      return WalkResult::advance();
    });
  }
};
} // namespace

namespace imex {
std::unique_ptr<mlir::Pass> createEmulateNonNativeBF16Pass() {
  return std::make_unique<EmulateNonNativeBF16Pass>();
}
} // namespace imex
