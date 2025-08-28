//===- VectorLinearize.cpp - VectorLinearize Pass  --------------*- C++- *-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains VectorLinearize pass.
///
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"

#include "imex/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cstdint>
#include <numeric>
#include <optional>

namespace imex {
#define GEN_PASS_DEF_VECTORLINEARIZE
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

namespace {

struct VectorLinearizePass final
    : public imex::impl::VectorLinearizeBase<VectorLinearizePass> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect, mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();

    // vector.broadcast and vector.gather requires progressive lowering
    {
      mlir::RewritePatternSet patterns(&getContext());
      mlir::vector::populateVectorBroadcastLoweringPatterns(patterns);
      mlir::vector::populateVectorGatherLoweringPatterns(patterns);
      mlir::vector::populateVectorGatherToConditionalLoadPatterns(patterns);
      // vector.transpose lowering
      // Shuffle16x16 will fallback to Shuffle1D for non 16x16 sizes.
      mlir::vector::populateVectorTransposeLoweringPatterns(
          patterns, mlir::vector::VectorTransposeLowering::Shuffle16x16);
      (void)mlir::applyPatternsGreedily(getOperation(), std::move(patterns));
    }

    // Unroll load store from <<MxN> to M <1xN> load/stores and then linearize
    {
      mlir::RewritePatternSet patterns(&getContext());
      mlir::vector::UnrollVectorOptions vectorOptions;
      vectorOptions.setNativeShapeFn(
          [](mlir::Operation *op) -> std::optional<mlir::SmallVector<int64_t>> {
            // Only unroll for vector::LoadOp and vector::StoreOp
            if (mlir::isa<mlir::vector::LoadOp>(op)) {
              if (auto vecType = mlir::dyn_cast<mlir::VectorType>(
                      op->getResult(0).getType())) {
                auto shape = vecType.getShape();
                if (shape.size() == 2)
                  return mlir::SmallVector<int64_t>{1, shape[1]};
              }
            }
            if (mlir::isa<mlir::vector::StoreOp>(op)) {
              if (auto vecType = mlir::dyn_cast<mlir::VectorType>(
                      op->getOperand(0).getType())) {
                auto shape = vecType.getShape();
                if (shape.size() == 2)
                  return mlir::SmallVector<int64_t>{1, shape[1]};
              }
            }
            return std::nullopt;
          });
      mlir::vector::populateVectorUnrollPatterns(patterns, vectorOptions);
      (void)mlir::applyPatternsGreedily(getOperation(), std::move(patterns));
    }

    // Use upstream linearization patterns
    {
      mlir::MLIRContext &context = getContext();
      mlir::TypeConverter converter;
      mlir::RewritePatternSet patterns(&context);
      mlir::ConversionTarget target(context);
      mlir::vector::populateForVectorLinearize(converter, target);
      mlir::vector::populateVectorLinearizeBasePatterns(converter, target,
                                                        patterns);
      mlir::vector::populateVectorLinearizeShuffleLikeOpsPatterns(
          converter, target, patterns);
      mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
          converter, patterns, target);
      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns))))
        return signalPassFailure();
    }

    mlir::TypeConverter typeConverter;
    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);
    typeConverter.addConversion([](mlir::Type type) { return type; });

    target.addIllegalOp<mlir::vector::TransposeOp>();
    target.addLegalOp<mlir::vector::ShapeCastOp>();
    target.addLegalOp<mlir::vector::ExtractOp>();
    target.addLegalDialect<mlir::xegpu::XeGPUDialect>();

  }
};
} // namespace

std::unique_ptr<mlir::Pass> imex::createVectorLinearizePass() {
  return std::make_unique<VectorLinearizePass>();
}

