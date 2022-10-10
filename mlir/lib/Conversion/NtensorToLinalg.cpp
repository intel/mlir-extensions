// Copyright 2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "imex/Conversion/NtensorToLinalg.hpp"

#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace {
struct ConvertCreateOp
    : public mlir::OpRewritePattern<imex::ntensor::CreateArrayOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::CreateArrayOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dstType = op.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!dstType)
      return mlir::failure();

    auto elemType = dstType.getElementType();
    auto initValue = op.getInitValue();
    if (initValue && initValue.getType() != elemType)
      return mlir::failure();

    auto tensorType = mlir::RankedTensorType::get(dstType.getShape(), elemType);

    auto loc = op->getLoc();
    mlir::Value result = rewriter.create<mlir::tensor::EmptyOp>(
        loc, tensorType, op.getDynamicSizes());
    if (initValue)
      result = rewriter.create<mlir::linalg::FillOp>(loc, initValue, result)
                   .getResult(0);

    result = rewriter.create<imex::ntensor::FromTensorOp>(loc, dstType, result);

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};
} // namespace
void imex::populateNtensorToLinalgPatterns(mlir::MLIRContext &context,
                                           mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertCreateOp>(&context);
}

namespace {
struct NtensorToLinalgPass
    : public mlir::PassWrapper<NtensorToLinalgPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NtensorToLinalgPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::ntensor::NTensorDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto &ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    imex::populateNtensorToLinalgPatterns(ctx, patterns);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};
} // namespace

std::unique_ptr<mlir::Pass> imex::createNtensorToLinalgPass() {
  return std::make_unique<NtensorToLinalgPass>();
}
