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

#include "imex/Dialect/imex_util/dialect.hpp"
#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

template <typename F>
static llvm::SmallVector<mlir::Value>
wrapEnvRegion(mlir::OpBuilder &builder, mlir::Location loc, mlir::Attribute env,
              mlir::TypeRange results, F &&func) {
  if (!env) {
    auto res = func(builder, loc);
    mlir::ValueRange range(res);
    assert(range.getTypes() == results && "Invalid result types");
    return {range.begin(), range.end()};
  }

  auto bodyBuilder = [&](mlir::OpBuilder &b, mlir::Location l) {
    auto res = func(b, l);
    mlir::ValueRange range(res);
    assert(range.getTypes() == results && "Invalid result types");
    b.create<imex::util::EnvironmentRegionYieldOp>(l, range);
  };

  auto res = builder
                 .create<imex::util::EnvironmentRegionOp>(
                     loc, env, /*args*/ llvm::None, results, bodyBuilder)
                 .getResults();
  return {res.begin(), res.end()};
}

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

    auto results = wrapEnvRegion(
        rewriter, op->getLoc(), dstType.getEnvironment(), dstType,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          auto tensorType =
              mlir::RankedTensorType::get(dstType.getShape(), elemType);
          mlir::Value result = builder.create<mlir::tensor::EmptyOp>(
              loc, tensorType, op.getDynamicSizes());
          if (initValue)
            result =
                builder.create<mlir::linalg::FillOp>(loc, initValue, result)
                    .getResult(0);

          result =
              builder.create<imex::ntensor::FromTensorOp>(loc, dstType, result);
          return result;
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ConvertCopyOp : public mlir::OpRewritePattern<imex::ntensor::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::CopyOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto srcType = src.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!srcType)
      return mlir::failure();

    auto dst = op.getTarget();
    auto dstType = dst.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!dstType)
      return mlir::failure();

    if (srcType.getRank() != dstType.getRank() ||
        srcType.getElementType() != dstType.getElementType() ||
        srcType.getEnvironment() != dstType.getEnvironment())
      return mlir::failure();

    wrapEnvRegion(
        rewriter, op->getLoc(), dstType.getEnvironment(), llvm::None,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          auto rank = static_cast<unsigned>(srcType.getRank());

          auto srcTensorType = mlir::RankedTensorType::get(
              srcType.getShape(), srcType.getElementType());
          mlir::Value srcTensor = builder.create<imex::ntensor::ToTensorOp>(
              loc, srcTensorType, src);

          auto dstMemrefType = mlir::MemRefType::get(dstType.getShape(),
                                                     dstType.getElementType());
          mlir::Value dstMemref = builder.create<imex::ntensor::ToMemrefOp>(
              loc, dstMemrefType, dst);

          auto affineMap = mlir::AffineMap::getMultiDimIdentityMap(
              rank, builder.getContext());
          const mlir::AffineMap maps[] = {
              affineMap,
              affineMap,
          };

          llvm::SmallVector<llvm::StringRef> iterators(
              rank, mlir::getParallelIteratorTypeName());
          auto bodyBuilder = [](mlir::OpBuilder &b, mlir::Location l,
                                mlir::ValueRange args) {
            assert(args.size() == 2);
            b.create<mlir::linalg::YieldOp>(l, args.front());
          };
          builder.create<mlir::linalg::GenericOp>(loc, srcTensor, dstMemref,
                                                  maps, iterators, bodyBuilder);
          return llvm::None;
        });

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ConvertElementwiseOp
    : public mlir::OpRewritePattern<imex::ntensor::ElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::ElementwiseOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto srcType = src.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!srcType)
      return mlir::failure();

    auto dstType = op.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!dstType)
      return mlir::failure();

    if (srcType.getRank() != dstType.getRank() ||
        srcType.getEnvironment() != dstType.getEnvironment())
      return mlir::failure();

    auto loc = op->getLoc();

    auto rank = static_cast<unsigned>(srcType.getRank());

    auto srcTensorType = mlir::RankedTensorType::get(srcType.getShape(),
                                                     srcType.getElementType());
    mlir::Value srcTensor =
        rewriter.create<imex::ntensor::ToTensorOp>(loc, srcTensorType, src);

    auto dstTensorType = mlir::RankedTensorType::get(dstType.getShape(),
                                                     dstType.getElementType());

    llvm::SmallVector<mlir::Value> dynSizes;
    for (auto [i, dim] : llvm::enumerate(dstTensorType.getShape()))
      if (mlir::ShapedType::isDynamic(dim))
        dynSizes.emplace_back(
            rewriter.create<mlir::tensor::DimOp>(loc, srcTensor, i));

    mlir::Value empty =
        rewriter.create<mlir::tensor::EmptyOp>(loc, dstTensorType, dynSizes);

    auto affineMap =
        mlir::AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    const mlir::AffineMap maps[] = {
        affineMap,
        affineMap,
    };

    llvm::SmallVector<llvm::StringRef> iterators(
        rank, mlir::getParallelIteratorTypeName());

    auto generic = rewriter.create<mlir::linalg::GenericOp>(
        loc, dstTensorType, srcTensor, empty, maps, iterators);

    mlir::Region &newRegion = generic.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), newRegion, newRegion.end());

    mlir::Block *block = &newRegion.front();
    block->addArgument(dstTensorType.getElementType(), loc);
    {
      auto term =
          mlir::cast<imex::ntensor::ElementwiseYieldOp>(block->getTerminator());
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(term);
      auto arg = term.getValue();
      rewriter.replaceOpWithNewOp<mlir::linalg::YieldOp>(term, arg);
    }

    mlir::Value res = rewriter.create<imex::ntensor::FromTensorOp>(
        loc, dstType, generic.getResult(0));

    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};
} // namespace

void imex::populateNtensorToLinalgPatterns(mlir::MLIRContext &context,
                                           mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertCreateOp, ConvertCopyOp, ConvertElementwiseOp>(
      &context);
}

namespace {
struct NtensorToLinalgPass
    : public mlir::PassWrapper<NtensorToLinalgPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NtensorToLinalgPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::ntensor::NTensorDialect>();
    registry.insert<imex::util::ImexUtilDialect>();
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
