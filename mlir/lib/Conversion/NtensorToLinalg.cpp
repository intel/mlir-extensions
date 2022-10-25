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

#include "imex/Dialect/imex_util/Dialect.hpp"
#include "imex/Dialect/imex_util/Utils.hpp"
#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

static const constexpr llvm::StringLiteral kReadonly("ntensor_readonly");

static mlir::RankedTensorType toTensorType(mlir::ShapedType type) {
  return mlir::RankedTensorType::get(type.getShape(), type.getElementType());
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

    auto results = imex::util::wrapEnvRegion(
        rewriter, op->getLoc(), dstType.getEnvironment(), dstType,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          auto tensorType = toTensorType(dstType);
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

    imex::util::wrapEnvRegion(
        rewriter, op->getLoc(), dstType.getEnvironment(), llvm::None,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          auto rank = static_cast<unsigned>(srcType.getRank());

          auto srcTensorType = toTensorType(srcType);
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

    auto results = imex::util::wrapEnvRegion(
        rewriter, op->getLoc(), dstType.getEnvironment(), dstType,
        [&](mlir::PatternRewriter &builder, mlir::Location loc) {
          auto rank = static_cast<unsigned>(srcType.getRank());

          auto srcTensorType = toTensorType(srcType);
          mlir::Value srcTensor = builder.create<imex::ntensor::ToTensorOp>(
              loc, srcTensorType, src);

          auto dstTensorType = toTensorType(dstType);

          llvm::SmallVector<mlir::Value> dynSizes;
          for (auto [i, dim] : llvm::enumerate(dstTensorType.getShape()))
            if (mlir::ShapedType::isDynamic(dim))
              dynSizes.emplace_back(
                  builder.create<mlir::tensor::DimOp>(loc, srcTensor, i));

          mlir::Value empty = builder.create<mlir::tensor::EmptyOp>(
              loc, dstTensorType, dynSizes);

          auto affineMap = mlir::AffineMap::getMultiDimIdentityMap(
              rank, builder.getContext());
          const mlir::AffineMap maps[] = {
              affineMap,
              affineMap,
          };

          llvm::SmallVector<llvm::StringRef> iterators(
              rank, mlir::getParallelIteratorTypeName());

          auto generic = builder.create<mlir::linalg::GenericOp>(
              loc, dstTensorType, srcTensor, empty, maps, iterators);

          mlir::Region &newRegion = generic.getRegion();
          builder.inlineRegionBefore(op.getRegion(), newRegion,
                                     newRegion.end());

          mlir::Block *block = &newRegion.front();
          block->addArgument(dstTensorType.getElementType(), loc);
          {
            auto term = mlir::cast<imex::ntensor::ElementwiseYieldOp>(
                block->getTerminator());
            mlir::OpBuilder::InsertionGuard g(builder);
            builder.setInsertionPoint(term);
            auto arg = term.getValue();
            builder.replaceOpWithNewOp<mlir::linalg::YieldOp>(term, arg);
          }

          mlir::Value res = builder.create<imex::ntensor::FromTensorOp>(
              loc, dstType, generic.getResult(0));
          return res;
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ConvertCastOp : public mlir::OpRewritePattern<imex::ntensor::CastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::CastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto srcType = src.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!srcType)
      return mlir::failure();

    auto dstType = op.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!dstType)
      return mlir::failure();

    if (srcType.getEnvironment() != dstType.getEnvironment())
      return mlir::failure();

    auto srcTensorType = toTensorType(srcType);
    auto dstTensorType = toTensorType(dstType);

    if (!mlir::tensor::CastOp::areCastCompatible(srcTensorType, dstTensorType))
      return mlir::failure();

    auto results = imex::util::wrapEnvRegion(
        rewriter, op->getLoc(), dstType.getEnvironment(), dstType,
        [&](mlir::PatternRewriter &builder, mlir::Location loc) {
          auto srcTensor = builder.create<imex::ntensor::ToTensorOp>(
              loc, srcTensorType, src);
          auto cast = builder.create<mlir::tensor::CastOp>(loc, dstTensorType,
                                                           srcTensor);
          return builder.create<imex::ntensor::FromTensorOp>(loc, dstType, cast)
              .getResult();
        });
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ConvertFromElementsOp
    : public mlir::OpRewritePattern<imex::ntensor::FromElementsOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::FromElementsOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dstType = op.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!dstType)
      return mlir::failure();

    auto elements = op.getElements();
    if (llvm::any_of(elements.getTypes(), [&](mlir::Type t) {
          return t != dstType.getElementType();
        }))
      return mlir::failure();

    auto dstTensorType = toTensorType(dstType);

    auto results = imex::util::wrapEnvRegion(
        rewriter, op->getLoc(), dstType.getEnvironment(), dstType,
        [&](mlir::PatternRewriter &builder, mlir::Location loc) {
          auto res = builder.create<mlir::tensor::FromElementsOp>(
              loc, dstTensorType, elements);
          return builder.create<imex::ntensor::FromTensorOp>(loc, dstType, res)
              .getResult();
        });
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ConvertSubviewOp
    : public mlir::OpRewritePattern<imex::ntensor::SubviewOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::SubviewOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto srcType = src.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!srcType)
      return mlir::failure();

    auto dstType = op.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!dstType)
      return mlir::failure();

    if (srcType.getEnvironment() != dstType.getEnvironment())
      return mlir::failure();

    auto results = imex::util::wrapEnvRegion(
        rewriter, op->getLoc(), dstType.getEnvironment(), dstType,
        [&](mlir::PatternRewriter &builder, mlir::Location loc) {
          auto srcTensorType = toTensorType(srcType);
          mlir::Value srcTensor = builder.create<imex::ntensor::ToTensorOp>(
              loc, srcTensorType, src);

          auto offsets = op.getMixedOffsets();
          auto sizes = op.getMixedSizes();
          auto strides = op.getMixedStrides();

          auto dstRank = static_cast<unsigned>(dstType.getRank());
          auto viewTensorType =
              mlir::tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                  dstRank, srcTensorType, offsets, sizes, strides);

          mlir::Value view = builder.create<mlir::tensor::ExtractSliceOp>(
              loc, viewTensorType, srcTensor, offsets, sizes, strides);
          mlir::Value result =
              builder.create<imex::ntensor::FromTensorOp>(loc, dstType, view);
          return result;
        });
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ConvertLoadOp : public mlir::OpRewritePattern<imex::ntensor::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getArray();
    auto srcType = src.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!srcType || op.getType() != srcType.getElementType())
      return mlir::failure();

    auto results = imex::util::wrapEnvRegion(
        rewriter, op->getLoc(), srcType.getEnvironment(),
        srcType.getElementType(),
        [&](mlir::PatternRewriter &builder, mlir::Location loc) {
          auto srcTensorType = toTensorType(srcType);
          mlir::Value srcTensor = builder.create<imex::ntensor::ToTensorOp>(
              loc, srcTensorType, src);

          mlir::Value result = builder.create<mlir::tensor::ExtractOp>(
              loc, srcTensor, op.getIndices());
          return result;
        });
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ConvertDimOp : public mlir::OpRewritePattern<imex::ntensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::DimOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto srcType = src.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!srcType)
      return mlir::failure();

    auto results = imex::util::wrapEnvRegion(
        rewriter, op->getLoc(), srcType.getEnvironment(),
        rewriter.getIndexType(),
        [&](mlir::PatternRewriter &builder, mlir::Location loc) {
          auto tensorType = toTensorType(srcType);
          mlir::Value tensor =
              builder.create<imex::ntensor::ToTensorOp>(loc, tensorType, src);
          mlir::Value result =
              builder.create<mlir::tensor::DimOp>(loc, tensor, op.getIndex());
          return result;
        });
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};
} // namespace

void imex::populateNtensorToLinalgPatterns(mlir::MLIRContext &context,
                                           mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertCreateOp, ConvertCopyOp, ConvertElementwiseOp,
                  ConvertCastOp, ConvertFromElementsOp, ConvertSubviewOp,
                  ConvertLoadOp, ConvertDimOp>(&context);
}

namespace {
struct NtensorAliasAnalysisPass
    : public mlir::PassWrapper<NtensorAliasAnalysisPass,
                               mlir::InterfacePass<mlir::FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NtensorAliasAnalysisPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::ntensor::NTensorDialect>();
  }

  void runOnOperation() override {
    auto &context = getContext();
    auto func = getOperation();

    auto *ntensorDialect =
        context.getOrLoadDialect<imex::ntensor::NTensorDialect>();
    assert(ntensorDialect);

    llvm::SmallVector<mlir::Operation *, 0> writers;
    func->walk([&](mlir::Operation *op) {
      if (op->getDialect() != ntensorDialect)
        return;

      auto memInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
      if (!memInterface || memInterface.hasEffect<mlir::MemoryEffects::Write>())
        writers.emplace_back(op);
    });

    bool hasWriters = !writers.empty();
    auto *analysis = [&]() -> mlir::AliasAnalysis * {
      if (!hasWriters)
        return nullptr;

      return &getAnalysis<mlir::AliasAnalysis>();
    }();

    auto getTensor = [](mlir::Operation *op) -> mlir::Value {
      assert(op);
      if (auto subview = mlir::dyn_cast<imex::ntensor::SubviewOp>(op))
        return subview.getResult();

      return {};
    };

    auto attrName = mlir::StringAttr::get(&context, kReadonly);
    auto unitAttr = mlir::UnitAttr::get(&context);
    func->walk([&](mlir::Operation *op) {
      if (auto tens = getTensor(op)) {
        if (hasWriters) {
          op->removeAttr(attrName);
          assert(analysis);
          for (auto writer : writers) {
            assert(writer);
            if (analysis->getModRef(writer, tens).isMod())
              return;
          }
        }
        op->setAttr(attrName, unitAttr);
      }
    });
    markAllAnalysesPreserved();
  }
};

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

std::unique_ptr<mlir::Pass> imex::createNtensorAliasAnalysisPass() {
  return std::make_unique<NtensorAliasAnalysisPass>();
}

std::unique_ptr<mlir::Pass> imex::createNtensorToLinalgPass() {
  return std::make_unique<NtensorToLinalgPass>();
}
