// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Conversion/NtensorToLinalg.hpp"

#include "imex/Dialect/imex_util/Dialect.hpp"
#include "imex/Dialect/imex_util/Utils.hpp"
#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Interfaces/CallInterfaces.h>
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
    if (!op->hasAttr(kReadonly))
      return mlir::failure();

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
        rewriter, op->getLoc(), dstType.getEnvironment(), std::nullopt,
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

          llvm::SmallVector<mlir::utils::IteratorType> iterators(
              rank, mlir::utils::IteratorType::parallel);
          auto bodyBuilder = [](mlir::OpBuilder &b, mlir::Location l,
                                mlir::ValueRange args) {
            assert(args.size() == 2);
            b.create<mlir::linalg::YieldOp>(l, args.front());
          };
          builder.create<mlir::linalg::GenericOp>(loc, srcTensor, dstMemref,
                                                  maps, iterators, bodyBuilder);
          return std::nullopt;
        });

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

static bool isAllTensor(mlir::TypeRange types) {
  return llvm::all_of(types, [](mlir::Type type) {
    return type.isa<imex::ntensor::NTensorType>();
  });
}

struct ConvertElementwiseOp
    : public mlir::OpRewritePattern<imex::ntensor::ElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::ElementwiseOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::ValueRange src = op.getInputs();
    mlir::TypeRange srcType = src.getTypes();
    if (srcType.empty() || !isAllTensor(srcType))
      return mlir::failure();

    mlir::TypeRange dstType = op.getResultTypes();
    if (dstType.empty() || !isAllTensor(dstType))
      return mlir::failure();

    auto type = srcType.front().cast<imex::ntensor::NTensorType>();

    for (auto range : {srcType.drop_front(), dstType}) {
      for (auto t : range) {
        auto nt = t.cast<imex::ntensor::NTensorType>();
        if (nt.getRank() != type.getRank() ||
            nt.getEnvironment() != type.getEnvironment())
          return mlir::failure();
      }
    }

    auto results = imex::util::wrapEnvRegion(
        rewriter, op.getLoc(), type.getEnvironment(), dstType,
        [&](mlir::PatternRewriter &builder, mlir::Location loc) {
          auto rank = static_cast<unsigned>(type.getRank());

          llvm::SmallVector<mlir::Value> inputs(src.size());
          for (auto [i, arg] : llvm::enumerate(src)) {
            auto srcTensorType =
                toTensorType(arg.getType().cast<imex::ntensor::NTensorType>());
            inputs[i] = builder.create<imex::ntensor::ToTensorOp>(
                loc, srcTensorType, arg);
          }

          llvm::SmallVector<mlir::Value> results(dstType.size());
          llvm::SmallVector<mlir::Type> resultTypes(dstType.size());
          llvm::SmallVector<mlir::Value> dynSizes(rank);
          for (auto [i, argType] : llvm::enumerate(dstType)) {
            auto dstTensorType =
                toTensorType(argType.cast<imex::ntensor::NTensorType>());

            dynSizes.clear();
            for (auto [i, dim] : llvm::enumerate(dstTensorType.getShape()))
              if (mlir::ShapedType::isDynamic(dim))
                dynSizes.emplace_back(builder.create<mlir::tensor::DimOp>(
                    loc, inputs.front(), i));

            results[i] = builder.create<mlir::tensor::EmptyOp>(
                loc, dstTensorType, dynSizes);
            resultTypes[i] = dstTensorType;
          }

          auto affineMap = mlir::AffineMap::getMultiDimIdentityMap(
              rank, builder.getContext());
          llvm::SmallVector<mlir::AffineMap> maps(
              srcType.size() + dstType.size(), affineMap);

          llvm::SmallVector<mlir::utils::IteratorType> iterators(
              rank, mlir::utils::IteratorType::parallel);

          auto generic = builder.create<mlir::linalg::GenericOp>(
              loc, resultTypes, inputs, results, maps, iterators);

          mlir::Region &newRegion = generic.getRegion();
          builder.inlineRegionBefore(op.getRegion(), newRegion,
                                     newRegion.end());

          mlir::Block *block = &newRegion.front();

          for (auto type : resultTypes)
            block->addArgument(type.cast<mlir::ShapedType>().getElementType(),
                               loc);

          {
            auto term = mlir::cast<imex::ntensor::ElementwiseYieldOp>(
                block->getTerminator());
            mlir::OpBuilder::InsertionGuard g(builder);
            builder.setInsertionPoint(term);
            auto args = term.getValues();
            builder.replaceOpWithNewOp<mlir::linalg::YieldOp>(term, args);
          }

          llvm::SmallVector<mlir::Value> res(generic->getNumResults());
          for (auto [i, arg] : llvm::enumerate(generic->getResults()))
            res[i] = builder.create<imex::ntensor::FromTensorOp>(
                loc, dstType[i], arg);

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
    if (!op->hasAttr(kReadonly))
      return mlir::failure();

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

/// Get broadcasted dimension value from 2 values, if v1 value is equal to 1
/// or dims are equal then select val2 otherwise val1.
static mlir::Value broadcastDim(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value val1, mlir::Value val2) {
  assert(val1.getType().isa<mlir::IndexType>());
  assert(val2.getType().isa<mlir::IndexType>());
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto isOne = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, one, val1);
  auto isSame = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, val1, val2);
  auto tmp = builder.create<mlir::arith::AndIOp>(loc, isOne, isSame);
  return builder.create<mlir::arith::SelectOp>(loc, tmp, val2, val1);
}

/// Generate code for expanding specified dim of value src to corresponding
/// value in targetShape. Assume src dimension is either 1 or equal to the
/// target shape.
static mlir::Value expandDim(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value initial, mlir::Value src, unsigned dim,
                             mlir::ValueRange targetShape) {
  assert(initial.getType().isa<mlir::RankedTensorType>());
  assert(src.getType().isa<mlir::RankedTensorType>());
  auto context = builder.getContext();
  auto srcType = src.getType().cast<mlir::ShapedType>();
  auto numDims = static_cast<unsigned>(srcType.getRank());
  auto shape = llvm::to_vector(srcType.getShape());
  shape[dim] = mlir::ShapedType::kDynamic;
  mlir::Type targetType =
      mlir::RankedTensorType::get(shape, srcType.getElementType());
  auto dimVal = builder.create<mlir::tensor::DimOp>(loc, initial, dim);
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  mlir::Value cond = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, one, dimVal);
  mlir::Value cond2 = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, targetShape[dim], dimVal);
  cond = builder.create<mlir::arith::AndIOp>(loc, cond, cond2);
  llvm::SmallVector<mlir::OpFoldResult> newShape(numDims);
  for (unsigned i = 0; i < numDims; ++i) {
    if (i == dim) {
      newShape[i] = targetShape[i];
    } else {
      newShape[i] =
          builder.create<mlir::tensor::DimOp>(loc, src, i).getResult();
    }
  }
  auto trueBody = [&](mlir::OpBuilder &builder, mlir::Location loc) {
    assert(dim < shape.size());
    shape[dim] = 1;
    auto init = builder
                    .create<mlir::tensor::EmptyOp>(loc, newShape,
                                                   srcType.getElementType())
                    .getResult();
    llvm::SmallVector<mlir::AffineExpr> exprs(numDims);
    for (unsigned i = 0; i < numDims; ++i) {
      if (i == dim) {
        exprs[i] = mlir::getAffineConstantExpr(0, context);
      } else {
        exprs[i] = mlir::getAffineDimExpr(i, context);
      }
    }
    const mlir::AffineMap maps[] = {
        mlir::AffineMap::get(numDims, 0, exprs, context),
        mlir::AffineMap::getMultiDimIdentityMap(numDims, context),
    };
    llvm::SmallVector<mlir::utils::IteratorType> iterators(
        numDims, mlir::utils::IteratorType::parallel);

    auto body = [&](mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::ValueRange values) {
      assert(values.size() == 2);
      builder.create<mlir::linalg::YieldOp>(loc, values[0]);
    };

    auto expanded = builder.create<mlir::linalg::GenericOp>(
        loc, init.getType(), src, init, maps, iterators, body);
    auto res = builder.createOrFold<mlir::tensor::CastOp>(
        loc, targetType, expanded.getResult(0));
    builder.create<mlir::scf::YieldOp>(loc, res);
  };
  auto falseBody = [&](mlir::OpBuilder &builder, mlir::Location loc) {
    mlir::Value res = src;
    if (res.getType() != targetType)
      res = builder.create<mlir::tensor::CastOp>(loc, targetType, src);
    builder.create<mlir::scf::YieldOp>(loc, res);
  };
  return builder
      .create<mlir::scf::IfOp>(loc, targetType, cond, trueBody, falseBody)
      .getResult(0);
}

/// Expand all dims of val to targetShape.
static mlir::Value expandDims(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value val, unsigned numDims,
                              mlir::ValueRange targetShape) {
  assert(numDims <= targetShape.size());
  if (numDims < targetShape.size())
    targetShape = targetShape.drop_front(targetShape.size() - numDims);

  mlir::Value current = val;
  for (unsigned i = 0; i < numDims; ++i)
    current = expandDim(builder, loc, val, current, i, targetShape);

  if (!targetShape.empty())
    current =
        builder.create<imex::util::EnforceShapeOp>(loc, current, targetShape);
  return current;
}

template <typename C> static auto getTempShape(const C &container) {
  return llvm::SmallVector<mlir::OpFoldResult>(std::begin(container),
                                               std::end(container));
}

struct ConvertBroadcastOp
    : public mlir::OpRewritePattern<imex::ntensor::BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::BroadcastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::ValueRange inputs = op.getInputs();
    if (inputs.empty())
      return mlir::failure();

    mlir::ValueRange results = op.getResults();
    assert(inputs.size() == results.size());

    for (auto [src, dst] : llvm::zip(inputs, results))
      if (src.getType().cast<mlir::ShapedType>().getElementType() !=
          dst.getType().cast<mlir::ShapedType>().getElementType())
        return mlir::failure();

    auto env = inputs.front()
                   .getType()
                   .cast<imex::ntensor::NTensorType>()
                   .getEnvironment();
    for (auto args : {inputs.drop_front(), results})
      for (auto arg : args)
        if (arg.getType().cast<imex::ntensor::NTensorType>().getEnvironment() !=
            env)
          return mlir::failure();

    mlir::TypeRange resultTypes = op->getResultTypes();

    auto newResults = imex::util::wrapEnvRegion(
        rewriter, op.getLoc(), env, resultTypes,
        [&](mlir::OpBuilder &rewriter, mlir::Location loc) {
          llvm::SmallVector<mlir::Value> tensorInputs(inputs.size());
          for (auto [i, input] : llvm::enumerate(inputs)) {
            auto tensorType = toTensorType(
                input.getType().cast<imex::ntensor::NTensorType>());
            tensorInputs[i] = rewriter.create<imex::ntensor::ToTensorOp>(
                loc, tensorType, input);
          }

          using ShapeT = llvm::SmallVector<mlir::Value>;
          auto getShape = [&](mlir::Value val) -> ShapeT {
            auto tensorType = val.getType().cast<mlir::RankedTensorType>();

            auto rank = static_cast<unsigned>(tensorType.getRank());
            ShapeT retShape(rank);
            for (auto i : llvm::seq(0u, rank))
              retShape[i] = rewriter.create<mlir::tensor::DimOp>(loc, val, i);

            return retShape;
          };

          // Compute resulting size
          auto retShape = getShape(tensorInputs.front());

          for (auto input : llvm::makeArrayRef(tensorInputs).drop_front()) {
            auto newShape = getShape(input);

            for (auto &&[dim, newDim] :
                 llvm::zip(llvm::reverse(retShape), llvm::reverse(newShape))) {
              dim = broadcastDim(rewriter, loc, dim, newDim);
            }
            if (newShape.size() > retShape.size()) {
              auto front =
                  llvm::makeArrayRef(newShape).drop_back(retShape.size());
              assert(!front.empty());
              retShape.insert(retShape.begin(), front.begin(), front.end());
            }
          }

          auto context = getContext();
          auto dstRank = static_cast<unsigned>(retShape.size());

          // Broadcast individual arrays
          llvm::SmallVector<mlir::Value> newResults(tensorInputs.size());
          for (auto [i, input] : llvm::enumerate(tensorInputs)) {
            auto srcType = input.getType().cast<mlir::ShapedType>();
            auto srcRank = static_cast<unsigned>(srcType.getRank());
            auto result = expandDims(rewriter, loc, input, srcRank, retShape);

            auto resultType = results[i].getType().cast<mlir::ShapedType>();
            if (srcRank != dstRank) {
              auto elementType = srcType.getElementType();
              auto resultTensorType = toTensorType(resultType);
              auto init = rewriter
                              .create<mlir::tensor::EmptyOp>(
                                  loc, getTempShape(retShape), elementType)
                              .getResult();

              const mlir::AffineMap maps[] = {
                  mlir::AffineMap::getMinorIdentityMap(dstRank, srcRank,
                                                       context),
                  mlir::AffineMap::getMultiDimIdentityMap(dstRank, context),
              };
              llvm::SmallVector<mlir::utils::IteratorType> iterators(
                  dstRank, mlir::utils::IteratorType::parallel);
              auto body = [&](mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::ValueRange values) {
                assert(values.size() == 2);
                auto res = values[0];
                builder.create<mlir::linalg::YieldOp>(loc, res);
              };
              result = rewriter
                           .create<mlir::linalg::GenericOp>(
                               loc, resultTensorType, result, init, maps,
                               iterators, body)
                           .getResult(0);
            }

            result = rewriter.create<imex::ntensor::FromTensorOp>(
                loc, resultType, result);
            newResults[i] = result;
          }
          return newResults;
        });

    rewriter.replaceOp(op, newResults);
    return mlir::success();
  }
};
} // namespace

void imex::populateNtensorToLinalgPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertCreateOp, ConvertCopyOp, ConvertElementwiseOp,
                  ConvertCastOp, ConvertFromElementsOp, ConvertSubviewOp,
                  ConvertLoadOp, ConvertDimOp, ConvertBroadcastOp>(
      patterns.getContext());
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
      if (mlir::isa<mlir::CallOpInterface>(op)) {
        writers.emplace_back(op);
        return;
      }

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

      if (auto create = mlir::dyn_cast<imex::ntensor::CreateArrayOp>(op))
        return create.getResult();

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
            if (auto call = mlir::dyn_cast<mlir::CallOpInterface>(writer)) {
              for (auto arg : call.getArgOperands())
                if (!analysis->alias(tens, arg).isNo())
                  return;

            } else if (analysis->getModRef(writer, tens).isMod())
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
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto &ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    imex::populateNtensorToLinalgPatterns(patterns);

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
