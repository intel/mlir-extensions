// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Dialect/ntensor/Transforms/ResolveArrayOps.hpp"

#include "imex/Dialect/imex_util/Dialect.hpp"
#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"
#include "imex/Transforms/CastUtils.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

static bool isIndexOrSlice(mlir::Type type) {
  return type
      .isa<imex::ntensor::SliceType, mlir::IndexType, mlir::IntegerType>();
}

static bool isValidGetitemIndex(mlir::Type type) {
  if (isIndexOrSlice(type))
    return true;

  if (auto tupleType = type.dyn_cast<mlir::TupleType>())
    return llvm::all_of(tupleType.getTypes(), &isIndexOrSlice);

  return false;
}

static mlir::Value convertIndex(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value value) {
  auto intType = value.getType().dyn_cast<mlir::IntegerType>();
  if (intType) {
    if (intType.getSignedness() != mlir::IntegerType::Signless) {
      auto signlessType =
          mlir::IntegerType::get(builder.getContext(), intType.getWidth());
      value = builder.create<imex::util::SignCastOp>(loc, signlessType, value);
    }

    auto indexType = builder.getIndexType();
    value = builder.create<mlir::arith::IndexCastOp>(loc, indexType, value);
  }

  return value;
}

static mlir::LogicalResult
computeIndices(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
               mlir::Value index,
               llvm::SmallVectorImpl<mlir::OpFoldResult> &offsets,
               llvm::SmallVectorImpl<mlir::OpFoldResult> &sizes,
               llvm::SmallVectorImpl<mlir::OpFoldResult> &strides,
               llvm::SmallVectorImpl<unsigned> &dimsIndices) {
  auto shapedType = value.getType().cast<mlir::ShapedType>();

  auto getDim = [&](unsigned dim) -> mlir::Value {
    return builder.create<imex::ntensor::DimOp>(loc, value, dim);
  };

  auto foldConst = [&](mlir::Value val) -> mlir::OpFoldResult {
    if (auto intVal = mlir::getConstantIntValue(val))
      return builder.getIndexAttr(*intVal);

    return val;
  };

  auto getPos =
      [&](mlir::Value indexVal,
          unsigned dim) -> std::tuple<mlir::OpFoldResult, mlir::OpFoldResult,
                                      mlir::OpFoldResult, bool> {
    auto valType = indexVal.getType();
    auto len = getDim(dim);
    if (valType.isa<imex::ntensor::SliceType>()) {
      auto resolved =
          builder.create<imex::ntensor::ResolveSliceOp>(loc, indexVal, len);

      auto begin = resolved.getBegin();
      auto step = resolved.getStep();
      auto size = resolved.getCount();
      return {foldConst(begin), foldConst(size), foldConst(step), true};
    } else {
      mlir::Value index = convertIndex(builder, loc, indexVal);
      index = builder.create<imex::ntensor::ResolveIndexOp>(loc, index, len);
      return {index, builder.getIndexAttr(1), builder.getIndexAttr(1), false};
    }
  };

  auto makeFullSlice =
      [&](unsigned dim) -> std::tuple<mlir::OpFoldResult, mlir::OpFoldResult,
                                      mlir::OpFoldResult> {
    auto begin = builder.getIndexAttr(0);
    auto end = getDim(dim);
    auto step = builder.getIndexAttr(1);
    return {begin, end, step};
  };

  auto rank = static_cast<unsigned>(shapedType.getRank());
  offsets.resize(rank);
  sizes.resize(rank);
  strides.resize(rank);

  if (auto tupleType = index.getType().dyn_cast<mlir::TupleType>()) {
    auto count = static_cast<unsigned>(tupleType.size());
    if (count > rank)
      return mlir::failure();

    for (auto it : llvm::enumerate(tupleType)) {
      auto i = static_cast<unsigned>(it.index());
      auto getitemInd = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
      auto ind = builder.createOrFold<imex::util::TupleExtractOp>(
          loc, it.value(), index, getitemInd);
      bool isSlice = false;
      std::tie(offsets[i], sizes[i], strides[i], isSlice) = getPos(ind, i);
      if (isSlice)
        dimsIndices.emplace_back(i);
    }

    for (auto i : llvm::seq(count, rank)) {
      std::tie(offsets[i], sizes[i], strides[i]) = makeFullSlice(i);
      dimsIndices.emplace_back(i);
    }
  } else {
    bool isSlice = false;
    std::tie(offsets[0], sizes[0], strides[0], isSlice) = getPos(index, 0);
    if (isSlice)
      dimsIndices.emplace_back(0);

    for (auto i : llvm::seq(1u, rank)) {
      std::tie(offsets[i], sizes[i], strides[i]) = makeFullSlice(i);
      dimsIndices.emplace_back(i);
    }
  }

  return mlir::success();
}

static mlir::Value makeSubview(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value src,
                               llvm::ArrayRef<mlir::OpFoldResult> offsets,
                               llvm::ArrayRef<mlir::OpFoldResult> sizes,
                               llvm::ArrayRef<mlir::OpFoldResult> strides,
                               llvm::ArrayRef<unsigned> dimIndices) {
  auto srcType = src.getType().cast<imex::ntensor::NTensorType>();
  auto srcRank = static_cast<unsigned>(srcType.getRank());
  auto dstRank = dimIndices.size();
  assert(srcRank > 0);
  assert(dstRank > 0);
  assert(dstRank <= srcRank);

  auto resType = imex::ntensor::SubviewOp::inferResultType(srcType, offsets,
                                                           sizes, strides);

  mlir::Value view = builder.create<imex::ntensor::SubviewOp>(
      loc, resType, src, offsets, sizes, strides);

  if (srcRank != dstRank) {
    llvm::SmallVector<mlir::OpFoldResult> newOfsets(srcRank,
                                                    builder.getIndexAttr(0));
    llvm::SmallVector<mlir::OpFoldResult> newStrides(srcRank,
                                                     builder.getIndexAttr(1));
    auto viewType = view.getType().cast<imex::ntensor::NTensorType>();

    llvm::SmallVector<int64_t> dstShape(dstRank);
    for (auto [i, ind] : llvm::enumerate(dimIndices)) {
      auto sz = sizes[ind];
      if (auto szVal = sz.dyn_cast<mlir::Attribute>()) {
        dstShape[i] = szVal.cast<mlir::IntegerAttr>().getValue().getSExtValue();
      } else {
        dstShape[i] = mlir::ShapedType::kDynamicSize;
      }
    }

    auto reducedType = imex::ntensor::SubviewOp::inferRankReducedResultType(
        dstShape, viewType, newOfsets, sizes, newStrides);
    view = builder.create<imex::ntensor::SubviewOp>(
        loc, reducedType, view, newOfsets, sizes, newStrides);
    resType = reducedType;
  }

  return view;
}

static llvm::SmallVector<mlir::Value>
toValues(mlir::OpBuilder &builder, mlir::Location loc,
         mlir::ArrayRef<mlir::OpFoldResult> vals) {
  llvm::SmallVector<mlir::Value> ret(vals.size());
  for (auto it : llvm::enumerate(vals)) {
    auto i = it.index();
    auto val = it.value();
    if (auto attr = val.dyn_cast<mlir::Attribute>()) {
      ret[i] = builder.create<mlir::arith::ConstantIndexOp>(
          loc, attr.cast<mlir::IntegerAttr>().getValue().getSExtValue());
    } else {
      ret[i] = val.template get<mlir::Value>();
    }
  }

  return ret;
}

static bool isCompatibleSetitemValue(mlir::Type valueType,
                                     imex::ntensor::NTensorType targetType) {
  if (imex::canConvert(targetType.getElementType(), valueType))
    return true;

  if (auto valueArray = valueType.dyn_cast<imex::ntensor::NTensorType>())
    return imex::canConvert(valueArray.getElementType(),
                            targetType.getElementType());

  return false;
}

namespace {
struct SetitemOpLowering
    : public mlir::OpRewritePattern<imex::ntensor::SetitemOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::SetitemOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto target = op.getSource();
    auto targetType = target.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!targetType)
      return mlir::failure();

    auto index = op.getIndex();
    if (!isValidGetitemIndex(index.getType()))
      return mlir::failure();

    auto value = op.getValue();
    if (!isCompatibleSetitemValue(value.getType(), targetType))
      return mlir::failure();

    auto loc = op.getLoc();
    llvm::SmallVector<mlir::OpFoldResult> offsets;
    llvm::SmallVector<mlir::OpFoldResult> sizes;
    llvm::SmallVector<mlir::OpFoldResult> strides;
    llvm::SmallVector<unsigned> dimsIndices;
    if (mlir::failed(computeIndices(rewriter, loc, target, index, offsets,
                                    sizes, strides, dimsIndices)))
      return mlir::failure();

    if (!dimsIndices.empty()) {
      // Is slice
      auto dst = makeSubview(rewriter, loc, target, offsets, sizes, strides,
                             dimsIndices);

      auto newArray = [&]() -> mlir::Value {
        if (auto srcType =
                value.getType().dyn_cast<imex::ntensor::NTensorType>()) {
          auto dstElementType = targetType.getElementType();
          if (srcType.getElementType() != dstElementType) {
            auto bodyBuilder = [&](mlir::OpBuilder &b, mlir::Location l,
                                   mlir::ValueRange vals) {
              assert(vals.size() == 1);
              auto res = imex::doConvert(b, l, vals.front(), dstElementType);
              assert(res);
              b.create<imex::ntensor::ElementwiseYieldOp>(l, res);
            };

            return rewriter
                .create<imex::ntensor::ElementwiseOp>(loc, targetType, value,
                                                      bodyBuilder)
                .getResult(0);
          }
          return value;
        }

        auto dstType = dst.getType().cast<imex::ntensor::NTensorType>();
        mlir::SmallVector<mlir::Value> dynamicDims;
        auto rank = static_cast<unsigned>(dstType.getRank());
        dynamicDims.reserve(rank);
        for (auto i : llvm::seq(0u, rank)) {
          if (dstType.isDynamicDim(i))
            dynamicDims.emplace_back(
                rewriter.create<imex::ntensor::DimOp>(loc, dst, i));
        }
        auto dstVal =
            imex::doConvert(rewriter, loc, value, dstType.getElementType());
        assert(dstVal);
        return rewriter.create<imex::ntensor::CreateArrayOp>(
            loc, dstType, dynamicDims, dstVal);
      }();
      rewriter.replaceOpWithNewOp<imex::ntensor::CopyOp>(op, newArray, dst);
    } else {
      // Is single element
      rewriter.replaceOpWithNewOp<imex::ntensor::StoreOp>(
          op, value, target, toValues(rewriter, loc, offsets));
    }

    return mlir::success();
  }
};

struct SetitemMaskOpLowering
    : public mlir::OpRewritePattern<imex::ntensor::SetitemOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::SetitemOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto target = op.getSource();
    auto targetType = target.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!targetType)
      return mlir::failure();

    auto mask = op.getIndex();
    auto maskType = mask.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!maskType || maskType.getElementType() != rewriter.getI1Type() ||
        maskType.getRank() != targetType.getRank() ||
        maskType.getEnvironment() != targetType.getEnvironment())
      return mlir::failure();

    auto val = op.getValue();
    if (!imex::ntensor::NTensorType::isValidElementType(val.getType()))
      return mlir::failure();

    auto elemType = targetType.getElementType();
    if (!imex::canConvert(val.getType(), elemType))
      return mlir::failure();

    auto loc = op.getLoc();
    mlir::SmallVector<mlir::Value> dynamicDims;
    auto rank = static_cast<unsigned>(targetType.getRank());
    dynamicDims.reserve(rank);
    for (auto i : llvm::seq(0u, rank)) {
      if (targetType.isDynamicDim(i))
        dynamicDims.emplace_back(
            rewriter.create<imex::ntensor::DimOp>(loc, target, i));
    }
    auto dstVal = imex::doConvert(rewriter, loc, val, elemType);
    assert(dstVal);
    val = rewriter.create<imex::ntensor::CreateArrayOp>(loc, targetType,
                                                        dynamicDims, dstVal);

    auto bodyBuilder = [&](mlir::OpBuilder &b, mlir::Location l,
                           mlir::ValueRange vals) {
      llvm::errs() << "asdasdas " << vals.size() << "'n";
      assert(vals.size() == 3);
      auto cond = vals[0];
      auto val = vals[1];
      auto src = vals[2];
      mlir::Value res = b.create<mlir::arith::SelectOp>(l, cond, val, src);
      b.create<imex::ntensor::ElementwiseYieldOp>(l, res);
    };

    mlir::Value args[] = {mask, val, target};
    auto res = rewriter
                   .create<imex::ntensor::ElementwiseOp>(loc, targetType, args,
                                                         bodyBuilder)
                   .getResult(0);

    rewriter.create<imex::ntensor::CopyOp>(loc, res, target);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct GetitemOpLowering
    : public mlir::OpRewritePattern<imex::ntensor::GetitemOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::GetitemOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto value = op.getSource();
    auto index = op.getIndex();
    if (!value.getType().isa<imex::ntensor::NTensorType>())
      return mlir::failure();

    if (!isValidGetitemIndex(index.getType()))
      return mlir::failure();

    auto loc = op.getLoc();
    llvm::SmallVector<mlir::OpFoldResult> offsets;
    llvm::SmallVector<mlir::OpFoldResult> sizes;
    llvm::SmallVector<mlir::OpFoldResult> strides;
    llvm::SmallVector<unsigned> dimsIndices;
    if (mlir::failed(computeIndices(rewriter, loc, value, index, offsets, sizes,
                                    strides, dimsIndices)))
      return mlir::failure();

    mlir::Value res;
    if (!dimsIndices.empty()) {
      // Is slice
      res = makeSubview(rewriter, loc, value, offsets, sizes, strides,
                        dimsIndices);
      auto resType = op.getResult().getType();
      if (res.getType() != resType)
        res = rewriter.create<imex::ntensor::CastOp>(loc, resType, res);
    } else {
      // Is single element
      res = rewriter.create<imex::ntensor::LoadOp>(
          loc, value, toValues(rewriter, loc, offsets));
    }
    assert(res.getType() == op.getResult().getType() && "Invalid result type");
    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};

static llvm::Optional<mlir::Type> isUnituple(mlir::Type type) {
  auto tupleType = type.dyn_cast<mlir::TupleType>();
  if (!tupleType || tupleType.size() == 0)
    return llvm::None;

  auto types = tupleType.getTypes();
  auto ret = types.front();
  if (llvm::any_of(types.drop_front(), [&](mlir::Type t) { return t != ret; }))
    return llvm::None;

  return ret;
}

struct GetitemUnitupleOpLowering
    : public mlir::OpRewritePattern<imex::ntensor::GetitemOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::GetitemOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto value = op.getSource();
    auto index = op.getIndex();
    if (!isValidGetitemIndex(index.getType()))
      return mlir::failure();

    auto valType = value.getType();
    auto elemType = isUnituple(valType);
    if (!elemType || *elemType != op.getType() ||
        !imex::ntensor::NTensorType::isValidElementType(*elemType))
      return mlir::failure();

    auto count = static_cast<int64_t>(valType.cast<mlir::TupleType>().size());
    auto arrayType = imex::ntensor::NTensorType::get(count, *elemType);

    auto loc = op->getLoc();

    llvm::SmallVector<mlir::Value> elements(static_cast<size_t>(count));
    for (auto i : llvm::seq<int64_t>(0, count)) {
      auto idx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
      elements[i] = rewriter.create<imex::util::TupleExtractOp>(loc, *elemType,
                                                                value, idx);
    }

    auto array = rewriter.create<imex::ntensor::FromElementsOp>(loc, arrayType,
                                                                elements);

    auto dynArrayType = imex::ntensor::NTensorType::get(
        mlir::ShapedType::kDynamicSize, *elemType);
    auto dynArray =
        rewriter.create<imex::ntensor::CastOp>(loc, dynArrayType, array);
    rewriter.replaceOpWithNewOp<imex::ntensor::GetitemOp>(op, op.getType(),
                                                          dynArray, index);
    return mlir::success();
  }
};
} // namespace

void imex::ntensor::populateResolveArrayOpsPatterns(
    mlir::RewritePatternSet &patterns) {
  patterns.insert<SetitemOpLowering, SetitemMaskOpLowering, GetitemOpLowering,
                  GetitemUnitupleOpLowering>(patterns.getContext());
}

namespace {
struct ResolveArrayOpsPass
    : public mlir::PassWrapper<ResolveArrayOpsPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveArrayOpsPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<imex::ntensor::NTensorDialect>();
    registry.insert<imex::util::ImexUtilDialect>();
  }

  void runOnOperation() override {
    auto &ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    imex::ntensor::populateResolveArrayOpsPatterns(patterns);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};
} // namespace

std::unique_ptr<mlir::Pass> imex::ntensor::createResolveArrayOpsPass() {
  return std::make_unique<ResolveArrayOpsPass>();
}
