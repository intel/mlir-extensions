// Copyright 2021 Intel Corporation
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

#include "imex/Dialect/ntensor/Transforms/ResolveArrayOps.hpp"

#include "imex/Dialect/imex_util/Dialect.hpp"
#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"

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
  if (targetType.getElementType() == valueType)
    return true;

  if (auto valueArray = valueType.dyn_cast<imex::ntensor::NTensorType>())
    return valueArray.getElementType() == targetType.getElementType();

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
        if (value.getType().isa<imex::ntensor::NTensorType>())
          return value;

        auto dstType = dst.getType().cast<imex::ntensor::NTensorType>();
        mlir::SmallVector<mlir::Value> dynamicDims;
        auto rank = static_cast<unsigned>(dstType.getRank());
        dynamicDims.reserve(rank);
        for (auto i : llvm::seq(0u, rank)) {
          if (dstType.isDynamicDim(i))
            dynamicDims.emplace_back(
                rewriter.create<imex::ntensor::DimOp>(loc, dst, i));
        }
        return rewriter.create<imex::ntensor::CreateArrayOp>(
            loc, dstType, dynamicDims, value);
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
    mlir::MLIRContext &context, mlir::RewritePatternSet &patterns) {
  patterns
      .insert<SetitemOpLowering, GetitemOpLowering, GetitemUnitupleOpLowering>(
          &context);
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

    imex::ntensor::populateResolveArrayOpsPatterns(ctx, patterns);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};
} // namespace

std::unique_ptr<mlir::Pass> imex::ntensor::createResolveArrayOpsPass() {
  return std::make_unique<ResolveArrayOpsPass>();
}
