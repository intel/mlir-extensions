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

#include "imex/Dialect/imex_util/dialect.hpp"
#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

static bool isValidGetitemIndex(mlir::Type type) {
  if (type.isa<imex::ntensor::SliceType, mlir::IndexType>())
    return true;

  if (auto tupleType = type.dyn_cast<mlir::TupleType>())
    return llvm::all_of(tupleType.getTypes(), &isValidGetitemIndex);

  return false;
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

  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

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
      auto end = resolved.getEnd();
      auto step = resolved.getStep();
      auto size = builder.createOrFold<mlir::arith::SubIOp>(loc, end, begin);

      auto constStride = mlir::getConstantIntValue(step);
      if (!constStride || *constStride > 1 || *constStride < -1) {
        size = builder.createOrFold<mlir::arith::SubIOp>(loc, size, one);
        size = builder.createOrFold<mlir::arith::AddIOp>(loc, size, step);
        size = builder.createOrFold<mlir::arith::DivUIOp>(loc, size, step);
      }
      return {foldConst(begin), foldConst(size), step, true};
    } else {
      mlir::Value index =
          builder.create<imex::ntensor::ResolveIndexOp>(loc, indexVal, len);
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

static auto getDynShape(size_t rank) {
  return llvm::SmallVector<int64_t>(rank, mlir::ShapedType::kDynamicSize);
}

static mlir::Value makeSubview(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value src,
                               llvm::ArrayRef<mlir::OpFoldResult> offsets,
                               llvm::ArrayRef<mlir::OpFoldResult> sizes,
                               llvm::ArrayRef<mlir::OpFoldResult> strides,
                               llvm::ArrayRef<unsigned> dimIndices) {
  auto srcType = src.getType().cast<mlir::MemRefType>();
  auto srcRank = static_cast<unsigned>(srcType.getRank());
  auto dstRank = dimIndices.size();
  assert(srcRank > 0);
  assert(dstRank > 0);
  assert(dstRank <= srcRank);

  auto memrefType = srcType.cast<imex::ntensor::NTensorType>();
  auto resType = imex::ntensor::SubviewOp::inferResultType(memrefType, offsets,
                                                           sizes, strides);

  mlir::Value view = builder.create<imex::ntensor::SubviewOp>(
      loc, resType, src, offsets, sizes, strides);

  if (srcRank != dstRank) {
    llvm::SmallVector<mlir::OpFoldResult> newOfsets(srcRank,
                                                    builder.getIndexAttr(0));
    llvm::SmallVector<mlir::OpFoldResult> newStrides(srcRank,
                                                     builder.getIndexAttr(1));
    auto viewType = view.getType().cast<imex::ntensor::NTensorType>();
    auto reducedType = imex::ntensor::SubviewOp::inferRankReducedResultType(
        getDynShape(dstRank), viewType, newOfsets, sizes, newStrides);
    view = builder.create<imex::ntensor::SubviewOp>(
        loc, reducedType, view, newOfsets, sizes, newStrides);
    resType = reducedType;
  }

  auto flatMemrefType =
      mlir::MemRefType::get(resType.getShape(), resType.getElementType());

  if (resType != flatMemrefType)
    view =
        builder.create<imex::util::ChangeLayoutOp>(loc, flatMemrefType, view);

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

namespace {
struct SetitemOpLowering
    : public mlir::OpRewritePattern<imex::ntensor::SetitemOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::SetitemOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto target = op.getSource();
    auto targetType = target.getType().dyn_cast<mlir::MemRefType>();
    if (!targetType)
      return mlir::failure();

    auto index = op.getIndex();
    if (!isValidGetitemIndex(index.getType()))
      return mlir::failure();

    auto value = op.getValue();
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
      // TODO: Covered elsewhere
      return mlir::failure();
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
    auto memrefType = value.getType().dyn_cast<mlir::MemRefType>();
    if (!memrefType)
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
    } else {
      // Is single element
      res = rewriter.create<imex::ntensor::LoadOp>(
          loc, value, toValues(rewriter, loc, offsets));
    }
    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};
} // namespace

void imex::ntensor::populateResolveArrayOpsPatterns(
    mlir::MLIRContext &context, mlir::RewritePatternSet &patterns) {
  patterns.insert<SetitemOpLowering, GetitemOpLowering>(&context);
}

namespace {
struct ResolveArrayOpsPass
    : public mlir::PassWrapper<ResolveArrayOpsPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveArrayOpsPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithmeticDialect>();
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
