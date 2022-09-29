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

#include "imex/Conversion/ntensor_to_memref.hpp"
#include "imex/Dialect/imex_util/dialect.hpp"
#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"
#include "imex/Transforms/cast_utils.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

static bool isValidGetitemIndex(mlir::Type type) {
  if (type.isa<imex::ntensor::SliceType>())
    return true;

  if (auto tupleType = type.dyn_cast<mlir::TupleType>())
    return llvm::all_of(tupleType.getTypes(), &isValidGetitemIndex);

  return type.isa<mlir::IndexType>();
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
    return builder.create<mlir::memref::DimOp>(loc, value, dim);
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

  auto memrefType = srcType.cast<mlir::MemRefType>();
  auto resType = mlir::memref::SubViewOp::inferResultType(memrefType, offsets,
                                                          sizes, strides)
                     .cast<mlir::MemRefType>();

  mlir::Value view = builder.create<mlir::memref::SubViewOp>(
      loc, resType, src, offsets, sizes, strides);

  if (srcRank != dstRank) {
    llvm::SmallVector<mlir::OpFoldResult> newOfsets(srcRank,
                                                    builder.getIndexAttr(0));
    llvm::SmallVector<mlir::OpFoldResult> newStrides(srcRank,
                                                     builder.getIndexAttr(1));
    auto viewType = view.getType().cast<mlir::MemRefType>();
    auto reducedType =
        mlir::memref::SubViewOp::inferRankReducedResultType(
            getDynShape(dstRank), viewType, newOfsets, sizes, newStrides)
            .cast<mlir::MemRefType>();
    view = builder.create<mlir::memref::SubViewOp>(
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

namespace {
struct SetitemOpLowering
    : public mlir::OpConversionPattern<imex::ntensor::SetitemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::SetitemOp op,
                  imex::ntensor::SetitemOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto target = adaptor.getSource();
    auto targetType = target.getType().dyn_cast<mlir::MemRefType>();
    if (!targetType)
      return mlir::failure();

    auto index = adaptor.getIndex();
    if (!isValidGetitemIndex(index.getType()))
      return mlir::failure();

    auto value = adaptor.getValue();
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
      auto toValues = [&](auto &vals) {
        llvm::SmallVector<mlir::Value> ret(vals.size());
        for (auto it : llvm::enumerate(vals)) {
          auto i = it.index();
          auto val = it.value();
          if (auto attr = val.template dyn_cast<mlir::Attribute>()) {
            ret[i] = rewriter.create<mlir::arith::ConstantIndexOp>(
                loc, attr.template cast<mlir::IntegerAttr>()
                         .getValue()
                         .getSExtValue());
          } else {
            ret[i] = val.template get<mlir::Value>();
          }
        }

        return ret;
      };

      rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, value, target,
                                                         toValues(offsets));
    }

    return mlir::success();
  }
};

struct GetitemOpLowering
    : public mlir::OpConversionPattern<imex::ntensor::GetitemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::GetitemOp op,
                  imex::ntensor::GetitemOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto value = adaptor.getSource();
    auto index = adaptor.getIndex();
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
      auto toValues = [&](auto &vals) {
        llvm::SmallVector<mlir::Value> ret(vals.size());
        for (auto it : llvm::enumerate(vals)) {
          auto i = it.index();
          auto val = it.value();
          if (auto attr = val.template dyn_cast<mlir::Attribute>()) {
            ret[i] = rewriter.create<mlir::arith::ConstantIndexOp>(
                loc, attr.template cast<mlir::IntegerAttr>()
                         .getValue()
                         .getSExtValue());
          } else {
            ret[i] = val.template get<mlir::Value>();
          }
        }

        return ret;
      };
      res =
          rewriter.create<mlir::memref::LoadOp>(loc, value, toValues(offsets));
    }
    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};
} // namespace

void imex::populateNtensorToMemrefPatterns(mlir::MLIRContext &context,
                                           mlir::TypeConverter &converter,
                                           mlir::RewritePatternSet &patterns,
                                           mlir::ConversionTarget &target) {
  converter.addConversion(
      [](imex::ntensor::NTensorType type) -> llvm::Optional<mlir::Type> {
        auto elemType = type.getElementType();
        if (mlir::MemRefType::isValidElementType(elemType))
          return mlir::MemRefType::get(type.getShape(), elemType);

        return llvm::None;
      });
}

namespace {
struct NtensorToMemrefPass
    : public mlir::PassWrapper<NtensorToMemrefPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NtensorToMemrefPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::util::ImexUtilDialect>();
    registry.insert<mlir::arith::ArithmeticDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
    mlir::TypeConverter converter;
    mlir::RewritePatternSet patterns(&context);
    mlir::ConversionTarget target(context);

    imex::populateNtensorToMemrefPatterns(context, converter, patterns, target);

    auto op = getOperation();
    if (mlir::failed(
            mlir::applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> imex::createNtensorToMemrefPass() {
  return std::make_unique<NtensorToMemrefPass>();
}
