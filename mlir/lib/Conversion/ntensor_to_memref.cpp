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
#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"
#include "imex/Dialect/imex_util/dialect.hpp"
#include "imex/Transforms/cast_utils.hpp"

#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

static bool isValidGetitemIndex(mlir::Type type) {
//  if (type.isa<plier::SliceType>())
//    return true;

  if (auto tupleType = type.dyn_cast<mlir::TupleType>())
    return llvm::all_of(tupleType.getTypes(), &isValidGetitemIndex);

  return type.isa<mlir::IntegerType, mlir::IndexType>();
}

static mlir::LogicalResult
computeIndices(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
               mlir::Value index,
               llvm::SmallVectorImpl<mlir::OpFoldResult> &offsets,
               llvm::SmallVectorImpl<mlir::OpFoldResult> &sizes,
               llvm::SmallVectorImpl<mlir::OpFoldResult> &strides,
               llvm::SmallVectorImpl<unsigned> &dimsIndices) {
  auto shapedType = value.getType().cast<mlir::MemRefType>();
  auto indexType = builder.getIndexType();

  auto getDim = [&](unsigned dim) -> mlir::Value {
    return builder.createOrFold<mlir::memref::DimOp>(loc, value, dim);
  };

  auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
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
    bool ignoreNegativeInd = false;
    auto handleNegativeVal = [&](mlir::OpFoldResult val) -> mlir::Value {
      mlir::Value idx;
      if (auto v = val.dyn_cast<mlir::Value>()) {
        idx = v;
      } else {
        auto attr = val.get<mlir::Attribute>();
        auto attrVal = attr.cast<mlir::IntegerAttr>().getValue().getSExtValue();
        idx = builder.create<mlir::arith::ConstantIndexOp>(loc, attrVal);
      }
      if (ignoreNegativeInd) {
        return idx;
      } else {
        auto isNeg = builder.createOrFold<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::slt, idx, zero);
        auto negIndex =
            builder.createOrFold<mlir::arith::AddIOp>(loc, len, idx);
        return builder.createOrFold<mlir::arith::SelectOp>(loc, isNeg, negIndex,
                                                           idx);
      }
    };

    /*if (auto sliceType = valType.dyn_cast<plier::SliceType>()) {
      auto getItemOrConst = [&](unsigned i) -> mlir::Value {
        assert(i < 3);
        auto createInd = [&](int64_t i) {
          return builder.create<mlir::arith::ConstantIndexOp>(loc, i);
        };
        return builder.createOrFold<plier::SliceGetItemOp>(
            loc, indexType, indexVal, value, createInd(i), dim);
      };

      auto offset = handleNegativeVal(getItemOrConst(0));
      auto end = handleNegativeVal(getItemOrConst(1));
      auto stride = getItemOrConst(2);
      auto size = builder.createOrFold<mlir::arith::SubIOp>(loc, end, offset);

      auto constStride = mlir::getConstantIntValue(stride);
      if (!constStride || *constStride > 1 || *constStride < -1) {
        size = builder.createOrFold<mlir::arith::SubIOp>(loc, size, one);
        size = builder.createOrFold<mlir::arith::AddIOp>(loc, size, stride);
        size = builder.createOrFold<mlir::arith::DivUIOp>(loc, size, stride);
      }
      return {foldConst(offset), foldConst(size), stride, true};
    } else if (auto literal = valType.dyn_cast<plier::LiteralType>()) {
      auto offset = foldConst(handleNegativeVal(literal.getValue()));
      return {offset, builder.getIndexAttr(1), builder.getIndexAttr(1), false};
    } else*/ {
      auto offset =
          foldConst(handleNegativeVal(imex::indexCast(builder, loc, indexVal)));
      return {offset, builder.getIndexAttr(1), builder.getIndexAttr(1), false};
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

namespace {
struct SetitemOpLowering : public mlir::OpConversionPattern<imex::ntensor::SetitemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::SetitemOp op, imex::ntensor::SetitemOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto target = adaptor.getSource();
    auto targetType = target.getType().dyn_cast<mlir::MemRefType>();
    if (!targetType)
      return mlir::failure();

    auto index = adaptor.getIndex();
    if (!isValidGetitemIndex(index.getType()))
      return mlir::failure();

    auto elemType = targetType.getElementType();
    auto signlessElemType = imex::makeSignlessType(elemType);

    auto value = adaptor.getValue();
    auto loc = op.getLoc();
    llvm::SmallVector<mlir::OpFoldResult> offsets;
    llvm::SmallVector<mlir::OpFoldResult> sizes;
    llvm::SmallVector<mlir::OpFoldResult> strides;
    llvm::SmallVector<unsigned> dimsIndices;
    if (mlir::failed(computeIndices(rewriter, loc, target, index, offsets,
                                    sizes, strides, dimsIndices)))
      return mlir::failure();

    auto castElem = [&](mlir::Value val) -> mlir::Value {
      if (val.getType() != elemType) {
        // TODO
        val = rewriter.createOrFold<plier::CastOp>(loc, elemType, val);
        rerunScfPipeline(op);
      }
      if (elemType != signlessElemType)
        val =
            rewriter.create<imex::util::SignCastOp>(loc, signlessElemType, val);

      return val;
    };

    if (!dimsIndices.empty()) {
      // Is slice
      auto dst = makeSubview(rewriter, loc, target, offsets, sizes, strides,
                             dimsIndices);

      auto castView = [&](mlir::Value val) -> mlir::Value {
        auto viewType = val.getType().cast<mlir::MemRefType>();
        if (viewType.getElementType() != signlessElemType) {
          auto signlessMemref = viewType.clone(signlessElemType);
          val =
              rewriter.create<imex::util::SignCastOp>(loc, signlessMemref, val);
        }
        return val;
      };

      auto valType = value.getType();
      if (auto tensType = valType.dyn_cast<mlir::TensorType>()) {
        auto memrefType = mlir::MemRefType::get(tensType.getShape(),
                                                tensType.getElementType());
        auto src =
            rewriter
                .create<mlir::bufferization::ToMemrefOp>(loc, memrefType, value)
                .getResult();
        genCopy(rewriter, loc, castView(src), dst);
        rewriter.eraseOp(op);
      } else if (valType.isa<mlir::MemRefType>()) {
        auto srcView = castView(value);
        auto dstView = castView(dst);
        genCopy(rewriter, loc, srcView, dstView);
        rewriter.eraseOp(op);
      } else {
        auto elem = castElem(value);
        auto view = castView(dst);
        rewriter.replaceOpWithNewOp<mlir::linalg::FillOp>(op, elem, view);
      }
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

      if (signlessElemType != elemType) {
        auto signlessMemref = targetType.clone(signlessElemType);
        target = rewriter.create<imex::util::SignCastOp>(loc, signlessMemref,
                                                         target);
      }

      rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(
          op, castElem(value), target, toValues(offsets));
    }

    return mlir::success();
  }
};

//struct GetitemOpLowering : public mlir::OpConversionPattern<imex::ntensor::GetitemOp> {
//  using OpConversionPattern::OpConversionPattern;

//  mlir::LogicalResult
//  matchAndRewrite(imex::ntensor::GetitemOp op, imex::ntensor::GetitemOp::Adaptor adaptor,
//                  mlir::ConversionPatternRewriter &rewriter) const override {
//    auto value = adaptor.getValue();
//    auto index = adaptor.getIndex();
//    auto type = value.getType();
//    bool isMemref = type.isa<mlir::MemRefType>();
//    bool isTensor = type.isa<mlir::TensorType>();

//    if (!isMemref && !isTensor)
//      return mlir::failure();

//    if (!isValidGetitemIndex(index.getType()))
//      return mlir::failure();

//    auto loc = op.getLoc();
//    auto shapedType = type.cast<mlir::ShapedType>();
//    llvm::SmallVector<mlir::OpFoldResult> offsets;
//    llvm::SmallVector<mlir::OpFoldResult> sizes;
//    llvm::SmallVector<mlir::OpFoldResult> strides;
//    llvm::SmallVector<unsigned> dimsIndices;
//    if (mlir::failed(computeIndices(rewriter, loc, value, index, offsets, sizes,
//                                    strides, dimsIndices)))
//      return mlir::failure();

//    mlir::Value res;
//    auto elemType = shapedType.getElementType();
//    auto elemTypeSignless = imex::makeSignlessType(elemType);
//    if (elemType != elemTypeSignless) {
//      if (isMemref) {
//        auto memrefType = type.cast<mlir::MemRefType>();
//        auto signlessType = mlir::MemRefType::get(
//            memrefType.getShape(), elemTypeSignless, memrefType.getLayout());
//        value =
//            rewriter.create<imex::util::SignCastOp>(loc, signlessType, value);
//      } else if (isTensor) {
//        auto tensorType = type.cast<mlir::RankedTensorType>();
//        auto signlessType = mlir::RankedTensorType::get(
//            tensorType.getShape(), elemTypeSignless, tensorType.getEncoding());
//        value =
//            rewriter.create<imex::util::SignCastOp>(loc, signlessType, value);
//      } else {
//        llvm_unreachable("Invalid getitem");
//      }
//    }

//    if (!dimsIndices.empty()) {
//      // Is slice
//      res = makeSubview(rewriter, loc, value, offsets, sizes, strides,
//                        dimsIndices);

//      mlir::ShapedType resultTypeSignless =
//          res.getType().cast<mlir::ShapedType>();
//      mlir::Type resultType;
//      if (isMemref) {
//        resultType =
//            mlir::MemRefType::get(resultTypeSignless.getShape(), elemType);
//      } else if (isTensor) {
//        resultType = mlir::RankedTensorType::get(resultTypeSignless.getShape(),
//                                                 elemType);
//      } else {
//        llvm_unreachable("Invalid getitem");
//      }

//      if (resultType != resultTypeSignless)
//        res = rewriter.create<imex::util::SignCastOp>(loc, resultType, res);
//    } else {
//      // Is single element
//      auto toValues = [&](auto &vals) {
//        llvm::SmallVector<mlir::Value> ret(vals.size());
//        for (auto it : llvm::enumerate(vals)) {
//          auto i = it.index();
//          auto val = it.value();
//          if (auto attr = val.template dyn_cast<mlir::Attribute>()) {
//            ret[i] = rewriter.create<mlir::arith::ConstantIndexOp>(
//                loc, attr.template cast<mlir::IntegerAttr>()
//                         .getValue()
//                         .getSExtValue());
//          } else {
//            ret[i] = val.template get<mlir::Value>();
//          }
//        }

//        return ret;
//      };
//      if (isMemref) {
//        res = rewriter.create<mlir::memref::LoadOp>(loc, value,
//                                                    toValues(offsets));
//      } else if (isTensor) {
//        res = rewriter.create<mlir::tensor::ExtractOp>(loc, value,
//                                                       toValues(offsets));
//      } else {
//        llvm_unreachable("Invalid getitem");
//      }

//      if (elemType != elemTypeSignless)
//        res = rewriter.create<imex::util::SignCastOp>(loc, elemType, res);
//    }

//    rerunScfPipeline(op);
//    rewriter.replaceOp(op, res);
//    return mlir::success();
//  }
//};
}


void imex::populateNtensorToMemrefPatterns(mlir::MLIRContext &context,
                                           mlir::TypeConverter &converter,
                                           mlir::RewritePatternSet &patterns,
                                           mlir::ConversionTarget &target)
{
  converter.addConversion([](imex::ntensor::NTensorType type) -> llvm::Optional<mlir::Type>{
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
}

std::unique_ptr<mlir::Pass> imex::createNtensorToMemrefPass()
{
  return std::make_unique<NtensorToMemrefPass>();
}

