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
#include "imex/Transforms/type_conversion.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

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
struct DimOpLowering : public mlir::OpConversionPattern<imex::ntensor::DimOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::DimOp op,
                  imex::ntensor::DimOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto origType = op.getSource().getType().cast<imex::ntensor::NTensorType>();
    auto src = adaptor.getSource();
    if (!src.getType().isa<mlir::MemRefType>())
      return mlir::failure();

    auto indexType = rewriter.getIndexType();
    auto results = wrapEnvRegion(
        rewriter, op->getLoc(), origType.getEnvironment(), indexType,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          return builder
              .create<mlir::memref::DimOp>(loc, src, adaptor.getIndex())
              .getResult();
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct SubviewOpLowering
    : public mlir::OpConversionPattern<imex::ntensor::SubviewOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::SubviewOp op,
                  imex::ntensor::SubviewOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto origType = op.getSource().getType().cast<imex::ntensor::NTensorType>();
    auto src = adaptor.getSource();
    auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
    if (!srcType)
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto dstType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::MemRefType>();
    if (!dstType)
      return mlir::failure();

    auto results = wrapEnvRegion(
        rewriter, op->getLoc(), origType.getEnvironment(), dstType,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          auto offsets = mlir::getMixedStridesOrOffsets(
              adaptor.getStaticOffsets(), adaptor.getOffsets());
          auto sizes =
              mlir::getMixedSizes(adaptor.getStaticSizes(), adaptor.getSizes());
          auto strides = mlir::getMixedStridesOrOffsets(
              adaptor.getStaticStrides(), adaptor.getStrides());

          auto resType =
              mlir::memref::SubViewOp::inferRankReducedResultType(
                  dstType.getShape(), srcType, offsets, sizes, strides)
                  .cast<mlir::MemRefType>();

          mlir::Value res = builder.create<mlir::memref::SubViewOp>(
              loc, resType, src, offsets, sizes, strides);

          if (resType != dstType)
            res = builder.create<imex::util::ChangeLayoutOp>(loc, dstType, res);

          return res;
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct LoadOpLowering
    : public mlir::OpConversionPattern<imex::ntensor::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::LoadOp op,
                  imex::ntensor::LoadOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto origType = op.getArray().getType().cast<imex::ntensor::NTensorType>();
    auto src = adaptor.getArray();
    if (!src.getType().isa<mlir::MemRefType>())
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto dstType = converter->convertType(op.getType());
    if (!dstType || dstType != origType.getElementType())
      return mlir::failure();

    auto results = wrapEnvRegion(
        rewriter, op->getLoc(), origType.getEnvironment(), dstType,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          return builder
              .create<mlir::memref::LoadOp>(loc, src, adaptor.getIndices())
              .getResult();
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct StoreOpLowering
    : public mlir::OpConversionPattern<imex::ntensor::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::StoreOp op,
                  imex::ntensor::StoreOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto origType = op.getArray().getType().cast<imex::ntensor::NTensorType>();
    auto src = adaptor.getArray();
    if (!src.getType().isa<mlir::MemRefType>())
      return mlir::failure();

    auto results = wrapEnvRegion(
        rewriter, op->getLoc(), origType.getEnvironment(), llvm::None,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          auto val = adaptor.getValue();
          builder.create<mlir::memref::StoreOp>(loc, val, src,
                                                adaptor.getIndices());
          return llvm::None;
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ToTensorOpLowering
    : public mlir::OpConversionPattern<imex::ntensor::ToTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::ToTensorOp op,
                  imex::ntensor::ToTensorOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto array = adaptor.getArray();
    if (!array.getType().isa<mlir::MemRefType>())
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto retType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::TensorType>();
    if (!retType)
      return mlir::failure();

    auto origType = op.getArray().getType().cast<imex::ntensor::NTensorType>();
    auto results = wrapEnvRegion(
        rewriter, op->getLoc(), origType.getEnvironment(), retType,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          return builder
              .create<mlir::bufferization::ToTensorOp>(loc, retType, array)
              .getResult();
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct FromTensorOpLowering
    : public mlir::OpConversionPattern<imex::ntensor::FromTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::FromTensorOp op,
                  imex::ntensor::FromTensorOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto tensor = adaptor.getTensor();
    if (!tensor.getType().isa<mlir::RankedTensorType>())
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto origType = op.getType().cast<imex::ntensor::NTensorType>();
    auto retType =
        converter->convertType(origType).dyn_cast_or_null<mlir::MemRefType>();
    if (!retType)
      return mlir::failure();

    auto results = wrapEnvRegion(
        rewriter, op->getLoc(), origType.getEnvironment(), retType,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          return builder
              .create<mlir::bufferization::ToMemrefOp>(loc, retType, tensor)
              .getResult();
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ToMemrefOpLowering
    : public mlir::OpConversionPattern<imex::ntensor::ToMemrefOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::ToMemrefOp op,
                  imex::ntensor::ToMemrefOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getArray();
    auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
    if (!srcType)
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto retType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::MemRefType>();
    if (!retType)
      return mlir::failure();

    if (srcType != retType)
      return mlir::failure();

    rewriter.replaceOp(op, src);
    return mlir::success();
  }
};

struct FromMemrefOpLowering
    : public mlir::OpConversionPattern<imex::ntensor::FromMemrefOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::FromMemrefOp op,
                  imex::ntensor::FromMemrefOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getMemref();
    auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
    if (!srcType)
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto retType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::MemRefType>();
    if (!retType)
      return mlir::failure();

    if (srcType != retType)
      return mlir::failure();

    rewriter.replaceOp(op, src);
    return mlir::success();
  }
};

struct CastOpLowering
    : public mlir::OpConversionPattern<imex::ntensor::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::CastOp op,
                  imex::ntensor::CastOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSource();
    auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
    if (!srcType)
      return mlir::failure();

    auto origSrcType =
        op.getSource().getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!origSrcType)
      return mlir::failure();

    auto origDstType = op.getType().dyn_cast<imex::ntensor::NTensorType>();
    if (!origDstType)
      return mlir::failure();

    if (origSrcType.getEnvironment() != origDstType.getEnvironment())
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto retType = converter->convertType(origDstType)
                       .dyn_cast_or_null<mlir::MemRefType>();

    if (!retType)
      return mlir::failure();

    if (!mlir::memref::CastOp::areCastCompatible(srcType, retType))
      return mlir::failure();

    auto results = wrapEnvRegion(
        rewriter, op->getLoc(), origSrcType.getEnvironment(), retType,
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          return builder.create<mlir::memref::CastOp>(loc, retType, src)
              .getResult();
        });

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct CopyOpLowering
    : public mlir::OpConversionPattern<imex::ntensor::CopyOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::CopyOp op,
                  imex::ntensor::CopyOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSource();
    if (!src.getType().isa<mlir::MemRefType>())
      return mlir::failure();

    auto dst = adaptor.getTarget();
    if (!dst.getType().isa<mlir::MemRefType>())
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::CopyOp>(op, src, dst);
    return mlir::success();
  }
};
} // namespace

void imex::populateNtensorToMemrefRewritesAndTarget(
    mlir::MLIRContext &context, mlir::TypeConverter &converter,
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target) {
  converter.addConversion(
      [](imex::ntensor::NTensorType type) -> llvm::Optional<mlir::Type> {
        auto elemType = type.getElementType();
        if (mlir::MemRefType::isValidElementType(elemType))
          return mlir::MemRefType::get(type.getShape(), elemType);

        return llvm::None;
      });

  patterns
      .insert<DimOpLowering, SubviewOpLowering, LoadOpLowering, StoreOpLowering,
              ToTensorOpLowering, FromTensorOpLowering, ToMemrefOpLowering,
              FromMemrefOpLowering, CastOpLowering, CopyOpLowering>(converter,
                                                                    &context);

  target.addIllegalOp<imex::ntensor::DimOp, imex::ntensor::SubviewOp,
                      imex::ntensor::LoadOp, imex::ntensor::StoreOp,
                      imex::ntensor::ToTensorOp, imex::ntensor::FromTensorOp,
                      imex::ntensor::ToMemrefOp, imex::ntensor::FromMemrefOp,
                      imex::ntensor::CastOp, imex::ntensor::CopyOp>();
}

namespace {
struct NtensorToMemrefPass
    : public mlir::PassWrapper<NtensorToMemrefPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NtensorToMemrefPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::util::ImexUtilDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::bufferization::BufferizationDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
    mlir::TypeConverter converter;
    mlir::RewritePatternSet patterns(&context);
    mlir::ConversionTarget target(context);

    // Convert unknown types to itself
    converter.addConversion([](mlir::Type type) { return type; });

    imex::populateNtensorToMemrefRewritesAndTarget(context, converter, patterns,
                                                   target);
    imex::populateControlFlowTypeConversionRewritesAndTarget(converter,
                                                             patterns, target);

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
