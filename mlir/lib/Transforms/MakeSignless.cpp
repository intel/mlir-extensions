// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/MakeSignless.hpp"

#include "imex/Dialect/imex_util/Dialect.hpp"
#include "imex/Transforms/TypeConversion.hpp"

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace {
struct ConvertChangeLayout
    : public mlir::OpConversionPattern<imex::util::ChangeLayoutOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::ChangeLayoutOp op,
                  imex::util::ChangeLayoutOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter);

    auto oldResType = op.getType();
    auto newResType =
        converter->convertType(oldResType).dyn_cast_or_null<mlir::MemRefType>();
    if (!newResType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<imex::util::ChangeLayoutOp>(
        op, newResType, adaptor.getSource());
    return mlir::success();
  }
};

struct ConvertToMemref
    : public mlir::OpConversionPattern<mlir::bufferization::ToMemrefOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::bufferization::ToMemrefOp op,
                  mlir::bufferization::ToMemrefOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter);

    auto oldResType = op.getType();
    auto newResType =
        converter->convertType(oldResType).dyn_cast_or_null<mlir::MemRefType>();
    if (!newResType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::bufferization::ToMemrefOp>(
        op, newResType, adaptor.getTensor());
    return mlir::success();
  }
};

struct ConvertToTensor
    : public mlir::OpConversionPattern<mlir::bufferization::ToTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::bufferization::ToTensorOp op,
                  mlir::bufferization::ToTensorOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter);

    auto oldResType = op.getType();
    auto newResType = converter->convertType(oldResType)
                          .dyn_cast_or_null<mlir::RankedTensorType>();
    if (!newResType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::bufferization::ToTensorOp>(
        op, newResType, adaptor.getMemref());
    return mlir::success();
  }
};

template <typename Op>
struct ConvertAlloc : public mlir::OpConversionPattern<Op> {
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = this->getTypeConverter();
    assert(converter);

    auto oldResType = op.getType();
    auto newResType = converter->convertType(oldResType)
                          .template dyn_cast_or_null<mlir::MemRefType>();
    if (!newResType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<Op>(op, newResType, adaptor.getDynamicSizes(),
                                    adaptor.getSymbolOperands(),
                                    adaptor.getAlignmentAttr());
    return mlir::success();
  }
};

struct ConvertDealloc
    : public mlir::OpConversionPattern<mlir::memref::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::DeallocOp op,
                  mlir::memref::DeallocOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::DeallocOp>(op,
                                                         adaptor.getMemref());
    return mlir::success();
  }
};

struct ConvertSubview
    : public mlir::OpConversionPattern<mlir::memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::SubViewOp op,
                  mlir::memref::SubViewOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter);

    auto oldResType = op.getType();
    auto newResType =
        converter->convertType(oldResType).dyn_cast_or_null<mlir::MemRefType>();
    if (!newResType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::SubViewOp>(
        op, newResType, adaptor.getSource(), adaptor.getOffsets(),
        adaptor.getSizes(), adaptor.getStrides(),
        adaptor.getStaticOffsetsAttr(), adaptor.getStaticSizesAttr(),
        adaptor.getStaticStridesAttr());
    return mlir::success();
  }
};

struct ConvertTensorEmpty
    : public mlir::OpConversionPattern<mlir::tensor::EmptyOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::EmptyOp op,
                  mlir::tensor::EmptyOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = this->getTypeConverter();
    assert(converter);

    auto oldResType = op.getType();
    auto newResType = converter->convertType(oldResType)
                          .dyn_cast_or_null<mlir::RankedTensorType>();
    if (!newResType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::tensor::EmptyOp>(
        op, newResType, adaptor.getDynamicSizes());
    return mlir::success();
  }
};

struct ConvertTensorFromElements
    : public mlir::OpConversionPattern<mlir::tensor::FromElementsOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::FromElementsOp op,
                  mlir::tensor::FromElementsOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = this->getTypeConverter();
    assert(converter);

    auto oldResType = op.getType();
    auto newResType = converter->convertType(oldResType)
                          .dyn_cast_or_null<mlir::RankedTensorType>();
    if (!newResType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::tensor::FromElementsOp>(
        op, newResType, adaptor.getElements());
    return mlir::success();
  }
};

struct ConvertTensorExpandShape
    : public mlir::OpConversionPattern<mlir::tensor::ExpandShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ExpandShapeOp op,
                  mlir::tensor::ExpandShapeOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = this->getTypeConverter();
    assert(converter);

    auto oldResType = op.getType();
    auto newResType = converter->convertType(oldResType)
                          .dyn_cast_or_null<mlir::RankedTensorType>();
    if (!newResType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::tensor::ExpandShapeOp>(
        op, newResType, adaptor.getSrc(), adaptor.getReassociation());
    return mlir::success();
  }
};

struct ConvertTensorReshape
    : public mlir::OpConversionPattern<mlir::tensor::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ReshapeOp op,
                  mlir::tensor::ReshapeOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = this->getTypeConverter();
    assert(converter);

    auto oldResType = op.getType();
    auto newResType = converter->convertType(oldResType)
                          .dyn_cast_or_null<mlir::RankedTensorType>();
    if (!newResType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::tensor::ReshapeOp>(
        op, newResType, adaptor.getSource(), adaptor.getShape());
    return mlir::success();
  }
};

struct ConvertTensorExtractSlice
    : public mlir::OpConversionPattern<mlir::tensor::ExtractSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ExtractSliceOp op,
                  mlir::tensor::ExtractSliceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter);

    auto oldResType = op.getType();
    auto newResType = converter->convertType(oldResType)
                          .dyn_cast_or_null<mlir::RankedTensorType>();
    if (!newResType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::tensor::ExtractSliceOp>(
        op, newResType, adaptor.getSource(), adaptor.getOffsets(),
        adaptor.getSizes(), adaptor.getStrides(),
        adaptor.getStaticOffsetsAttr(), adaptor.getStaticSizesAttr(),
        adaptor.getStaticStridesAttr());
    return mlir::success();
  }
};

struct ConvertTensorInserSlice
    : public mlir::OpConversionPattern<mlir::tensor::InsertSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::InsertSliceOp op,
                  mlir::tensor::InsertSliceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto offsets = mlir::getMixedValues(adaptor.getStaticOffsets(),
                                        adaptor.getOffsets(), rewriter);
    auto sizes = mlir::getMixedValues(adaptor.getStaticSizes(),
                                      adaptor.getSizes(), rewriter);
    auto strides = mlir::getMixedValues(adaptor.getStaticStrides(),
                                        adaptor.getStrides(), rewriter);
    rewriter.replaceOpWithNewOp<mlir::tensor::InsertSliceOp>(
        op, adaptor.getSource(), adaptor.getDest(), offsets, sizes, strides);
    return mlir::success();
  }
};

struct ConvertLinalgFill
    : public mlir::OpConversionPattern<mlir::linalg::FillOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::FillOp op,
                  mlir::linalg::FillOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = this->getTypeConverter();
    assert(converter);

    llvm::SmallVector<mlir::Type> results;
    if (mlir::failed(converter->convertTypes(op.getResultTypes(), results)))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::linalg::FillOp>(
        op, results, adaptor.getInputs(), adaptor.getOutputs());
    return mlir::success();
  }
};

struct ConvertLinalgGeneric
    : public mlir::OpConversionPattern<mlir::linalg::GenericOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::GenericOp op,
                  mlir::linalg::GenericOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = this->getTypeConverter();
    assert(converter);

    llvm::SmallVector<mlir::Type> results;
    if (mlir::failed(converter->convertTypes(op.getResultTypes(), results)))
      return mlir::failure();

    if (mlir::failed(rewriter.convertRegionTypes(&op.getRegion(), *converter)))
      return mlir::failure();

    auto inputs = adaptor.getInputs();
    auto outputs = adaptor.getOutputs();
    auto maps = adaptor.getIndexingMaps();
    auto iterators = adaptor.getIteratorTypes();
    auto doc = adaptor.getDocAttr();
    auto libCall = adaptor.getLibraryCallAttr();

    auto loc = op->getLoc();
    auto res = rewriter.create<mlir::linalg::GenericOp>(
        loc, results, inputs, outputs, maps, iterators, doc, libCall);
    mlir::Region &newRegion = res.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), newRegion, newRegion.end());
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

struct ConvertLinalgYield
    : public mlir::OpConversionPattern<mlir::linalg::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::YieldOp op,
                  mlir::linalg::YieldOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto parent = op->getParentOp();
    if (!mlir::isa<mlir::linalg::GenericOp>(parent))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::linalg::YieldOp>(op, adaptor.getValues());
    return mlir::success();
  }
};
} // namespace

static llvm::Optional<mlir::Type> makeSignlessType(mlir::Type type) {
  if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    if (intType.getSignedness() != mlir::IntegerType::Signless)
      return mlir::IntegerType::get(type.getContext(), intType.getWidth());
  } else if (auto shapedType = type.dyn_cast<mlir::ShapedType>()) {
    if (auto signlessElem = makeSignlessType(shapedType.getElementType()))
      return shapedType.clone(*signlessElem);
  }

  return std::nullopt;
}

void imex::populateMakeSignlessRewritesAndTarget(
    mlir::TypeConverter &converter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {
  converter.addConversion(&makeSignlessType);

  auto materializeSignCast = [](mlir::OpBuilder &builder, mlir::Type type,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
    assert(inputs.size() == 1);
    return builder.create<imex::util::SignCastOp>(loc, type, inputs.front());
  };
  converter.addArgumentMaterialization(materializeSignCast);
  converter.addSourceMaterialization(materializeSignCast);
  converter.addTargetMaterialization(materializeSignCast);

  target.addDynamicallyLegalOp<
      imex::util::ChangeLayoutOp, mlir::bufferization::ToMemrefOp,
      mlir::bufferization::ToTensorOp, mlir::memref::AllocOp,
      mlir::memref::AllocaOp, mlir::memref::DeallocOp, mlir::memref::SubViewOp,
      mlir::tensor::EmptyOp, mlir::tensor::FromElementsOp,
      mlir::tensor::ExpandShapeOp, mlir::tensor::ReshapeOp,
      mlir::tensor::ExtractSliceOp, mlir::tensor::InsertSliceOp,
      mlir::linalg::FillOp, mlir::linalg::GenericOp, mlir::linalg::YieldOp>(
      [&converter](mlir::Operation *op) { return converter.isLegal(op); });

  patterns.insert<ConvertChangeLayout, ConvertToMemref, ConvertToTensor,
                  ConvertAlloc<mlir::memref::AllocOp>,
                  ConvertAlloc<mlir::memref::AllocaOp>, ConvertDealloc,
                  ConvertSubview, ConvertTensorEmpty, ConvertTensorFromElements,
                  ConvertTensorExpandShape, ConvertTensorReshape,
                  ConvertTensorExtractSlice, ConvertTensorInserSlice,
                  ConvertLinalgFill, ConvertLinalgGeneric, ConvertLinalgYield>(
      converter, patterns.getContext());
}

namespace {
struct MakeSignlessPass
    : public mlir::PassWrapper<MakeSignlessPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MakeSignlessPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::util::ImexUtilDialect>();
    registry.insert<mlir::bufferization::BufferizationDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
    mlir::TypeConverter converter;
    mlir::RewritePatternSet patterns(&context);
    mlir::ConversionTarget target(context);

    // Convert unknown types to itself
    converter.addConversion([](mlir::Type type) { return type; });

    imex::populateTupleTypeConverter(converter);

    imex::populateMakeSignlessRewritesAndTarget(converter, patterns, target);

    imex::populateTupleTypeConversionRewritesAndTarget(converter, patterns,
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

std::unique_ptr<mlir::Pass> imex::createMakeSignlessPass() {
  return std::make_unique<MakeSignlessPass>();
}
