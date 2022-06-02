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

#include "mlir-extensions/Transforms/promote_bool_memref.hpp"

#include "mlir-extensions/Dialect/plier_util/dialect.hpp"
#include "mlir-extensions/Transforms/type_conversion.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace {

static bool isI1(mlir::Type type) { return type.isSignlessInteger(1); }

static bool isMemI1(mlir::Type type) {
  if (auto memref = type.dyn_cast<mlir::MemRefType>())
    return isI1(memref.getElementType());

  return false;
}

static llvm::Optional<bool> checkOp(mlir::Operation *op) {
  if (llvm::any_of(op->getOperandTypes(), &isMemI1) ||
      llvm::any_of(op->getResultTypes(), &isMemI1))
    return false;

  return true;
}

class ConvertDimOp : public mlir::OpConversionPattern<mlir::memref::DimOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::DimOp op, mlir::memref::DimOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::DimOp>(op, adaptor.source(),
                                                     adaptor.index());
    return mlir::success();
  }
};

class ConvertLoadOp : public mlir::OpConversionPattern<mlir::memref::LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::memref::LoadOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto res = rewriter.create<mlir::memref::LoadOp>(loc, adaptor.memref(),
                                                     adaptor.indices());
    rewriter.replaceOpWithNewOp<mlir::arith::TruncIOp>(
        op, rewriter.getIntegerType(1), res);
    return mlir::success();
  }
};

class ConvertStoreOp : public mlir::OpConversionPattern<mlir::memref::StoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::memref::StoreOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto val = rewriter.create<mlir::arith::ExtUIOp>(
        loc, rewriter.getIntegerType(8), adaptor.value());
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(
        op, val, adaptor.memref(), adaptor.indices());
    return mlir::success();
  }
};

class ConvertAllocOp : public mlir::OpConversionPattern<mlir::memref::AllocOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::AllocOp op,
                  mlir::memref::AllocOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    auto resType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::MemRefType>();
    if (!resType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::AllocOp>(
        op, resType, adaptor.dynamicSizes(), adaptor.symbolOperands(),
        adaptor.alignmentAttr());
    return mlir::success();
  }
};

class ConvertAllocaOp
    : public mlir::OpConversionPattern<mlir::memref::AllocaOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::AllocaOp op,
                  mlir::memref::AllocaOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    auto resType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::MemRefType>();
    if (!resType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(
        op, resType, adaptor.dynamicSizes(), adaptor.symbolOperands(),
        adaptor.alignmentAttr());
    return mlir::success();
  }
};

class ConvertDeallocOp
    : public mlir::OpConversionPattern<mlir::memref::DeallocOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::DeallocOp op,
                  mlir::memref::DeallocOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::DeallocOp>(op, adaptor.memref());
    return mlir::success();
  }
};

class ConvertCastOp : public mlir::OpConversionPattern<mlir::memref::CastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::CastOp op,
                  mlir::memref::CastOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    auto resType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::MemRefType>();
    if (!resType)
      return mlir::failure();
    rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(op, resType,
                                                      adaptor.source());
    return mlir::success();
  }
};

class ConvertSubviewOp
    : public mlir::OpConversionPattern<mlir::memref::SubViewOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::SubViewOp op,
                  mlir::memref::SubViewOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    auto resType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::MemRefType>();
    if (!resType)
      return mlir::failure();
    rewriter.replaceOpWithNewOp<mlir::memref::SubViewOp>(
        op, resType, adaptor.source(), adaptor.offsets(), adaptor.sizes(),
        adaptor.strides(), adaptor.static_offsets(), adaptor.static_sizes(),
        adaptor.static_strides());
    return mlir::success();
  }
};

class ConvertRetainOp : public mlir::OpConversionPattern<plier::RetainOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::RetainOp op, plier::RetainOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    auto resType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::MemRefType>();
    if (!resType)
      return mlir::failure();
    rewriter.replaceOpWithNewOp<plier::RetainOp>(op, resType, adaptor.source());
    return mlir::success();
  }
};
} // namespace

void plier::populatePromoteBoolMemrefConversionRewritesAndTarget(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {
  auto context = patterns.getContext();
  auto i8 = mlir::IntegerType::get(context, 8);
  typeConverter.addConversion(
      [i8](mlir::MemRefType type) -> llvm::Optional<mlir::Type> {
        auto elemType = type.getElementType();
        if (isI1(elemType))
          return type.clone(i8);

        return llvm::None;
      });

  target.addDynamicallyLegalDialect<mlir::memref::MemRefDialect>(&checkOp);
  target.addDynamicallyLegalOp<plier::RetainOp>(&checkOp);

  patterns.insert<ConvertDimOp, ConvertLoadOp, ConvertStoreOp, ConvertAllocOp,
                  ConvertAllocaOp, ConvertDeallocOp, ConvertCastOp,
                  ConvertSubviewOp, ConvertRetainOp>(typeConverter, context);
}

namespace {
struct PromoteBoolMemrefPass
    : public mlir::PassWrapper<PromoteBoolMemrefPass, mlir::OperationPass<>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithmeticDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto &context = getContext();

    mlir::TypeConverter typeConverter;
    // Convert unknown types to itself
    typeConverter.addConversion([](mlir::Type type) { return type; });

    mlir::RewritePatternSet patterns(&context);
    mlir::ConversionTarget target(context);

    plier::populateTupleTypeConversionRewritesAndTarget(typeConverter, patterns,
                                                        target);
    plier::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                              patterns, target);

    plier::populatePromoteBoolMemrefConversionRewritesAndTarget(
        typeConverter, patterns, target);
    if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                               std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> plier::createPromoteBoolMemrefPass() {
  return std::make_unique<PromoteBoolMemrefPass>();
}
