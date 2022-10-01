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

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

template <typename F>
static mlir::ValueRange wrapEnvRegion(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::Attribute env,
                                      mlir::TypeRange results, F &&func) {
  if (!env) {
    mlir::ValueRange res = func(builder, loc);
    assert(res.getTypes() == results && "Invalid result types");
    return res;
  }

  auto bodyBuilder = [&](mlir::OpBuilder &b, mlir::Location l) {
    mlir::ValueRange res = func(b, l);
    assert(res.getTypes() == results && "Invalid result types");
    b.create<imex::util::EnvironmentRegionYieldOp>(l, res);
  };

  auto region = builder.create<imex::util::EnvironmentRegionOp>(
      loc, env, /*args*/ llvm::None, results, bodyBuilder);
  return region.getResults();
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

    auto results = wrapEnvRegion(
        rewriter, op->getLoc(), origType.getEnvironment(),
        rewriter.getIndexType(),
        [&](mlir::OpBuilder &builder, mlir::Location loc) {
          return builder
              .create<mlir::memref::DimOp>(loc, src, adaptor.getIndex())
              .getResult();
        });

    rewriter.replaceOp(op, results);
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

  patterns.insert<DimOpLowering>(converter, &context);

  target.addIllegalOp<imex::ntensor::DimOp>();
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
