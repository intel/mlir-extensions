//===- ArithToVC.cpp - Conversion---------------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements conversion of the select arith dialect operations into
/// Func dialect calls to vc-intrinsics functions
///
//===----------------------------------------------------------------------===//

#include "imex/Conversion/ArithToVC/ArithToVC.h"
#include "imex/Utils/VCUtils.h"
#include "imex/Utils/XeCommon.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"

namespace imex {
#define GEN_PASS_DEF_CONVERTARITHTOVC
#include "imex/Conversion/Passes.h.inc"
} // namespace imex

using namespace mlir;
using namespace imex;

namespace {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Get the VC intrinsic name for the given arith operation
template <typename MOp> std::string getVCIntrinsicName() {
  constexpr bool isFMaxOp = std::is_same_v<MOp, arith::MaximumFOp>;
  if (isFMaxOp)
    return "llvm.genx.fmax.";
  else
    assert(0 && "Unsupported arith Op. Add more support!");
}

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

// Elementwise arith to vc-intrinsics conversion pattern for ops that only
// supports f32
template <typename MOp>
struct ElementwiseArithOpPattern final : public OpConversionPattern<MOp> {
  using OpConversionPattern<MOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MOp op, typename MOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if the result type is a 1D vector
    auto *converter = this->getTypeConverter();
    auto vecTy = dyn_cast<VectorType>(converter->convertType(op.getType()));
    if (!vecTy || vecTy.getRank() != 1)
      return failure();

    auto loc = op.getLoc();
    auto args = adaptor.getOperands();

    bool isVectorAnyINTELType = imex::isVectorAnyINTELType(vecTy);
    bool isFastmath =
        (op.getFastmathAttr().getValue() != arith::FastMathFlags::none);
    if (!isVectorAnyINTELType && !isFastmath)
      return failure();
    // for large vectors, generate the corresponding VC intrinsic.
    auto funcName = getVCIntrinsicName<MOp>();
    funcName += encodeVectorType(rewriter, vecTy).first;
    auto callOp = createFuncCall(rewriter, loc, funcName, {vecTy}, args, false);
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void imex::populateArithToVCPatterns(
    ::mlir::TypeConverter &typeConverter, ::mlir::RewritePatternSet &patterns,
    bool enableHighPrecisionInterimCalculation) {
  // Add patterns
  patterns.add<ElementwiseArithOpPattern<arith::MaximumFOp>>(
      typeConverter, patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Conversion Legality configuration
//===----------------------------------------------------------------------===//
void imex::configureArithToVCConversionLegality(
    ::mlir::ConversionTarget &target) {
  // Add legal dialects
  target.addLegalDialect<func::FuncDialect, arith::ArithDialect>();
  // arith.maximumf is converted if they are vectors
  target.addDynamicallyLegalOp<arith::MaximumFOp>([&](arith::MaximumFOp op) {
    if (auto vecTy = dyn_cast<VectorType>(op.getType())) {
      vecTy = VectorType::get(vecTy.getNumElements(), vecTy.getElementType());
      bool isVectorAnyINTELType = imex::isVectorAnyINTELType(vecTy);
      bool isFastmath =
          (op.getFastmathAttr().getValue() != arith::FastMathFlags::none);
      if (!isVectorAnyINTELType && !isFastmath)
        return true;
      return false;
    }
    return true;
  });
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ArithToVCPass : public imex::impl::ConvertArithToVCBase<ArithToVCPass> {
  using Base::Base;
  ArithToVCPass(bool enableHPIC)
      : imex::impl::ConvertArithToVCBase<ArithToVCPass>() {
    this->enableHighPrecisionInterimCalculation.setValue(enableHPIC);
  }
  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::Type type) { return type; });
    typeConverter.addConversion([](mlir::VectorType type) {
      if (!mlir::vector::isLinearizableVector(type))
        return type;
      return mlir::VectorType::get(type.getNumElements(), type.getElementType(),
                                   type.isScalable());
    });
    auto materializeCast = [&](mlir::OpBuilder &builder, mlir::Type type,
                               mlir::ValueRange inputs,
                               mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1 || !mlir::isa<mlir::VectorType>(type) ||
          !mlir::isa<mlir::VectorType>(inputs.front().getType()))
        return nullptr;

      return builder.create<mlir::vector::ShapeCastOp>(loc, type,
                                                       inputs.front());
    };
    typeConverter.addArgumentMaterialization(materializeCast);
    typeConverter.addSourceMaterialization(materializeCast);
    typeConverter.addTargetMaterialization(materializeCast);

    // Add patterns
    imex::populateArithToVCPatterns(
        typeConverter, patterns,
        this->enableHighPrecisionInterimCalculation.getValue());
    configureArithToVCConversionLegality(target);

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
imex::createConvertArithToVCPass() {
  return std::make_unique<ArithToVCPass>();
}
