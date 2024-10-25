//===- MathToVC.cpp - Conversion---------------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements conversion of the select math dialect operations into
/// Func dialect calls to vc-intrinsics functions
///
//===----------------------------------------------------------------------===//

#include "imex/Conversion/MathToVC/MathToVC.h"
#include "imex/Utils/VCUtils.h"
#include "imex/Utils/XeCommon.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"

namespace imex {
#define GEN_PASS_DEF_CONVERTMATHTOVC
#include "imex/Conversion/Passes.h.inc"
} // namespace imex

using namespace mlir;
using namespace imex;

namespace {
//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Get the VC intrinsic name for the given math operation
template <typename MOp> std::string getVCIntrinsicName() {
  constexpr bool isCeilOp = std::is_same_v<MOp, math::CeilOp>;
  constexpr bool isFloorOp = std::is_same_v<MOp, math::FloorOp>;
  constexpr bool isExpOp = std::is_same_v<MOp, math::ExpOp>;
  constexpr bool isExp2Op = std::is_same_v<MOp, math::Exp2Op>;
  if (isCeilOp)
    return "llvm.genx.rndu.";
  else if (isFloorOp)
    return "llvm.genx.rndd.";
  else if (isExpOp || isExp2Op)
    return "llvm.genx.exp.";
  else
    assert(0 && "Unsupported math Op. Add more support!");
}

// Utility function to convert a scalar or vector type of any float bitwidth to
// another.
Type convertScalarOrVectorFloatType(Type srcType, Type dstElementType) {
  // get a vector type or scalar type of dstElementType with the same shape as
  // srcType
  if (auto vecTy = dyn_cast<VectorType>(srcType)) {
    auto newTy = VectorType::get(vecTy.getShape(), dstElementType);
    return newTy;
  } else if (auto scalarTy = dyn_cast<FloatType>(srcType)) {
    return dstElementType;
  } else {
    assert(0 && "Unsupported type");
  }
}

// Utility function to convert a range float args to a specific float type
// The function converts the float args to the dstElementType
// It generates an extension or truncation operation if the bitwidth of the src
// and dst types are different
SmallVector<Value> convertFloatArgsType(SmallVector<Value> args,
                                        Type dstElementType,
                                        ConversionPatternRewriter &rewriter) {
  SmallVector<Value> newArgs;
  auto dstBitWidth = dstElementType.getIntOrFloatBitWidth();
  for (auto arg : args) {
    // Assert if src and dst types are not float types
    assert(((isa<FloatType>(arg.getType()) ||
             isa<FloatType>(
                 dyn_cast<VectorType>(arg.getType()).getElementType())) &&
            isa<FloatType>(dstElementType)) &&
           "Unsupported type, src and dst both should be float types");
    auto srcBitWidth =
        dyn_cast<VectorType>(arg.getType())
            ? dyn_cast<VectorType>(arg.getType()).getElementTypeBitWidth()
            : arg.getType().getIntOrFloatBitWidth();

    if (srcBitWidth == dstBitWidth)
      newArgs.push_back(arg);
    else if (srcBitWidth < dstBitWidth) {
      auto newType =
          convertScalarOrVectorFloatType(arg.getType(), dstElementType);
      auto newOp = rewriter.create<arith::ExtFOp>(arg.getLoc(), newType, arg);
      newArgs.push_back(newOp);
    } else if (srcBitWidth > dstBitWidth) {
      auto newType =
          convertScalarOrVectorFloatType(arg.getType(), dstElementType);
      auto newOp = rewriter.create<arith::TruncFOp>(arg.getLoc(), newType, arg);
      newArgs.push_back(newOp);
    }
  }
  return newArgs;
}
//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

// Elementwise math to vc-intrinsics conversion pattern for ops that only
// supports f32
template <typename MOp>
struct ElementwiseFloatOnlyMathOpPattern final
    : public OpConversionPattern<MOp> {
  using OpConversionPattern<MOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MOp op, typename MOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type opElementType;
    // Check if the result type is a 1D vector
    if (auto vecTy = dyn_cast<VectorType>(op.getType())) {
      if (vecTy.getRank() != 1)
        return failure();
      opElementType = vecTy.getElementType();
    } else {
      opElementType = op.getType();
    }
    auto loc = op.getLoc();
    auto args = adaptor.getOperands();
    // Upconvert or downconvert all the operands' element types to f32
    // Warning message here for the truncation. If we are truncating
    // the value, the result can be different from the original value.
    if (opElementType.getIntOrFloatBitWidth() > 32)
      emitWarning(op.getLoc(), "Truncation is done on input during conversion, "
                               "may result in wrong result.\n");
    llvm::SmallVector<Value> newArgs =
        convertFloatArgsType(args, rewriter.getF32Type(), rewriter);

    // Result element type is always f32
    auto newType =
        convertScalarOrVectorFloatType(op.getType(), rewriter.getF32Type());
    std::string resStr = "f32";
    resStr.insert(
        0, ((dyn_cast<VectorType>(newType))
                ? llvm::formatv("v{0}",
                                dyn_cast<VectorType>(newType).getNumElements())
                      .str()
                : ""));

    // for large vectors, generate the corresponding VC intrinsic.
    auto funcName = getVCIntrinsicName<MOp>();
    funcName += resStr;
    auto callOp =
        createFuncCall(rewriter, loc, funcName, {newType}, newArgs, false);

    // Initialize a smallvector with the callOp
    SmallVector<Value> callOpResult;
    callOpResult.push_back(callOp.getResult(0));

    // Convert the result of the call to the original type
    auto originalResultType =
        convertFloatArgsType(callOpResult, opElementType, rewriter);

    rewriter.replaceOp(op, originalResultType);
    return success();
  }
};

// ExpOp conversion pattern, supports both math::exp and math::exp2
template <typename ExpOp>
struct ExpOpPattern final : public OpConversionPattern<ExpOp> {
  ExpOpPattern(MLIRContext *ctx, bool enableHighPrecisionInterimCalculation)
      : OpConversionPattern<ExpOp>(ctx),
        enableHighPrecisionInterimCalculation(
            enableHighPrecisionInterimCalculation) {}
  LogicalResult
  matchAndRewrite(ExpOp op, typename ExpOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vecTy = dyn_cast<VectorType>(op.getType());

    // Only deal with Exp op with 1-D vector type
    if (!(vecTy && vecTy.getRank() == 1))
      return failure();

    auto loc = op.getLoc();

    // "llvm.genx.exp" returns the base 2 exponentiation of the input.
    // To get the base e exponentiation, we need to scale the input by log2(e)
    bool isExpOp = std::is_same_v<ExpOp, math::ExpOp>;
    auto operands = adaptor.getOperands();
    SmallVector<Value> args{operands};
    // Create a constant vector with the value of 1.442695040888963
    if (isExpOp) {
      // Create the intermediate instructions of f32 vector type if the element
      // type is less than 32 bits
      if (this->enableHighPrecisionInterimCalculation &&
          vecTy.getElementType().getIntOrFloatBitWidth() < 32) {
        auto interimVectorType =
            VectorType::get(vecTy.getShape(), rewriter.getF32Type());
        auto vecAttr = DenseElementsAttr::get(
            interimVectorType,
            rewriter.getFloatAttr(interimVectorType.getElementType(),
                                  1.442695040888963));
        auto log2eConstVec =
            rewriter.create<arith::ConstantOp>(loc, interimVectorType, vecAttr);
        auto input = convertFloatArgsType({operands[0]}, rewriter.getF32Type(),
                                          rewriter);
        auto scaledInputf32 = rewriter.create<arith::MulFOp>(
            op.getLoc(), input[0], log2eConstVec);
        auto scaledInput = convertFloatArgsType(
            {scaledInputf32}, vecTy.getElementType(), rewriter);
        args.clear();
        args.push_back(scaledInput[0]);
      } else {
        auto vecAttr = DenseElementsAttr::get(
            vecTy,
            rewriter.getFloatAttr(vecTy.getElementType(), 1.442695040888963));
        auto log2eConstVec =
            rewriter.create<arith::ConstantOp>(loc, vecTy, vecAttr);
        auto input = operands[0];
        auto scaledInput =
            rewriter.create<arith::MulFOp>(op.getLoc(), input, log2eConstVec);
        args.clear();
        args.push_back(scaledInput);
      }
    }
    // for large vectors, generate the corresponding VC intrinsic.
    auto funcName = getVCIntrinsicName<ExpOp>();
    funcName += encodeVectorType(rewriter, vecTy).first;
    auto callOp =
        createFuncCall(rewriter, loc, funcName, {op.getType()}, args, false);
    rewriter.replaceOp(op, callOp);
    return success();
  }

private:
  const bool enableHighPrecisionInterimCalculation;
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void imex::populateMathToVCPatterns(
    ::mlir::LLVMTypeConverter &typeConverter,
    ::mlir::RewritePatternSet &patterns,
    bool enableHighPrecisionInterimCalculation) {
  // Add patterns
  patterns.add<ElementwiseFloatOnlyMathOpPattern<math::CeilOp>,
               ElementwiseFloatOnlyMathOpPattern<math::FloorOp>>(
      patterns.getContext());
  patterns.add<ExpOpPattern<math::ExpOp>, ExpOpPattern<math::Exp2Op>>(
      patterns.getContext(), enableHighPrecisionInterimCalculation);
}

//===----------------------------------------------------------------------===//
// Conversion Legality configuration
//===----------------------------------------------------------------------===//
void imex::configureMathToVCConversionLegality(
    ::mlir::ConversionTarget &target) {
  // Add legal dialects
  target.addLegalDialect<func::FuncDialect, arith::ArithDialect>();
  // math.exp and math.exp2 is only converted if they are 1D vectors
  target.addDynamicallyLegalOp<math::ExpOp, math::Exp2Op>([&](Operation *op) {
    if (auto vecTy = dyn_cast<VectorType>(op->getResult(0).getType())) {
      if (vecTy.getRank() != 1)
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
struct MathToVCPass : public imex::impl::ConvertMathToVCBase<MathToVCPass> {
  using Base::Base;
  MathToVCPass(bool emitDeallocs)
      : imex::impl::ConvertMathToVCBase<MathToVCPass>() {
    this->enableHighPrecisionInterimCalculation.setValue(emitDeallocs);
  }
  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();
    LLVMTypeConverter typeConverter(&getContext());
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    // Add patterns
    imex::populateMathToVCPatterns(
        typeConverter, patterns,
        this->enableHighPrecisionInterimCalculation.getValue());
    configureMathToVCConversionLegality(target);

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
imex::createConvertMathToVCPass() {
  return std::make_unique<MathToVCPass>();
}
