//===- XeTileToXeGPU.cpp - XeTileToXeGPU conversion  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the XeTileToXeGPU conversion, converting the XeTile
/// dialect to the XeGPU dialect.
///
//===----------------------------------------------------------------------===//

#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/Passes.h>

#include "ArithOpConversion.h"
#include "SCFOpConversion.h"
#include "XeTileOpConversion.h"
#include "imex/Conversion/XeTileToXeGPU/XeTileToXeGPU.h"
#include "imex/Utils/XeArch.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace imex {
#define GEN_PASS_DEF_CONVERTXETILETOXEGPU
#include "imex/Conversion/Passes.h.inc"
} // namespace imex

#include <memory>
namespace imex {

class XeTileConversionTarget : public mlir::ConversionTarget {
public:
  explicit XeTileConversionTarget(mlir::MLIRContext &context,
                                  std::shared_ptr<XeuArchInterface> ptruArch)
      : mlir::ConversionTarget(context) {

    this->uArchInterface = ptruArch;
    addIllegalOp<imex::xetile::InitTileOp>();

    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addLegalOp<mlir::vector::ExtractOp>();
    addLegalOp<mlir::vector::ExtractElementOp>();
    addLegalOp<mlir::vector::ExtractStridedSliceOp>();
    addLegalOp<mlir::vector::ReductionOp>();
    addLegalOp<mlir::vector::ShuffleOp>();
    addLegalOp<mlir::vector::ShapeCastOp>();
    addLegalOp<mlir::memref::ReinterpretCastOp>();

    addLegalDialect<mlir::xegpu::XeGPUDialect>();

    addDynamicallyLegalDialect<mlir::arith::ArithDialect>(
        [&](mlir::Operation *op) { return isLegalArithOp(op); });

    addDynamicallyLegalDialect<mlir::scf::SCFDialect>(
        [&](mlir::Operation *op) { return isLegalSCFOp(op); });

    addDynamicallyLegalOp<mlir::xegpu::DpasOp>(
        [&](mlir::Operation *op) -> bool {
          return (uArchInterface &&
                  mlir::succeeded(uArchInterface->isLegalDpasOp(op)));
        });

    addDynamicallyLegalOp<mlir::xegpu::LoadNdOp>(
        [&](mlir::Operation *op) -> bool {
          return (uArchInterface &&
                  mlir::succeeded(uArchInterface->isLegalLoad2dOp(op)));
        });

    addDynamicallyLegalOp<mlir::xegpu::StoreNdOp>(
        [&](mlir::Operation *op) -> bool {
          return (uArchInterface &&
                  mlir::succeeded(uArchInterface->isLegalStore2dOp(op)));
        });

    addDynamicallyLegalOp<mlir::xegpu::PrefetchNdOp>(
        [&](mlir::Operation *op) -> bool {
          return (uArchInterface &&
                  mlir::succeeded(uArchInterface->isLegalPrefetch2dOp(op)));
        });

    // Arith ops
    addDynamicallyLegalOp<mlir::arith::AddFOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::AddIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::AndIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::DivFOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::DivSIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::DivUIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::MulFOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::MulIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::CmpFOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::CmpIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::XOrIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::SubFOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::SubIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::MaximumFOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::MaxSIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::MaxUIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::RemFOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::RemSIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::RemUIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::NegFOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::MinimumFOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::MinSIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::MinUIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::SelectOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::ExtFOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::ExtSIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::ExtUIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::FPToSIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::FPToUIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::IndexCastOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::IndexCastUIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::SIToFPOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::UIToFPOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::TruncFOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::arith::TruncIOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });

    // Math Ops
    addDynamicallyLegalOp<mlir::math::ExpOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::math::PowFOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::math::SqrtOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::math::LogOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::math::ErfOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::math::SinOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::math::CosOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::math::RsqrtOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });
    addDynamicallyLegalOp<mlir::math::TanhOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });

    addDynamicallyLegalOp<mlir::vector::CreateMaskOp>(
        [&](mlir::Operation *op) -> bool { return isLegalElementWiseOp(op); });

    addDynamicallyLegalOp<mlir::vector::TransposeOp>(
        [](mlir::vector::TransposeOp op) {
          return op.getResult().getType().getRank() == 2;
        });

    addDynamicallyLegalOp<mlir::vector::SplatOp>([&](mlir::vector::SplatOp op) {
      return op.getAggregate().getType().getRank() != 4;
    });
  }

private:
  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;
};

// Full Pass
struct ConvertXeTileToXeGPUPass // convert XeTile to XeGPU
    : public imex::impl::ConvertXeTileToXeGPUBase<ConvertXeTileToXeGPUPass> {
  ConvertXeTileToXeGPUPass() = default;

  ConvertXeTileToXeGPUPass(const std::string &deviceName) {
    if (this->device.getNumOccurrences() == 0) {
      this->device = deviceName;

      if (deviceName == "pvc") {
        uArchInterface = std::make_shared<XePVCuArch>();
      }
    }
  }

  void runOnOperation() override {
    auto mod = getOperation();
    mlir::MLIRContext &context = getContext();

    // skip functions with XeTile.TileType inputs and outputs
    if (!isSupportedModule(mod)) {
      mod.emitOpError(
          "Currently FunctionType with xetile.TileType is not supported.");
      return signalPassFailure();
    }

    if (!uArchInterface) {
      mod.emitOpError("Can not get GPU Arch Definition for given Arch param");
      return signalPassFailure();
    }

    auto &analysis = getAnalysis<TileUsageAnalysis>();
    XeOneToNTypeConverter typeConverter(context);
    XeTileConversionTarget target(context, uArchInterface);
    mlir::RewritePatternSet patterns(&context);

    populateXeTileToXeGPUConversionPatterns(typeConverter, patterns, analysis);

    if (mlir::failed(
            mlir::applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }

private:
  std::shared_ptr<XeuArchInterface> uArchInterface = nullptr;
};

/// Populate the given list with patterns that convert XeTile to XeGPU
void populateXeTileToXeGPUConversionPatterns(
    imex::XeOneToNTypeConverter &converter, mlir::RewritePatternSet &patterns,
    TileUsageAnalysis &analysis) {
  populateSCFOpConversionPatterns(converter, patterns, analysis);
  populateArithOpConversionPatterns(converter, patterns, analysis);
  populateXeTileOpConversionPatterns(converter, patterns, analysis);
}

/// Create a pass that convert XeTile to XeGPU
std::unique_ptr<::mlir::OperationPass<::mlir::gpu::GPUModuleOp>>
createConvertXeTileToXeGPUPass(const std::string &deviceName) {
  return std::make_unique<ConvertXeTileToXeGPUPass>(deviceName);
}

} // namespace imex
