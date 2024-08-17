//===- RegionBufferize.cpp - Bufferization for region ops -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements bufferization of region ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#include "imex/Dialect/Region/IR/RegionOps.h"
#include "imex/Dialect/Region/Transforms/Passes.h"
#include "imex/Dialect/Region/Transforms/RegionConversions.h"

namespace imex {
#define GEN_PASS_DEF_REGIONBUFFERIZE
#include "imex/Dialect/Region/Transforms/Passes.h.inc"
} // namespace imex

static ::mlir::Value materializeToTensorRestrict(::mlir::OpBuilder &builder,
                                                 ::mlir::TensorType type,
                                                 ::mlir::ValueRange inputs,
                                                 ::mlir::Location loc) {
  assert(inputs.size() == 1);
  assert(::mlir::isa<::mlir::BaseMemRefType>(inputs[0].getType()));
  return builder.create<::mlir::bufferization::ToTensorOp>(loc, type, inputs[0],
                                                           /*restrict=*/true);
}

namespace {
struct RegionBufferizePass
    : public ::imex::impl::RegionBufferizeBase<RegionBufferizePass> {
  using ::imex::impl::RegionBufferizeBase<
      RegionBufferizePass>::RegionBufferizeBase;

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    ::mlir::bufferization::BufferizeTypeConverter typeConverter;
    ::mlir::RewritePatternSet patterns(context);
    ::mlir::ConversionTarget target(*context);

    typeConverter.addArgumentMaterialization(materializeToTensorRestrict);
    typeConverter.addSourceMaterialization(materializeToTensorRestrict);
    ::imex::populateRegionTypeConversionPatterns(patterns, typeConverter);

    target.addDynamicallyLegalOp<::imex::region::EnvironmentRegionOp,
                                 ::imex::region::EnvironmentRegionYieldOp>(
        [&](mlir::Operation *op) { return typeConverter.isLegal(op); });

    if (::mlir::failed(::mlir::applyPartialConversion(module, target,
                                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<::mlir::Pass> imex::createRegionBufferizePass() {
  return std::make_unique<RegionBufferizePass>();
}
