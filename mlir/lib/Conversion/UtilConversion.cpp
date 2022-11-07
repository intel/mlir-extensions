// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Conversion/UtilConversion.hpp"

#include "imex/Dialect/imex_util/Dialect.hpp"

#include <mlir/Conversion/Passes.h>

namespace {
struct ConvertTakeContext
    : public mlir::OpConversionPattern<imex::util::TakeContextOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::TakeContextOp op,
                  imex::util::TakeContextOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto srcTypes = op.getResultTypes();
    auto count = static_cast<unsigned>(srcTypes.size());
    llvm::SmallVector<mlir::Type> newTypes(count);
    auto converter = getTypeConverter();
    assert(converter);
    for (auto i : llvm::seq(0u, count)) {
      auto oldType = srcTypes[i];
      auto newType = converter->convertType(oldType);
      newTypes[i] = (newType ? newType : oldType);
    }

    auto initFunc = adaptor.getInitFunc().value_or(mlir::SymbolRefAttr());
    auto releaseFunc = adaptor.getReleaseFunc().value_or(mlir::SymbolRefAttr());
    rewriter.replaceOpWithNewOp<imex::util::TakeContextOp>(
        op, newTypes, initFunc, releaseFunc);
    return mlir::success();
  }
};
} // namespace

void imex::populateUtilConversionPatterns(mlir::TypeConverter &converter,
                                          mlir::RewritePatternSet &patterns,
                                          mlir::ConversionTarget &target) {
  patterns.insert<ConvertTakeContext>(converter, patterns.getContext());

  target.addDynamicallyLegalOp<imex::util::TakeContextOp>(
      [&](mlir::Operation *op) -> llvm::Optional<bool> {
        for (auto range : {mlir::TypeRange(op->getOperandTypes()),
                           mlir::TypeRange(op->getResultTypes())})
          for (auto type : range)
            if (converter.isLegal(type))
              return true;

        return llvm::None;
      });
}
