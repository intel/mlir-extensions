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

#include <mlir/Conversion/Passes.h>

#include "mlir-extensions/Dialect/plier_util/dialect.hpp"

#include "mlir-extensions/Conversion/util_conversion.hpp"

namespace {
struct ConvertTakeContext
    : public mlir::OpConversionPattern<plier::TakeContextOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::TakeContextOp op,
                  plier::TakeContextOp::Adaptor adaptor,
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

    auto initFunc = adaptor.initFunc().value_or(mlir::SymbolRefAttr());
    auto releaseFunc = adaptor.releaseFunc().value_or(mlir::SymbolRefAttr());
    rewriter.replaceOpWithNewOp<plier::TakeContextOp>(op, newTypes, initFunc,
                                                      releaseFunc);
    return mlir::success();
  }
};
} // namespace

void imex::populateUtilConversionPatterns(mlir::MLIRContext &context,
                                          mlir::TypeConverter &converter,
                                          mlir::RewritePatternSet &patterns,
                                          mlir::ConversionTarget &target) {
  patterns.insert<ConvertTakeContext>(converter, &context);

  target.addDynamicallyLegalOp<plier::TakeContextOp>(
      [&](mlir::Operation *op) -> llvm::Optional<bool> {
        for (auto range : {mlir::TypeRange(op->getOperandTypes()),
                           mlir::TypeRange(op->getResultTypes())})
          for (auto type : range)
            if (converter.isLegal(type))
              return true;

        return llvm::None;
      });
}
