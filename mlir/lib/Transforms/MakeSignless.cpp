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

#include "imex/Transforms/MakeSignless.hpp"

#include "imex/Dialect/imex_util/dialect.hpp"
#include "imex/Transforms/type_conversion.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

static llvm::Optional<mlir::Type> makeSignlessType(mlir::Type type) {
  if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    if (intType.getSignedness() != mlir::IntegerType::Signless)
      return mlir::IntegerType::get(type.getContext(), intType.getWidth());
  } else if (auto shapedType = type.dyn_cast<mlir::ShapedType>()) {
    if (auto signlessElem = makeSignlessType(shapedType.getElementType()))
      return shapedType.clone(*signlessElem);
  }

  return llvm::None;
}

void imex::populateMakeSignlessRewritesAndTarget(
    mlir::MLIRContext &context, mlir::TypeConverter &converter,
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target) {
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
}

namespace {
struct MakeSignlessPass
    : public mlir::PassWrapper<MakeSignlessPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MakeSignlessPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<imex::util::ImexUtilDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
    mlir::TypeConverter converter;
    mlir::RewritePatternSet patterns(&context);
    mlir::ConversionTarget target(context);

    // Convert unknown types to itself
    converter.addConversion([](mlir::Type type) { return type; });

    imex::populateTupleTypeConverter(context, converter);

    imex::populateMakeSignlessRewritesAndTarget(context, converter, patterns,
                                                target);

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
