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

#include "pipelines/pre_low_simplifications.hpp"

#include "pipelines/base_pipeline.hpp"

#include "imex/Dialect/imex_util/dialect.hpp"
#include "imex/Dialect/plier/dialect.hpp"
#include "imex/Transforms/expand_tuple.hpp"
#include "imex/Transforms/type_conversion.hpp"
#include "imex/compiler/pipeline_registry.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

namespace {
struct MakeSignlessPass
    : public mlir::PassWrapper<MakeSignlessPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MakeSignlessPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<plier::PlierDialect>();
  }

  void runOnOperation() override final {
    auto module = getOperation();
    auto *context = &getContext();

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::Type type) { return type; });
    typeConverter.addConversion(
        [](mlir::IntegerType type) -> llvm::Optional<mlir::Type> {
          if (!type.isSignless()) {
            return mlir::IntegerType::get(type.getContext(), type.getWidth());
          }
          return llvm::None;
        });
    imex::populateTupleTypeConverter(*context, typeConverter);

    auto materializeSignCast = [](mlir::OpBuilder &builder, mlir::Type type,
                                  mlir::ValueRange inputs,
                                  mlir::Location loc) -> mlir::Value {
      assert(inputs.size() == 1);
      return builder.create<imex::util::SignCastOp>(loc, type, inputs[0]);
    };
    typeConverter.addArgumentMaterialization(materializeSignCast);
    typeConverter.addSourceMaterialization(materializeSignCast);
    typeConverter.addTargetMaterialization(materializeSignCast);

    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);

    imex::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                             patterns, target);
    imex::populateTupleTypeConversionRewritesAndTarget(typeConverter, patterns,
                                                       target);

    if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

void populateUntuplePipeline(mlir::OpPassManager &pm) {
  pm.addPass(imex::createExpandTuplePass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void populateRemoveSignPipeline(mlir::OpPassManager &pm) {
  pm.addPass(std::make_unique<MakeSignlessPass>());
  pm.addPass(mlir::createCanonicalizerPass());
}
} // namespace

void registerPreLowSimpleficationsPipeline(imex::PipelineRegistry &registry) {
  registry.registerPipeline([](auto sink) {
    auto stage = getHighLoweringStage();
    sink(untuplePipelineName(), {stage.begin}, {stage.end}, {},
         &populateUntuplePipeline);
  });
  registry.registerPipeline([](auto sink) {
    auto stage = getHighLoweringStage();
    sink(removeSignPipelineName(), {untuplePipelineName(), stage.begin},
         {stage.end}, {}, &populateRemoveSignPipeline);
  });
}

llvm::StringRef untuplePipelineName() { return "pre_low_untuple"; }

llvm::StringRef removeSignPipelineName() { return "pre_low_remove_sign"; }
