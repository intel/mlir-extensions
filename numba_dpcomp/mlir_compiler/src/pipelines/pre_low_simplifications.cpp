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
#include "pipelines/plier_to_std.hpp"

#include "plier/compiler/pipeline_registry.hpp"
#include "plier/dialect.hpp"
#include "plier/rewrites/type_conversion.hpp"

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

namespace {

static void flattenTuple(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ValueRange values,
                         llvm::SmallVectorImpl<mlir::Value> &ret) {
  for (auto arg : values) {
    if (auto tupleType = arg.getType().dyn_cast<mlir::TupleType>()) {
      for (auto it : llvm::enumerate(tupleType.getTypes())) {
        auto i = it.index();
        auto argType = it.value();
        auto ind = builder.createOrFold<mlir::ConstantIndexOp>(loc, i);
        auto res =
            builder.createOrFold<plier::GetItemOp>(loc, argType, arg, ind);
        flattenTuple(builder, loc, res, ret);
      }
    } else {
      ret.emplace_back(arg);
    }
  }
}

struct UntupleReturn : public mlir::OpConversionPattern<mlir::ReturnOp> {
  using mlir::OpConversionPattern<mlir::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::ReturnOp op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> newOperands;
    auto loc = op.getLoc();
    flattenTuple(rewriter, loc, operands, newOperands);
    auto *operation = op.getOperation();
    rewriter.updateRootInPlace(op,
                               [&]() { operation->setOperands(newOperands); });
    return mlir::success();
  }
};

struct UntuplePass
    : public mlir::PassWrapper<UntuplePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    mlir::TypeConverter typeConverter;
    // Convert unknown types to itself
    typeConverter.addConversion([](mlir::Type type) { return type; });
    typeConverter.addConversion(
        [&typeConverter](mlir::TupleType type,
                         llvm::SmallVectorImpl<mlir::Type> &ret)
            -> llvm::Optional<mlir::LogicalResult> {
          if (mlir::failed(typeConverter.convertTypes(type.getTypes(), ret)))
            return llvm::None;
          return mlir::success();
        });

    auto materializeTupleCast =
        [](mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs,
           mlir::Location loc) -> llvm::Optional<mlir::Value> {
      if (auto tuple = type.dyn_cast<mlir::TupleType>()) {
        auto retType =
            mlir::TupleType::get(type.getContext(), inputs.getTypes());
        return builder.create<plier::BuildTupleOp>(loc, retType, inputs)
            .getResult();
      }
      return llvm::None;
    };
    typeConverter.addArgumentMaterialization(materializeTupleCast);
    typeConverter.addSourceMaterialization(materializeTupleCast);
    typeConverter.addTargetMaterialization(materializeTupleCast);

    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);

    plier::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                              patterns, target);

    patterns.insert<UntupleReturn>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

struct MakeSignlessPass
    : public mlir::PassWrapper<MakeSignlessPass, mlir::OperationPass<void>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect>();
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
    populateTupleTypeConverter(*context, typeConverter);

    auto materializeSignCast = [](mlir::OpBuilder &builder, mlir::Type type,
                                  mlir::ValueRange inputs,
                                  mlir::Location loc) -> mlir::Value {
      assert(inputs.size() == 1);
      return builder.create<plier::SignCastOp>(loc, type, inputs[0]);
    };
    typeConverter.addArgumentMaterialization(materializeSignCast);
    typeConverter.addSourceMaterialization(materializeSignCast);
    typeConverter.addTargetMaterialization(materializeSignCast);

    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);

    plier::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                              patterns, target);
    plier::populateTupleTypeConversionRewritesAndTarget(typeConverter, patterns,
                                                        target);

    if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

void populateUntuplePipeline(mlir::OpPassManager &pm) {
  pm.addPass(std::make_unique<UntuplePass>());
  pm.addPass(mlir::createCanonicalizerPass());
}

void populateRemoveSignPipeline(mlir::OpPassManager &pm) {
  pm.addPass(std::make_unique<MakeSignlessPass>());
  pm.addPass(mlir::createCanonicalizerPass());
}
} // namespace

void registerPreLowSimpleficationsPipeline(plier::PipelineRegistry &registry) {
  registry.register_pipeline([](auto sink) {
    auto stage = getHighLoweringStage();
    sink(untuplePipelineName(), {stage.begin}, {stage.end}, {},
         &populateUntuplePipeline);
  });
  registry.register_pipeline([](auto sink) {
    auto stage = getHighLoweringStage();
    sink(removeSignPipelineName(), {untuplePipelineName(), stage.begin},
         {stage.end}, {}, &populateRemoveSignPipeline);
  });
}

llvm::StringRef untuplePipelineName() { return "pre_low_untuple"; }

llvm::StringRef removeSignPipelineName() { return "pre_low_remove_sign"; }
