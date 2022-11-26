// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/CompositePass.hpp"

#include "mlir/Pass/PassManager.h"
#include <mlir/Pass/Pass.h>

namespace {
struct CompositePass
    : public mlir::PassWrapper<CompositePass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CompositePass)

  CompositePass(std::string name_,
                std::function<void(mlir::OpPassManager &)> func)
      : name(std::move(name_)), populateFunc(std::move(func)) {}

  void runOnOperation() override {
    assert(populateFunc);
    auto op = getOperation();
    mlir::OperationFingerPrint fp(op);

    int currentIter = 0;
    int maxIters = 10;

    mlir::OpPassManager dynamicPM(op->getName());
    populateFunc(dynamicPM);
    while (true) {
      if (mlir::failed(runPipeline(dynamicPM, op)))
        return signalPassFailure();

      if (currentIter++ >= maxIters) {
        //        op->emitWarning("Composite pass didn't converge in " +
        //                        llvm::Twine(maxIters) + " iterations");
        break;
      }

      mlir::OperationFingerPrint newFp(op);
      if (newFp == fp)
        break;

      fp = newFp;
    }
  }

protected:
  llvm::StringRef getName() const override {
    assert(!name.empty());
    return name;
  }

private:
  std::string name;
  std::function<void(mlir::OpPassManager &)> populateFunc;
};
} // namespace

std::unique_ptr<mlir::Pass> imex::createCompositePass(
    std::string name, std::function<void(mlir::OpPassManager &)> populateFunc) {
  assert(!name.empty());
  assert(populateFunc);
  return std::make_unique<CompositePass>(std::move(name),
                                         std::move(populateFunc));
}
