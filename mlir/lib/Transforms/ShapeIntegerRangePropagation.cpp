// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/ShapeIntegerRangePropagation.hpp"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/IntegerRangeAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/ShapedOpInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace {
static auto getIndexRange(int64_t umin, int64_t umax) {
  unsigned width = mlir::IndexType::kInternalStorageBitWidth;
  return mlir::ConstantIntRanges::fromSigned(llvm::APInt(width, umin),
                                             llvm::APInt(width, umax));
}

class IntegerRangeAnalysisEx : public mlir::dataflow::IntegerRangeAnalysis {
public:
  using IntegerRangeAnalysis::IntegerRangeAnalysis;

  void visitOperation(
      mlir::Operation *op,
      llvm::ArrayRef<const mlir::dataflow::IntegerValueRangeLattice *> operands,
      llvm::ArrayRef<mlir::dataflow::IntegerValueRangeLattice *> results)
      override {
    if (auto dim = mlir::dyn_cast<mlir::ShapedDimOpInterface>(op)) {
      assert(op->getNumResults() == 1);
      assert(results.size() == 1);

      auto *lattice = results.front();
      auto oldRange = lattice->getValue();
      auto newRange = getIndexRange(0, std::numeric_limits<int64_t>::max());
      auto changed = lattice->join(mlir::dataflow::IntegerValueRange{newRange});
      propagateIfChanged(lattice, changed);
      return;
    }

    mlir::dataflow::IntegerRangeAnalysis::visitOperation(op, operands, results);
  }
};

static bool intersects(mlir::ConstantIntRanges lhs,
                       mlir::ConstantIntRanges rhs) {
  if ((lhs.smax().sle(rhs.smin()) || lhs.smin().sge(rhs.smax())) &&
      (lhs.umax().ule(rhs.umin()) || lhs.umin().uge(rhs.umax())))
    return false;

  return true;
}

static llvm::Optional<bool> handleEq(mlir::ConstantIntRanges lhs,
                                     mlir::ConstantIntRanges rhs) {
  if (!intersects(lhs, rhs))
    return false;

  return llvm::None;
}

static llvm::Optional<bool> handleNe(mlir::ConstantIntRanges lhs,
                                     mlir::ConstantIntRanges rhs) {
  if (!intersects(lhs, rhs))
    return true;

  return llvm::None;
}

static llvm::Optional<bool> handleSlt(mlir::ConstantIntRanges lhs,
                                      mlir::ConstantIntRanges rhs) {
  if (lhs.smax().slt(rhs.smin()))
    return true;

  if (lhs.smin().sge(rhs.smax()))
    return false;

  return llvm::None;
}

static llvm::Optional<bool> handleSle(mlir::ConstantIntRanges lhs,
                                      mlir::ConstantIntRanges rhs) {
  if (lhs.smax().sle(rhs.smin()))
    return true;

  if (lhs.smin().sgt(rhs.smax()))
    return false;

  return llvm::None;
}

static llvm::Optional<bool> handleSgt(mlir::ConstantIntRanges lhs,
                                      mlir::ConstantIntRanges rhs) {
  return handleSlt(rhs, lhs);
}

static llvm::Optional<bool> handleSge(mlir::ConstantIntRanges lhs,
                                      mlir::ConstantIntRanges rhs) {
  return handleSle(rhs, lhs);
}

static llvm::Optional<bool> handleUlt(mlir::ConstantIntRanges lhs,
                                      mlir::ConstantIntRanges rhs) {
  if (lhs.umax().ult(rhs.umin()))
    return true;

  if (lhs.umin().uge(rhs.umax()))
    return false;

  return llvm::None;
}

static llvm::Optional<bool> handleUle(mlir::ConstantIntRanges lhs,
                                      mlir::ConstantIntRanges rhs) {
  if (lhs.umax().ule(rhs.umin()))
    return true;

  if (lhs.umin().ugt(rhs.umax()))
    return false;

  return llvm::None;
}

static llvm::Optional<bool> handleUgt(mlir::ConstantIntRanges lhs,
                                      mlir::ConstantIntRanges rhs) {
  return handleUlt(rhs, lhs);
}

static llvm::Optional<bool> handleUge(mlir::ConstantIntRanges lhs,
                                      mlir::ConstantIntRanges rhs) {
  return handleUle(rhs, lhs);
}

struct ConvertCmpOp : public mlir::OpRewritePattern<mlir::arith::CmpIOp> {

  ConvertCmpOp(mlir::MLIRContext *context, mlir::DataFlowSolver &s)
      : mlir::OpRewritePattern<mlir::arith::CmpIOp>(context), solver(s) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::CmpIOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto *lhsResult =
        solver.lookupState<mlir::dataflow::IntegerValueRangeLattice>(
            op.getLhs());
    if (!lhsResult || lhsResult->getValue().isUninitialized())
      return mlir::failure();

    auto *rhsResult =
        solver.lookupState<mlir::dataflow::IntegerValueRangeLattice>(
            op.getRhs());
    if (!lhsResult || rhsResult->getValue().isUninitialized())
      return mlir::failure();

    using HandlerFunc = llvm::Optional<bool> (*)(mlir::ConstantIntRanges,
                                                 mlir::ConstantIntRanges);
    std::array<HandlerFunc, mlir::arith::getMaxEnumValForCmpIPredicate() + 1>
        handlers{};
    using Pred = mlir::arith::CmpIPredicate;
    handlers[static_cast<size_t>(Pred::eq)] = &handleEq;
    handlers[static_cast<size_t>(Pred::ne)] = &handleNe;
    handlers[static_cast<size_t>(Pred::slt)] = &handleSlt;
    handlers[static_cast<size_t>(Pred::sle)] = &handleSle;
    handlers[static_cast<size_t>(Pred::sgt)] = &handleSgt;
    handlers[static_cast<size_t>(Pred::sge)] = &handleSge;
    handlers[static_cast<size_t>(Pred::ult)] = &handleUlt;
    handlers[static_cast<size_t>(Pred::ule)] = &handleUle;
    handlers[static_cast<size_t>(Pred::ugt)] = &handleUgt;
    handlers[static_cast<size_t>(Pred::uge)] = &handleUge;

    auto handler = handlers[static_cast<size_t>(op.getPredicate())];
    if (!handler)
      return mlir::failure();

    auto result = handler(lhsResult->getValue().getValue(),
                          rhsResult->getValue().getValue());
    if (!result)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::arith::ConstantIntOp>(
        op, static_cast<int64_t>(*result), /*width*/ 1);
    return mlir::success();
  }

private:
  mlir::DataFlowSolver &solver;
};

struct ShapeIntegerRangePropagationPass
    : public mlir::PassWrapper<ShapeIntegerRangePropagationPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeIntegerRangePropagationPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto op = getOperation();
    mlir::DataFlowSolver solver;
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<IntegerRangeAnalysisEx>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.add<ConvertCmpOp>(ctx, solver);

    (void)mlir::applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<mlir::Pass> imex::createShapeIntegerRangePropagationPass() {
  return std::make_unique<ShapeIntegerRangePropagationPass>();
}
