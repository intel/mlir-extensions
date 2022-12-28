// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/UpliftMath.hpp"

#include "imex/Dialect/imex_util/Dialect.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

template <typename Op>
static mlir::Operation *replaceOp1(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::ValueRange args) {
  if (args.size() != 1)
    return nullptr;

  return builder.create<Op>(loc, args.front());
}

template <typename Op>
static mlir::Operation *replaceOp2(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::ValueRange args) {
  if (args.size() != 2)
    return nullptr;

  return builder.create<Op>(loc, args[0], args[1]);
}

namespace {
struct UpliftMathCalls : public mlir::OpRewritePattern<mlir::func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto funcName = op.getCallee();
    if (funcName.empty())
      return mlir::failure();

    auto isNotValidType = [](mlir::Type t) { return !t.isIntOrFloat(); };

    if (llvm::any_of(op.getOperandTypes(), isNotValidType) ||
        op.getNumResults() != 1 ||
        llvm::any_of(op.getResultTypes(), isNotValidType))
      return mlir::failure();

    llvm::StringRef funcNameF =
        (funcName.front() == 'f' ? funcName.drop_front() : llvm::StringRef{});

    using func_t = mlir::Operation *(*)(mlir::OpBuilder &, mlir::Location,
                                        mlir::ValueRange);
    const std::pair<llvm::StringRef, func_t> handlers[] = {
        {"floor", &replaceOp1<mlir::math::FloorOp>},
        {"log", &replaceOp1<mlir::math::LogOp>},
        {"sqrt", &replaceOp1<mlir::math::SqrtOp>},
        {"exp", &replaceOp1<mlir::math::ExpOp>},
        {"sin", &replaceOp1<mlir::math::SinOp>},
        {"cos", &replaceOp1<mlir::math::CosOp>},
        {"erf", &replaceOp1<mlir::math::ErfOp>},
        {"tanh", &replaceOp1<mlir::math::TanhOp>},
        {"atan2", &replaceOp2<mlir::math::Atan2Op>},
    };

    for (auto &handler : handlers) {
      auto name = handler.first;
      if (name == funcName || name == funcNameF) {
        auto res = handler.second(rewriter, op.getLoc(), op.getOperands());
        if (!res)
          return mlir::failure();

        assert(res->getNumResults() == op->getNumResults());
        rewriter.replaceOp(op, res->getResults());
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

struct UpliftFabsCalls : public mlir::OpRewritePattern<mlir::func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto funcName = op.getCallee();
    if (funcName.empty())
      return mlir::failure();

    if (funcName != "fabs" && funcName != "fabsf")
      return mlir::failure();

    auto isNotValidType = [](mlir::Type t) {
      return !t.isa<mlir::FloatType>();
    };

    if (op.getNumResults() != 1 || op.getNumOperands() != 1 ||
        llvm::any_of(op.getOperandTypes(), isNotValidType) ||
        llvm::any_of(op.getResultTypes(), isNotValidType))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::math::AbsFOp>(op,
                                                    op.getOperands().front());
    return mlir::success();
  }
};

struct UpliftCabsCalls : public mlir::OpRewritePattern<mlir::func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto funcName = op.getCallee();
    if (funcName.empty())
      return mlir::failure();

    if (funcName != "cabs" && funcName != "cabsf")
      return mlir::failure();

    if (op.getNumResults() != 1 || op.getNumOperands() != 1)
      return mlir::failure();

    auto val = op.getOperands().front();
    auto srcType = val.getType().dyn_cast<mlir::ComplexType>();

    if (!srcType || srcType.getElementType() != op.getResult(0).getType())
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::complex::AbsOp>(
        op, srcType.getElementType(), val);
    return mlir::success();
  }
};

struct UpliftFma : public mlir::OpRewritePattern<mlir::arith::AddFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::AddFOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto func = op->getParentOfType<mlir::func::FuncOp>();
    if (!func || !func->hasAttr(imex::util::attributes::getFastmathName()))
      return mlir::failure();

    mlir::Value c;
    mlir::arith::MulFOp ab;
    if ((ab = op.getLhs().getDefiningOp<mlir::arith::MulFOp>())) {
      c = op.getRhs();
    } else if ((ab = op.getRhs().getDefiningOp<mlir::arith::MulFOp>())) {
      c = op.getLhs();
    } else {
      return mlir::failure();
    }

    auto a = ab.getLhs();
    auto b = ab.getRhs();
    rewriter.replaceOpWithNewOp<mlir::math::FmaOp>(op, a, b, c);
    return mlir::success();
  }
};

struct UpliftMathPass
    : public mlir::PassWrapper<UpliftMathPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UpliftMathPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::complex::ComplexDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::math::MathDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    imex::populateUpliftMathPatterns(patterns);
    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

struct UpliftFMAPass
    : public mlir::PassWrapper<UpliftFMAPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UpliftFMAPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::math::MathDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    imex::populateUpliftFMAPatterns(patterns);
    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};
} // namespace

void imex::populateUpliftMathPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<UpliftMathCalls, UpliftFabsCalls, UpliftCabsCalls>(
      patterns.getContext());
}

void imex::populateUpliftFMAPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<UpliftFma>(patterns.getContext());
}

std::unique_ptr<mlir::Pass> imex::createUpliftMathPass() {
  return std::make_unique<UpliftMathPass>();
}

std::unique_ptr<mlir::Pass> imex::createUpliftFMAPass() {
  return std::make_unique<UpliftFMAPass>();
}
