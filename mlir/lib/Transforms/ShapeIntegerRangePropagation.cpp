// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Transforms/ShapeIntegerRangePropagation.hpp"

#include "imex/Dialect/imex_util/Dialect.hpp"

#include <llvm/Support/Debug.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/IntegerRangeAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/ShapedOpInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#define DEBUG_TYPE "imex-shape-range-propagation"

namespace {
static auto getIndexRange(int64_t smin, int64_t smax) {
  unsigned width = mlir::IndexType::kInternalStorageBitWidth;
  return mlir::ConstantIntRanges::fromSigned(llvm::APInt(width, smin),
                                             llvm::APInt(width, smax));
}

static auto getDefaultDimRange() {
  return getIndexRange(0, std::numeric_limits<int64_t>::max());
}

static auto getFixedDimRange(int64_t val) { return getIndexRange(val, val); }

class ShapeValue {
public:
  ShapeValue() = default;
  ShapeValue(mlir::ShapedType shaped) : shapeRanges(std::in_place) {
    shapeRanges->reserve(shaped.getRank());
    for (auto dim : shaped.getShape())
      shapeRanges->emplace_back(mlir::ShapedType::isDynamic(dim)
                                    ? getDefaultDimRange()
                                    : getFixedDimRange(dim));
  }
  ShapeValue(mlir::ArrayAttr attr) : shapeRanges(std::in_place) {
    shapeRanges->reserve(attr.size());
    for (auto elem : attr) {
      auto range = elem.cast<imex::util::IndexRangeAttr>();
      shapeRanges->emplace_back(getIndexRange(range.getMin(), range.getMax()));
    }
  }
  ShapeValue(mlir::ArrayRef<mlir::ConstantIntRanges> values)
      : shapeRanges(std::in_place) {
    shapeRanges->assign(values.begin(), values.end());
  }

  /// Whether the state is uninitialized.
  bool isUninitialized() const { return !shapeRanges; }

  llvm::ArrayRef<mlir::ConstantIntRanges> getShape() const {
    assert(!isUninitialized());
    return *shapeRanges;
  }

  static ShapeValue join(const ShapeValue &lhs, const ShapeValue &rhs) {
    if (lhs.isUninitialized())
      return rhs;

    if (rhs.isUninitialized())
      return lhs;

    llvm::SmallVector<mlir::ConstantIntRanges> resShapes;
    resShapes.reserve(
        std::min(lhs.shapeRanges->size(), rhs.shapeRanges->size()));
    for (auto [l, r] : llvm::zip(*lhs.shapeRanges, *rhs.shapeRanges))
      resShapes.emplace_back(l.rangeUnion(r));

    ShapeValue ret;
    ret.shapeRanges = std::move(resShapes);
    return ret;
  }

  static ShapeValue intersect(const ShapeValue &lhs, const ShapeValue &rhs) {
    if (lhs.isUninitialized())
      return rhs;

    if (rhs.isUninitialized())
      return lhs;

    llvm::SmallVector<mlir::ConstantIntRanges> resShapes;
    resShapes.reserve(
        std::min(lhs.shapeRanges->size(), rhs.shapeRanges->size()));
    for (auto [l, r] : llvm::zip(*lhs.shapeRanges, *rhs.shapeRanges))
      resShapes.emplace_back(l.intersection(r));

    ShapeValue ret;
    ret.shapeRanges = std::move(resShapes);
    return ret;
  }

  bool operator==(const ShapeValue &rhs) const {
    return shapeRanges == rhs.shapeRanges;
  }

  void print(llvm::raw_ostream &os) const {
    if (isUninitialized()) {
      os << "None";
    } else {
      llvm::interleaveComma(*shapeRanges, os);
    }
  }

private:
  llvm::Optional<llvm::SmallVector<mlir::ConstantIntRanges>> shapeRanges;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ShapeValue &state) {
  state.print(os);
  return os;
}

struct TensorValueLattice : public mlir::dataflow::Lattice<ShapeValue> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensorValueLattice)
  using Lattice::Lattice;
};

class TensorValueAnalysis
    : public mlir::dataflow::SparseDataFlowAnalysis<TensorValueLattice> {
public:
  using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

  void visitOperation(mlir::Operation *op,
                      llvm::ArrayRef<const TensorValueLattice *> /*operands*/,
                      llvm::ArrayRef<TensorValueLattice *> results) override {
    LLVM_DEBUG(llvm::dbgs()
               << "TensorValueAnalysis: Visiting operation: " << *op << "\n");

    if (auto fromElements = mlir::dyn_cast<mlir::tensor::FromElementsOp>(op)) {
      assert(results.size() == 1);
      auto tensorType =
          fromElements.getResult().getType().cast<mlir::RankedTensorType>();
      if (tensorType.getRank() != 1 ||
          mlir::ShapedType::isDynamic(tensorType.getShape().front()))
        return;

      auto args = fromElements.getElements();
      llvm::SmallVector<mlir::ConstantIntRanges> ranges;
      ranges.reserve(args.size());
      for (auto arg : args) {
        auto state =
            getOrCreateFor<mlir::dataflow::IntegerValueRangeLattice>(op, arg);

        if (!state)
          return;

        auto val = state->getValue();
        if (val.isUninitialized())
          return;

        ranges.emplace_back(val.getValue());
      }

      ShapeValue newVal(ranges);
      LLVM_DEBUG(llvm::dbgs()
                 << "TensorValueAnalysis: New result val: " << newVal << "\n");

      auto resultLattice = results.front();
      auto changed = resultLattice->join(newVal);
      propagateIfChanged(resultLattice, changed);
      return;
    }
  }

  void setToEntryState(TensorValueLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(ShapeValue{}));
  }
};

struct ShapeValueLattice : public mlir::dataflow::Lattice<ShapeValue> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeValueLattice)
  using Lattice::Lattice;
};

static bool isShapedCast(mlir::Operation *op) {
  return mlir::isa<mlir::CastOpInterface>(op) && op->getNumOperands() == 1 &&
         op->getNumResults() == 1 &&
         mlir::isa<mlir::ShapedType>(op->getOperand(0).getType()) &&
         mlir::isa<mlir::ShapedType>(op->getResult(0).getType());
}

class ShapeValueAnalysis
    : public mlir::dataflow::SparseDataFlowAnalysis<ShapeValueLattice> {
public:
  using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

  void visitOperation(mlir::Operation *op,
                      llvm::ArrayRef<const ShapeValueLattice *> operands,
                      llvm::ArrayRef<ShapeValueLattice *> results) override {
    LLVM_DEBUG(llvm::dbgs()
               << "ShapeValueAnalysis: Visiting operation: " << *op << "\n");

    if (auto reshape = mlir::dyn_cast<mlir::tensor::ReshapeOp>(op)) {
      assert(results.size() == 1);
      auto state = getOrCreateFor<TensorValueLattice>(op, reshape.getShape());
      if (!state)
        return;

      auto val = state->getValue();
      if (val.isUninitialized())
        return;

      LLVM_DEBUG(llvm::dbgs()
                 << "ShapeValueAnalysis: New reshape result: " << val << "\n");

      auto resultLattice = results.front();
      auto changed = resultLattice->join(val);
      propagateIfChanged(resultLattice, changed);
      return;
    }

    if (auto sizesInterface =
            mlir::dyn_cast<mlir::OffsetSizeAndStrideOpInterface>(op)) {
      if (op->getNumResults() != 1)
        return;

      auto result = op->getResult(0);
      auto shaped = result.getType().dyn_cast<mlir::ShapedType>();
      if (!shaped)
        return;

      auto resultShape = shaped.getShape();
      auto mixedSizes = sizesInterface.getMixedSizes();

      llvm::SmallBitVector droppedDims(mixedSizes.size());
      unsigned shapePos = 0;
      for (const auto &size : enumerate(mixedSizes)) {
        auto sizeVal = getConstantIntValue(size.value());
        // If the size is not 1, or if the current matched dimension of the
        // result is the same static shape as the size value (which is 1), then
        // the dimension is preserved.
        if (!sizeVal || *sizeVal != 1 ||
            (shapePos < resultShape.size() && resultShape[shapePos] == 1)) {
          shapePos++;
          continue;
        }
        droppedDims.set(size.index());
      }

      llvm::SmallVector<mlir::ConstantIntRanges> ranges;
      ranges.reserve(mixedSizes.size());
      for (auto [i, size] : llvm::enumerate(mixedSizes)) {
        if (droppedDims[i])
          continue;

        if (auto val = mlir::getConstantIntValue(size)) {
          ranges.emplace_back(getFixedDimRange(*val));
        } else {
          assert(size.is<mlir::Value>());
          auto state = getOrCreateFor<mlir::dataflow::IntegerValueRangeLattice>(
              op, size.get<mlir::Value>());

          if (!state)
            return;

          auto value = state->getValue();
          if (value.isUninitialized())
            return;

          ranges.emplace_back(value.getValue());
        }
      }

      auto newVal = ShapeValue::intersect({shaped}, {ranges});

      LLVM_DEBUG(llvm::dbgs()
                 << "ShapeValueAnalysis: New view result: " << newVal << "\n");

      auto resultLattice = results.front();
      auto changed = resultLattice->join(newVal);
      propagateIfChanged(resultLattice, changed);
      return;
    }

    if (auto enforceShape = mlir::dyn_cast<imex::util::EnforceShapeOp>(op)) {
      auto srcShaped =
          enforceShape.getValue().getType().dyn_cast<mlir::ShapedType>();
      if (!srcShaped)
        return;

      auto dstShaped =
          enforceShape.getResult().getType().dyn_cast<mlir::ShapedType>();
      if (!dstShaped)
        return;

      auto args = enforceShape.getSizes();

      llvm::SmallVector<mlir::ConstantIntRanges> ranges;
      ranges.reserve(args.size());
      for (auto arg : args) {
        auto state =
            getOrCreateFor<mlir::dataflow::IntegerValueRangeLattice>(op, arg);

        if (!state)
          return;

        auto value = state->getValue();
        if (value.isUninitialized())
          return;

        ranges.emplace_back(value.getValue());
      }

      ShapeValue newVal(ranges);
      newVal = ShapeValue::intersect(newVal, {srcShaped});
      newVal = ShapeValue::intersect(newVal, {dstShaped});

      LLVM_DEBUG(llvm::dbgs()
                 << "ShapeValueAnalysis: New enforce shape result: " << newVal
                 << "\n");

      auto resultLattice = results.front();
      auto changed = resultLattice->join(newVal);
      propagateIfChanged(resultLattice, changed);
      return;
    }

    if (isShapedCast(op)) {
      assert(operands.size() == 1);
      assert(results.size() == 1);

      auto srcShaped = op->getOperand(0).getType().cast<mlir::ShapedType>();
      auto dstShaped = op->getResult(0).getType().cast<mlir::ShapedType>();
      auto res =
          ShapeValue::intersect(ShapeValue{srcShaped}, ShapeValue{dstShaped});
      res = ShapeValue::intersect(operands.front()->getValue(),
                                  ShapeValue{dstShaped});

      LLVM_DEBUG(llvm::dbgs()
                 << "ShapeValueAnalysis: Shaped cast: " << res << "\n");

      auto *resultLattice = results.front();
      auto changed = resultLattice->join(res);
      propagateIfChanged(resultLattice, changed);
      return;
    }

    if (auto select = mlir::dyn_cast<mlir::arith::SelectOp>(op)) {
      if (!mlir::isa<mlir::ShapedType>(select.getResult().getType()))
        return;

      assert(operands.size() == 3);
      assert(results.size() == 1);
      auto lhs = operands[1];
      auto rhs = operands[2];
      auto newVal = ShapeValue::join(lhs->getValue(), rhs->getValue());

      LLVM_DEBUG(llvm::dbgs()
                 << "ShapeValueAnalysis: select: " << newVal << "\n");

      auto resultLattice = results.front();
      auto changed = resultLattice->join(newVal);
      propagateIfChanged(resultLattice, changed);
      return;
    }

    for (auto [res, resultLattice] : llvm::zip(op->getResults(), results)) {
      auto shaped = res.getType().dyn_cast<mlir::ShapedType>();
      if (!shaped)
        continue;

      auto newVal = ShapeValue{shaped};
      LLVM_DEBUG(llvm::dbgs()
                 << "ShapeValueAnalysis: New result val: " << newVal << "\n");

      auto changed = resultLattice->join(newVal);
      propagateIfChanged(resultLattice, changed);
    }
  }

  void setToEntryState(ShapeValueLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(ShapeValue{}));
  }

  mlir::LogicalResult initialize(mlir::Operation *top) override {
    if (mlir::failed(SparseDataFlowAnalysis::initialize(top)))
      return mlir::failure();

    auto attrName = mlir::StringAttr::get(
        top->getContext(), imex::util::attributes::getShapeRangeName());
    top->walk([&](mlir::FunctionOpInterface func) {
      if (func.isExternal())
        return;

      auto &body = func.getFunctionBody();
      assert(!body.empty());
      for (auto [i, arg] : llvm::enumerate(body.front().getArguments())) {
        auto shaped = arg.getType().dyn_cast<mlir::ShapedType>();
        if (!shaped)
          continue;

        auto ind = static_cast<unsigned>(i);
        auto newRange = [&]() -> llvm::Optional<ShapeValue> {
          auto attr = func.getArgAttrOfType<mlir::ArrayAttr>(ind, attrName);
          if (attr)
            return ShapeValue::intersect({shaped}, {attr});

          auto mod = func->getParentOfType<mlir::ModuleOp>();
          if (!mod)
            return llvm::None;

          auto uses = mlir::SymbolTable::getSymbolUses(func, mod);
          if (!uses || !uses->empty())
            return llvm::None;

          return ShapeValue{shaped};
        }();

        if (!newRange)
          continue;

        auto *lattice = getLatticeElement(arg);
        assert(lattice);
        assert(lattice->getValue().isUninitialized());
        propagateIfChanged(lattice, lattice->join(*newRange));
      }
    });

    return mlir::success();
  }
};

class IntegerRangeAnalysisEx : public mlir::dataflow::IntegerRangeAnalysis {
public:
  using IntegerRangeAnalysis::IntegerRangeAnalysis;

  void visitOperation(
      mlir::Operation *op,
      llvm::ArrayRef<const mlir::dataflow::IntegerValueRangeLattice *> operands,
      llvm::ArrayRef<mlir::dataflow::IntegerValueRangeLattice *> results)
      override {
    LLVM_DEBUG(llvm::dbgs() << "IntegerRangeAnalysisEx: Visiting operation: "
                            << *op << "\n");

    if (auto dim = mlir::dyn_cast<mlir::ShapedDimOpInterface>(op)) {
      assert(op->getNumResults() == 1);
      assert(results.size() == 1);

      auto *lattice = results.front();
      auto newRange = [&]() -> llvm::Optional<mlir::ConstantIntRanges> {
        auto state =
            getOrCreateFor<ShapeValueLattice>(op, dim.getShapedValue());
        if (!state)
          return llvm::None;

        auto &shapeVal = state->getValue();
        if (shapeVal.isUninitialized())
          return llvm::None;

        auto index = mlir::getConstantIntValue(dim.getDimension());
        if (!index)
          return llvm::None;

        auto shape = shapeVal.getShape();
        auto indexVal = *index;
        if (indexVal < 0 || indexVal >= static_cast<int64_t>(shape.size()))
          return llvm::None;

        return shape[indexVal];
      }();

      if (newRange) {
        auto changed =
            lattice->join(mlir::dataflow::IntegerValueRange{newRange});
        propagateIfChanged(lattice, changed);
      }
      return;
    }

    mlir::dataflow::IntegerRangeAnalysis::visitOperation(op, operands, results);
  }
};

static bool intersects(mlir::ConstantIntRanges lhs,
                       mlir::ConstantIntRanges rhs) {
  if ((lhs.smax().slt(rhs.smin()) || lhs.smin().sgt(rhs.smax())) &&
      (lhs.umax().ult(rhs.umin()) || lhs.umin().ugt(rhs.umax())))
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
    if (!rhsResult || rhsResult->getValue().isUninitialized())
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
    registry.insert<imex::util::ImexUtilDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "ShapeIntegerRangePropagationPass:\n");
    auto op = getOperation();
    mlir::DataFlowSolver solver;
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<TensorValueAnalysis>();
    solver.load<ShapeValueAnalysis>();
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
