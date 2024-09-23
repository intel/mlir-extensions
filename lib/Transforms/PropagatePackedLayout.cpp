//===- PropagatePackedLayout.cpp - PropagatePackedLayout Pass ---*- C++- *-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains PropagatePackedLayout pass.
///
//===----------------------------------------------------------------------===//

#include "imex/Transforms/Passes.h"

#include <numeric>

#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/XeGPU/IR/XeGPU.h>

namespace imex {
#define GEN_PASS_DEF_PROPAGATEPACKEDLAYOUT
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

namespace {
// Struct describing current layout per mlir::Value.
// Have 3 possible states:
// * Uninitialized (`layout` is empty) - initial state before any layout
// propagation.
// * Valid (`layout` holds some non-null value) - `layout` contains the current
// layout.
// * Invalid (`layout` holds some nullptr) - cannot determine layout, usually
// because of layout conflicts.
struct Layout {
  Layout() = default;
  Layout(mlir::Type l) : layout(l) {}
  Layout(std::nullopt_t) : layout(nullptr) {}

  bool isInitialized() const { return static_cast<bool>(layout); }

  bool isInvalid() const { return isInitialized() && !*layout; }

  mlir::Type getLayout() const { return layout ? *layout : nullptr; }

  void print(llvm::raw_ostream &os) const {
    if (!isInitialized()) {
      os << "uninitialized";
    } else if (isInvalid()) {
      os << "invalid";
    } else {
      os << *layout;
    }
  }

  bool operator==(const Layout &rhs) const { return layout == rhs.layout; }
  bool operator!=(const Layout &rhs) const { return layout != rhs.layout; }

  static Layout meet(const Layout &lhs, const Layout &rhs) {
    if (!lhs.isInitialized())
      return rhs;

    if (!rhs.isInitialized())
      return lhs;

    if (lhs.isInvalid() || rhs.isInvalid() || lhs != rhs)
      return std::nullopt;

    return lhs;
  }

  static Layout join(const Layout &lhs, const Layout &rhs) {
    return meet(lhs, rhs);
  }

  Layout clone(mlir::Type elemetType) const {
    if (!layout)
      return *this;

    auto shaped = mlir::dyn_cast_if_present<mlir::ShapedType>(*layout);
    if (!shaped)
      return *this;

    return Layout(shaped.clone(elemetType));
  }

private:
  std::optional<mlir::Type> layout;
};

struct LayoutLattice : public mlir::dataflow::Lattice<Layout> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutLattice)
  using Lattice::Lattice;

  // This should not be needed, probably some bug upstream.
  mlir::ChangeResult meet(const AbstractSparseLattice &rhs) override {
    return meet(
        static_cast<const mlir::dataflow::Lattice<Layout> &>(rhs).getValue());
  }

  mlir::ChangeResult meet(const Layout &rhs) {
    auto &val = getValue();
    Layout newValue = Layout::meet(val, rhs);
    assert(Layout::meet(newValue, val) == newValue &&
           "expected `meet` to be monotonic");
    assert(Layout::meet(newValue, rhs) == newValue &&
           "expected `meet` to be monotonic");

    // Update the current optimistic value if something changed.
    if (newValue == val)
      return mlir::ChangeResult::NoChange;

    val = newValue;
    return mlir::ChangeResult::Change;
  }
};

static mlir::Type getPackedType(mlir::VectorType vec, int64_t factor) {
  const unsigned axis = 0;
  auto shape = vec.getShape();
  assert(axis < shape.size());
  auto newShape = llvm::to_vector(shape);
  newShape.emplace_back(factor);
  newShape[axis] /= factor;
  return mlir::VectorType::get(newShape, vec.getElementType());
}

static mlir::Type getDpasLayout(mlir::xegpu::DpasOp dpas, mlir::Value arg) {
  auto type = mlir::cast<mlir::VectorType>(arg.getType());
  if (type.getRank() != 2)
    return nullptr;

  auto elementSize = type.getElementType().getIntOrFloatBitWidth() / 8;
  if (elementSize >= 4)
    return nullptr;

  auto factor = 4 / elementSize;

  if (dpas.getRhs() == arg)
    return getPackedType(type, factor);

  return nullptr;
}

static mlir::Type getElementType(mlir::Type type) {
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
  if (!shaped)
    return type;

  return shaped.getElementType();
}

// LayoutAnalysisImpl propagates layout info from SSA uses to defs.
class LayoutAnalysisImpl
    : public mlir::dataflow::SparseBackwardDataFlowAnalysis<LayoutLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  mlir::LogicalResult
  visitOperation(mlir::Operation *op, mlir::ArrayRef<LayoutLattice *> operands,
                 mlir::ArrayRef<const LayoutLattice *> results) override {
    if (mlir::OpTrait::hasElementwiseMappableTraits(op)) {
      Layout layout;
      for (auto &&[res, resLattice] :
           llvm::zip_equal(op->getResults(), results)) {
        layout = layout.clone(getElementType(res.getType()));
        layout = Layout::meet(layout, resLattice->getValue());
      }
      for (auto &&[arg, argLattice] :
           llvm::zip_equal(op->getOperands(), operands)) {
        layout = layout.clone(getElementType(arg.getType()));
        layout = Layout::meet(layout, argLattice->getValue());
      }

      for (auto &&[arg, argLattice] :
           llvm::zip_equal(op->getOperands(), operands)) {
        auto tmpLayout = layout.clone(getElementType(arg.getType()));
        propagateIfChanged(argLattice, argLattice->meet(tmpLayout));
      }

      return mlir::success();
    }

    if (auto dpas = mlir::dyn_cast<mlir::xegpu::DpasOp>(op)) {
      for (auto &&[operand, val] : llvm::zip(operands, dpas.getOperands())) {
        if (auto newType = getDpasLayout(dpas, val)) {
          propagateIfChanged(operand, operand->meet(newType));
        } else {
          propagateIfChanged(operand, operand->meet(std::nullopt));
        }
      }
      return mlir::success();
    }

    // Unknown ops: mark all args as invalid layout (no layout change).
    for (auto operand : operands)
      propagateIfChanged(operand, operand->meet(std::nullopt));
    return mlir::success();
  }

  void visitBranchOperand(mlir::OpOperand &operand) override {}

  void visitCallOperand(mlir::OpOperand &operand) override {}

  void setToExitState(LayoutLattice *lattice) override {
    (void)lattice->meet(std::nullopt);
  }
};

class LayoutAnalysis {
public:
  LayoutAnalysis() = default;

  mlir::LogicalResult run(mlir::Operation *op) {
    mlir::SymbolTableCollection symbolTable;

    // These analyses are needed for the data propagating properly (e.g. see
    // LivenessAnalysis upstream).
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<mlir::dataflow::SparseConstantPropagation>();

    solver.load<LayoutAnalysisImpl>(symbolTable);
    return solver.initializeAndRun(op);
  }

  mlir::Type getLayout(mlir::Value val) {
    auto *state = solver.lookupState<LayoutLattice>(val);
    if (!state)
      return nullptr;

    return state->getValue().getLayout();
  }

private:
  mlir::DataFlowSolver solver;
};
} // namespace

static std::pair<unsigned, int64_t> getVNNIInfo(mlir::ShapedType srcType,
                                                mlir::ShapedType dstType) {
  auto srcShape = srcType.getShape();
  auto dstShape = dstType.getShape();
  if (srcShape.size() == 3 && dstShape.size() == 2)
    return getVNNIInfo(dstType, srcType);

  if (srcShape.size() == 2 && dstShape.size() == 3) {
    auto factor = dstShape[2];
    if (srcShape[1] == dstShape[1] && srcShape[0] / factor == dstShape[0])
      return {0, factor};

    if (srcShape[0] == dstShape[0] && srcShape[1] / factor == dstShape[1])
      return {1, factor};
  }
  llvm_unreachable("Unsupported shapes");
}

static llvm::SmallVector<int64_t>
getVNNIShuffleIndices(mlir::ShapedType srcType, mlir::ShapedType dstType) {
  auto numElements = srcType.getNumElements();
  llvm::SmallVector<int64_t> ret(numElements, 0);
  auto &&[axis, factor] = getVNNIInfo(srcType, dstType);
  if (axis == 1) {
    // with axis == 1 it's noop.
    std::iota(ret.begin(), ret.end(), 0);
    return ret;
  }
  assert(axis == 0);
  bool toPacked = (srcType.getRank() < dstType.getRank());
  auto srcShape = srcType.getShape();
  auto dstShape = dstType.getShape();
  if (toPacked) {
    // Convert from contiguous layout to VNNI packed, e.g. from
    // `vector<16x16xf16>` to `vector<8x16x2xf16>`.
    assert(srcShape.size() == 2);
    assert(dstShape.size() == 3);
    // To arrange the data in VNNI format, the shuffle indices must satisfy
    // following mapping.
    // [i, j, k] => i * dstShape[1] * dstShape[2] + j + k * dstShape[1]
    int shuffleIndex = 0;
    for (unsigned i = 0; i < dstShape[0]; ++i) {
      for (unsigned j = 0; j < dstShape[1]; ++j) {
        for (unsigned k = 0; k < dstShape[2]; ++k) {
          ret[shuffleIndex++] =
              i * dstShape[1] * dstShape[2] + j + k * dstShape[1];
        }
      }
    }
  } else {
    // Convert from VNNI packed to contiguous layout, e.g. from
    // `vector<8x16x2xf16>` to `vector<16x16xf16>`.
    assert(srcShape.size() == 3);
    assert(dstShape.size() == 2);
    // To arrange the data in contiguous format, the shuffle indices must
    // satisfy following mapping, i.e. do the reverse mapping of the above
    // i * srcShape[1] * srcShape[2] + j + k * srcShape[1] => [i, j, k]
    int shuffleIndex = 0;
    for (unsigned i = 0; i < srcShape[0]; ++i) {
      for (unsigned j = 0; j < srcShape[1]; ++j) {
        for (unsigned k = 0; k < srcShape[2]; ++k) {
          ret[i * srcShape[1] * srcShape[2] + j + k * srcShape[1]] =
              shuffleIndex++;
        }
      }
    }
  }

  return ret;
}

static std::pair<mlir::Value, mlir::Operation *>
makeCast(mlir::OpBuilder &builder, mlir::Value src, mlir::Type srcType,
         mlir::Type dstType) {
  if (srcType == dstType)
    return {src, nullptr};

  auto srcVecType = mlir::cast<mlir::VectorType>(srcType);
  auto dstVecType = mlir::cast<mlir::VectorType>(dstType);
  auto numElements = srcVecType.getNumElements();
  assert(numElements == dstVecType.getNumElements());
  auto tmpVecType =
      mlir::VectorType::get(numElements, srcVecType.getElementType());

  auto loc = src.getLoc();

  auto root = builder.create<mlir::vector::ShapeCastOp>(loc, tmpVecType, src);
  mlir::Value tmp = root;

  tmp = builder.create<mlir::vector::ShuffleOp>(
      loc, tmp, tmp,
      builder.getDenseI64ArrayAttr(
          getVNNIShuffleIndices(srcVecType, dstVecType)));

  return {builder.create<mlir::vector::ShapeCastOp>(loc, dstVecType, tmp),
          root};
}

static std::pair<mlir::Value, mlir::Operation *>
makeCast(mlir::OpBuilder &builder, mlir::Value src, mlir::Type dstType) {
  return makeCast(builder, src, src.getType(), dstType);
}

static bool canUpdateElemetwiseInplace(mlir::TypeRange operands,
                                       mlir::TypeRange results) {
  mlir::ShapedType shaped;
  for (auto range : {operands, results}) {
    for (auto t : range) {
      auto s = mlir::dyn_cast<mlir::ShapedType>(t);
      if (!s)
        return false;

      if (!shaped) {
        shaped = s;
        continue;
      }
      if (shaped.getShape() != s.getShape())
        return false;
    }
  }
  return true;
}

static void updateUnknownOp(mlir::OpBuilder &builder, mlir::Operation &op,
                            mlir::TypeRange operands, mlir::TypeRange results) {
  builder.setInsertionPoint(&op);
  for (auto &&[arg, dstType] : llvm::zip_equal(op.getOpOperands(), operands)) {
    auto val = arg.get();
    auto &&[newArg, root] = makeCast(builder, val, dstType, val.getType());
    if (newArg == val)
      continue;

    arg.set(newArg);
  }
  builder.setInsertionPointAfter(&op);
  for (auto &&[res, dstType] : llvm::zip_equal(op.getResults(), results)) {
    auto &&[newRes, root] = makeCast(builder, res, dstType);
    if (newRes == res)
      continue;

    res.replaceAllUsesExcept(newRes, root);
  }
}

static void updateElemenwiseOp(mlir::OpBuilder &builder, mlir::Operation &op,
                               mlir::TypeRange operands,
                               mlir::TypeRange results) {
  if (canUpdateElemetwiseInplace(operands, results)) {
    for (auto [res, dstType] : llvm::zip_equal(op.getResults(), results))
      res.setType(dstType);
  } else {
    updateUnknownOp(builder, op, operands, results);
  }
}

static void updateLoadOp(mlir::OpBuilder &builder, mlir::xegpu::LoadNdOp op,
                         mlir::TypeRange operands, mlir::TypeRange results) {
  assert(results.size() == 1);
  auto srcType = mlir::cast<mlir::VectorType>(op.getType());
  auto dstType = mlir::cast<mlir::VectorType>(results.front());
  if (srcType == dstType)
    return;

  auto &&[axis, factor] = getVNNIInfo(srcType, dstType);
  op.getResult().setType(dstType);
  if (axis == 0)
    op.setPacked(true);
}

static void updateDpasOp(mlir::OpBuilder &builder, mlir::xegpu::DpasOp op,
                         mlir::TypeRange operands, mlir::TypeRange results) {
  builder.setInsertionPoint(op);
  for (auto &&[newType, arg] : llvm::zip(operands, op->getOpOperands())) {
    auto val = arg.get();
    auto packedType = getDpasLayout(op, val);
    if (!packedType || packedType == newType)
      continue;

    auto &&[newArg, root] = makeCast(builder, val, packedType);
    if (newArg == val)
      continue;

    arg.set(newArg);
  }

  builder.setInsertionPointAfter(op);
  for (auto &&[res, dstType] : llvm::zip_equal(op->getResults(), results)) {
    auto &&[newRes, root] = makeCast(builder, res, dstType);
    if (newRes == res)
      continue;

    res.replaceAllUsesExcept(newRes, root);
  }
}

static void updateOpTypes(mlir::OpBuilder &builder, mlir::Operation &op,
                          mlir::TypeRange operands, mlir::TypeRange results) {
  // Ignore shape casts as they are generated by the conversion itself.
  // Ignore RegionBranchOpInterface as it handled in `updateBlockTypes`.
  if (mlir::isa<mlir::vector::ShapeCastOp, mlir::RegionBranchOpInterface,
                mlir::RegionBranchTerminatorOpInterface>(op))
    return;

  if (mlir::OpTrait::hasElementwiseMappableTraits(&op))
    return updateElemenwiseOp(builder, op, operands, results);
  if (auto load = mlir::dyn_cast<mlir::xegpu::LoadNdOp>(op))
    return updateLoadOp(builder, load, operands, results);
  if (auto dpas = mlir::dyn_cast<mlir::xegpu::DpasOp>(op))
    return updateDpasOp(builder, dpas, operands, results);

  updateUnknownOp(builder, op, operands, results);
}

static void handleBranchOpInterface(mlir::OpBuilder &builder,
                                    mlir::Block &block,
                                    mlir::RegionBranchOpInterface branch,
                                    mlir::TypeRange argsTypes) {
  builder.setInsertionPointToStart(&block);

  mlir::Operation *op = branch.getOperation();
  llvm::SmallVector<mlir::RegionSuccessor> successors;
  llvm::SmallVector<mlir::Attribute> operands(op->getNumOperands(), nullptr);
  branch.getEntrySuccessorRegions(operands, successors);

  for (mlir::RegionSuccessor &successor : successors) {
    if (block.getParent() != successor.getSuccessor())
      continue;

    mlir::OperandRange operands = branch.getEntrySuccessorOperands(successor);
    mlir::ValueRange inputs = successor.getSuccessorInputs();
    for (auto [arg, input] : llvm::zip(operands, inputs)) {
      auto idx = mlir::cast<mlir::BlockArgument>(input).getArgNumber();
      mlir::Type dstType = argsTypes[idx];
      if (dstType == arg.getType()) {
        input.setType(dstType);
        continue;
      }

      auto &&[newArg, root] = makeCast(builder, arg, dstType);
      if (newArg == arg)
        continue;

      arg.replaceAllUsesExcept(newArg, root);
    }
  }

  auto terminator = mlir::cast<mlir::RegionBranchTerminatorOpInterface>(
      block.getTerminator());
  mlir::SmallVector<mlir::Attribute> operandAttributes(
      terminator->getNumOperands(), nullptr);

  successors.clear();
  terminator.getSuccessorRegions(operandAttributes, successors);

  for (const mlir::RegionSuccessor &successor : successors) {
    if (!successor.isParent())
      continue;

    mlir::ValueRange inputs = successor.getSuccessorInputs();
    mlir::OperandRange operands = terminator.getSuccessorOperands(successor);
    for (auto [operand, input] : llvm::zip(operands, inputs)) {
      input.setType(operand.getType());
    }
  }
}

static void updateBlockTypes(mlir::OpBuilder &builder, mlir::Block &block,
                             mlir::TypeRange args) {
  if (auto iface = mlir::dyn_cast_if_present<mlir::RegionBranchOpInterface>(
          block.getParentOp()))
    return handleBranchOpInterface(builder, block, iface, args);

  builder.setInsertionPointToStart(&block);
  for (auto &&[arg, dstType] : llvm::zip_equal(block.getArguments(), args)) {
    auto &&[newArg, root] = makeCast(builder, arg, dstType);
    if (newArg == arg)
      continue;

    arg.replaceAllUsesExcept(newArg, root);
  }
}

namespace imex {

struct PropagatePackedLayoutPass final
    : public imex::impl::PropagatePackedLayoutBase<PropagatePackedLayoutPass> {

  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    LayoutAnalysis analysis;
    if (mlir::failed(analysis.run(op)))
      return signalPassFailure();

    auto getLayout = [&](mlir::Value val) -> mlir::Type {
      auto t = analysis.getLayout(val);
      if (!t)
        return val.getType();

      return t;
    };

    mlir::OpBuilder builder(&getContext());
    llvm::SmallVector<mlir::Type> operands;
    op->walk<mlir::WalkOrder::PreOrder>([&](mlir::Block *block) {
      // Iterate block ops in reverse so op is updated before it's operands.
      for (mlir::Operation &innerOp : llvm::reverse(block->getOperations())) {
        operands.clear();
        for (auto args : {mlir::ValueRange(innerOp.getOperands()),
                          mlir::ValueRange(innerOp.getResults())}) {
          for (auto arg : args)
            operands.emplace_back(getLayout(arg));
        }
        mlir::TypeRange range(operands);
        auto numOperands = innerOp.getNumOperands();
        updateOpTypes(builder, innerOp, range.take_front(numOperands),
                      range.drop_front(numOperands));
      }
      operands.clear();
      for (auto arg : block->getArguments())
        operands.emplace_back(getLayout(arg));
      updateBlockTypes(builder, *block, operands);
    });
  }
};
} // namespace imex

std::unique_ptr<mlir::Pass> imex::createPropagatePackedLayoutPass() {
  return std::make_unique<PropagatePackedLayoutPass>();
}
