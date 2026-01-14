//===- VnniTransformation.cpp - VnniTransformation Pass ---*- C++- *-===//
//
// Copyright 2024 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains VnniTransformation pass.
///
//===----------------------------------------------------------------------===//
#include <llvm/Support/Debug.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/XeGPU/IR/XeGPU.h>
#include <mlir/IR/BuiltinTypes.h>

#include "imex/Transforms/Passes.h"
#include "imex/Utils/XeCommon.h"

#include <optional>

namespace imex {
#define GEN_PASS_DEF_VNNITRANSFORMATION
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

using namespace imex;

namespace {
// Struct describing current layout per mlir::Value.
// Have 3 possible states:
// * Uninitialized (`layout` is empty) - initial state before any layout
// propagation.
// * Valid (`layout` holds some non-null value) - `layout` contains the current
// layout.
// * Invalid (`layout` holds some nullptr) - cannot determine layout, usually
// because of layout conflicts.
class Layout {
public:
  Layout() = default;
  Layout(bool vnni) : vnniLayout(vnni) {}
  Layout(std::nullptr_t) : vnniLayout(false) {}
  bool getLayout() const { return vnniLayout.value_or(false); }

  bool isInitialized() const { return vnniLayout.has_value(); }

  void print(llvm::raw_ostream &os) const {
    if (!isInitialized()) {
      os << "uninitialized";
    } else {
      os << (vnniLayout.value() ? "vnni" : "non-vnni");
    }
  }

  bool operator==(const Layout &rhs) const {
    return vnniLayout == rhs.vnniLayout;
  }
  bool operator!=(const Layout &rhs) const {
    return vnniLayout != rhs.vnniLayout;
  }

  // and operation for lattice.
  static Layout meet(const Layout &lhs, const Layout &rhs) {
    if (!lhs.isInitialized())
      return rhs;
    if (!rhs.isInitialized())
      return lhs;
    return Layout(lhs.getLayout() && rhs.getLayout());
  }

  // or operation for lattice.
  static Layout join(const Layout &lhs, const Layout &rhs) {
    return meet(lhs, rhs);
  }

private:
  std::optional<bool> vnniLayout;
};

class LayoutLattice : public mlir::dataflow::Lattice<Layout> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutLattice)
  using Lattice::Lattice;

  // This should not be needed, probably some bug upstream.
  mlir::ChangeResult meet(const AbstractSparseLattice &rhs) override {
    return meet(static_cast<const LayoutLattice &>(rhs).getValue());
  }

  mlir::ChangeResult meet(const Layout &rhs) {
    auto &val = getValue();
    Layout newValue = Layout::meet(val, rhs);

    // Update the current optimistic value if something changed.
    if (newValue == val)
      return mlir::ChangeResult::NoChange;

    val = newValue;
    return mlir::ChangeResult::Change;
  }
};

static bool isVNNIApplicable(mlir::Type type) {
  auto vecTy = mlir::dyn_cast<mlir::VectorType>(type);

  // VNNI transform only available for 2D vectors.
  if (!vecTy || vecTy.getRank() != 2)
    return false;
  auto elemTy = vecTy.getElementType();
  if (!elemTy.isIntOrFloat())
    return false;
  auto factor = getVnniFactor(vecTy.getElementType());
  auto shape = vecTy.getShape();
  // factor == 1 means 32-bit data, and no need to apply VNNI.
  return factor > 1 && shape[0] % factor == 0;
}

// LayoutAnalysisImpl propagates layout info from SSA uses to defs.
class LayoutAnalysisImpl
    : public mlir::dataflow::SparseBackwardDataFlowAnalysis<LayoutLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  mlir::LogicalResult
  visitOperation(mlir::Operation *op, mlir::ArrayRef<LayoutLattice *> operands,
                 mlir::ArrayRef<const LayoutLattice *> results) override {
    // the B operand of a dpas operation is always in vnni layout
    // and it is the start point of the layout propagation
    if (auto dpas = mlir::dyn_cast<mlir::xegpu::DpasOp>(op)) {
      // for A operand, it cannot be in vnni format
      propagateIfChanged(operands[0], operands[0]->meet(Layout(false)));

      // for B operand, it must be in vnni format if applicable
      Layout vnni(isVNNIApplicable(dpas.getRhs().getType()));
      propagateIfChanged(operands[1], operands[1]->meet(vnni));

      if (operands.size() == 3) {
        // for C operand, it cannot be in vnni format
        propagateIfChanged(operands[2], operands[2]->meet(Layout(false)));
      }
      return mlir::success();
    }

    // for non-cast elementwise ops only. Propagation is stopped
    // when meet an cast op, e.g., truncf, in which source and result
    // needs different vnni factors. An exception is bitcast op, which
    // source and results has the same bitwidth.
    if (mlir::OpTrait::hasElementwiseMappableTraits(op)) {
      // stop propagation for cast ops that are not guaranteed
      // to have same bitwidth between source and result.
      if (mlir::isa<mlir::CastOpInterface>(op)) {
        auto srcTy = mlir::getElementTypeOrSelf(op->getOperand(0));
        auto dstTy = mlir::getElementTypeOrSelf(op->getResult(0));
        if (!srcTy.isIntOrFloat() || !dstTy.isIntOrFloat() ||
            srcTy.getIntOrFloatBitWidth() != dstTy.getIntOrFloatBitWidth()) {
          for (auto operand : operands)
            propagateIfChanged(operand, operand->join(Layout(false)));
          return mlir::success();
        }
      }

      Layout layout;

      // if the op has results, initial the layout to be vnni
      // because meet is going to do "and" operation. It will
      // be reset by results layout.
      if (results.size())
        layout = Layout(true);

      // for elementwise ops, propagate only if every use needs vnni.
      for (auto &&res : results)
        layout = Layout::meet(layout, res->getValue());

      // propagate only when all results are initialized.
      if (layout.isInitialized()) {
        // make sure all operands are vnni-transformable.
        // if not, mark all operands as non-vnni layout.
        for (auto &&[opr, lattice] :
             llvm::zip_equal(op->getOperands(), operands)) {
          layout = Layout::meet(layout, lattice->getValue());
          layout =
              Layout::meet(layout, Layout(isVNNIApplicable(opr.getType())));
        }

        // propagate operands layout.
        for (auto &&lattice : operands)
          propagateIfChanged(lattice, lattice->meet(layout));
      }
      return mlir::success();
    }

    if (auto extractStrideSliceOp =
            mlir::dyn_cast<mlir::vector::ExtractStridedSliceOp>(op)) {
      auto srcTy = extractStrideSliceOp.getSourceVectorType();
      Layout layout = results[0]->getValue();
      if (layout.isInitialized()) {
        layout = Layout::meet(layout, Layout(isVNNIApplicable(srcTy)));
        propagateIfChanged(operands[0], operands[0]->meet(layout));
      }
      return mlir::success();
    }

    if (auto extractOp = mlir::dyn_cast<mlir::vector::ExtractOp>(op)) {
      auto src = extractOp.getSource();
      auto srcTy = src.getType();
      Layout layout = results[0]->getValue();
      auto loadOp = src.getDefiningOp<mlir::xegpu::LoadNdOp>();
      // only interested if the source is a LoadNdOp.
      if (layout.isInitialized() && srcTy.getRank() == 3 && loadOp) {
        auto shape = srcTy.getShape().take_back(2);
        auto vecTy = mlir::VectorType::get(shape, srcTy.getElementType());
        layout = Layout::meet(layout, Layout(isVNNIApplicable(vecTy)));
        propagateIfChanged(operands[0], operands[0]->meet(layout));
      }
      return mlir::success();
    }

    // Unknown ops: mark all args as non-vnni layout (no layout change).
    for (auto operand : operands)
      propagateIfChanged(operand, operand->join(Layout(false)));

    return mlir::success();
  }

  void visitBranchOperand(mlir::OpOperand &operand) override {}

  void visitCallOperand(mlir::OpOperand &operand) override {}

  void visitNonControlFlowArguments(
      mlir::RegionSuccessor &successor,
      mlir::ArrayRef<mlir::BlockArgument> arguments) override{};

  void setToExitState(LayoutLattice *lattice) override {
    (void)lattice->meet(false);
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

  bool getLayout(mlir::Value val) {
    auto *state = solver.lookupState<LayoutLattice>(val);
    if (!state)
      return false;

    return state->getValue().getLayout();
  }

private:
  mlir::DataFlowSolver solver;
};
} // namespace

static void applyVnniTransformOnResults(mlir::OpBuilder &builder,
                                        mlir::Operation *op,
                                        LayoutAnalysis &analysis) {
  builder.setInsertionPointAfter(op);
  for (auto &&res : op->getResults()) {
    if (analysis.getLayout(res)) {
      // if the layout is vnni it must be a vector
      auto cast = mlir::cast<mlir::TypedValue<mlir::VectorType>>(res);
      auto &&[newRes, root] = applyVnniTransform(builder, cast);
      res.replaceAllUsesExcept(newRes, root);
    }
  }
}

// for Unknown ops, we simply insert the vectorshuffles to do in-register vnni
// transform for result values if needed, since we don't know the semantics of
// the op, and whether it is safe to apply vnni transform on operands too.
static void updateUnknownOp(mlir::OpBuilder &builder, mlir::Operation &op,
                            LayoutAnalysis &analysis) {
  // Ignore ops that has packed attribute, since they are inserted by the pass.
  if (op.hasAttr("packed"))
    return;
  applyVnniTransformOnResults(builder, &op, analysis);
}

// for elementwise ops, if all operands can be in vnni layout, we only need to
// update the data type of operands and results to reveal the vnni layout.
// The defining ops of operands will be updated to do actual vnni transform.
// Otherwise, we need to apply vnni transform on results if the corresponding
// value need to be in vnni layout.
static void updateElemenwiseOp(mlir::OpBuilder &builder, mlir::Operation &op,
                               LayoutAnalysis &analysis) {

  // vnni layout is only applicable when all operands and results are in vnni
  bool doVnni = true;
  for (auto range : {mlir::ValueRange(op.getOperands()),
                     mlir::ValueRange(op.getResults())}) {
    for (auto val : range)
      doVnni &= analysis.getLayout(val);
  }

  if (doVnni) {
    for (auto res : op.getResults()) {
      auto vecTy = mlir::cast<mlir::VectorType>(res.getType());
      res.setType(getPackedType(vecTy));
    }
  } else {
    applyVnniTransformOnResults(builder, &op, analysis);
  }
}

// in most cases, we don't need to do anything for dpas op, since the vnni
// layout of B operand applied by its defining op. However, if the B operand
// has multiple uses and the rest uses don't use vnni, then its defining op
// cannot apply the vnni transform on the B operand, and we need to apply
// vnni transform locally.
static void updateDpasOp(mlir::OpBuilder &builder, mlir::xegpu::DpasOp &op,
                         LayoutAnalysis &analysis) {
  auto rhs = op.getRhs();
  // B operand of DPAS has multiple uses and
  // the rest uses don't use vnni.
  if (!analysis.getLayout(rhs) && isVNNIApplicable(rhs.getType())) {
    builder.setInsertionPoint(op);
    auto cast = mlir::cast<mlir::TypedValue<mlir::VectorType>>(rhs);
    auto &&[newRhs, root] = applyVnniTransform(builder, cast);
    op->getOpOperand(1).set(newRhs);
  }
  // apply vnni transform on results if needed (when the result is used
  // as B operand of another dpas op). Rarly happens, just in case.
  applyVnniTransformOnResults(builder, op.getOperation(), analysis);
}

// for load op, we only need to set the packed flag and the result type
// if the layout of its result need to be vnni.
static void updateLoadOp(mlir::OpBuilder &builder, mlir::xegpu::LoadNdOp &op,
                         LayoutAnalysis &analysis) {
  // for load, we simply set the packed flag and the result type
  // if the layout of its result need to be vnni.
  auto result = op.getResult();
  if (analysis.getLayout(result)) {
    op.setPacked(true);
    auto vecTy = mlir::cast<mlir::VectorType>(result.getType());
    result.setType(getPackedType(vecTy));
  }
}

// for extract op, it is similar to elementwise ops, we only need to update
// data types of source vector and result vector if they need to be in vnni.
// Otherwise, we need to apply vnni transform on the result if needed
static void updateExtractOp(mlir::OpBuilder &builder,
                            mlir::vector::ExtractOp &op,
                            LayoutAnalysis &analysis) {
  auto src = op.getSource();
  auto res = op.getResult();
  if (analysis.getLayout(src) && analysis.getLayout(res)) {
    auto packedResType =
        getPackedType(mlir::cast<mlir::VectorType>(op.getType()));
    res.setType(packedResType);
  } else if (analysis.getLayout(res)) {
    applyVnniTransformOnResults(builder, op.getOperation(), analysis);
  }
}

// similar to extract op, except that we also need to update the
// offsets, sizes and strides
static void updateExtractStrideSliceOp(mlir::OpBuilder &builder,
                                       mlir::vector::ExtractStridedSliceOp &op,
                                       LayoutAnalysis &analysis) {
  auto src = op.getSource();
  auto result = op.getResult();
  // simply to update offsets and strides when both source and result
  // are in vnni format.
  if (analysis.getLayout(src) && analysis.getLayout(result)) {
    auto to_vector = [](auto range) {
      llvm::SmallVector<int64_t> ret;
      for (auto &&val : range)
        ret.push_back(mlir::cast<mlir::IntegerAttr>(val).getInt());
      return ret;
    };

    auto resTy = mlir::cast<mlir::VectorType>(result.getType());
    auto factor = getVnniFactor(resTy.getElementType());
    auto offsets = to_vector(op.getOffsets());
    auto strides = to_vector(op.getStrides());
    auto sizes = to_vector(op.getSizes());
    offsets[0] /= factor;
    sizes[0] /= factor;
    offsets.push_back(0);
    sizes.push_back(factor);
    strides.push_back(1);

    auto sizesAttr = builder.getI64ArrayAttr(sizes);
    auto offsetsAttr = builder.getI64ArrayAttr(offsets);
    auto stridesAttr = builder.getI64ArrayAttr(strides);
    op.setSizesAttr(sizesAttr);
    op.setStridesAttr(stridesAttr);
    op.setOffsetsAttr(offsetsAttr);
    op.getResult().setType(getPackedType(resTy));
  } else if (analysis.getLayout(result)) {
    applyVnniTransformOnResults(builder, op.getOperation(), analysis);
  }
}

// handle terminal ops, e.g., scf.Yield. Update
// the types of its successor inputs if successor
// operands needs vnni format.
static void handleBranchTerminatorOpInterface(
    mlir::OpBuilder &builder,
    mlir::RegionBranchTerminatorOpInterface terminator,
    LayoutAnalysis &analysis) {

  if (!mlir::isa<mlir::RegionBranchOpInterface>(terminator->getParentOp()))
    return;

  llvm::SmallVector<mlir::RegionSuccessor> successors;
  llvm::SmallVector<mlir::Attribute> operands(terminator->getNumOperands(),
                                              nullptr);
  terminator.getSuccessorRegions(operands, successors);

  for (mlir::RegionSuccessor &successor : successors) {
    if (!successor.isParent())
      continue;

    mlir::OperandRange operands = terminator.getSuccessorOperands(successor);
    mlir::ValueRange inputs = successor.getSuccessorInputs();
    for (auto [arg, inp] : llvm::zip(operands, inputs)) {
      if (analysis.getLayout(arg)) {
        auto vecTy = mlir::cast<mlir::VectorType>(arg.getType());
        auto packedTy = getPackedType(vecTy);
        inp.setType(packedTy);
      }
    }
  }
}

// handle REgionBranchOps, e.g., scf.for. Update the
// region argument types, if the argument needs to be
// in vnni format, but the initArg is not, a vnni
// transform is applied on the initArg.
static void handleBranchOpInterface(mlir::OpBuilder &builder,
                                    mlir::RegionBranchOpInterface branch,
                                    LayoutAnalysis &analysis) {
  mlir::Operation *op = branch.getOperation();
  llvm::SmallVector<mlir::RegionSuccessor> successors;
  llvm::SmallVector<mlir::Attribute> operands(op->getNumOperands(), nullptr);
  branch.getEntrySuccessorRegions(operands, successors);

  for (mlir::RegionSuccessor &successor : successors) {
    if (successor.isParent())
      continue;

    mlir::OperandRange operands = branch.getEntrySuccessorOperands(successor);
    mlir::ValueRange inputs = successor.getSuccessorInputs();

    for (auto [arg, input] : llvm::zip(operands, inputs)) {
      if (analysis.getLayout(input)) {
        auto vecTy = mlir::cast<mlir::VectorType>(input.getType());
        auto packedTy = getPackedType(vecTy);
        input.setType(packedTy);
        if (!analysis.getLayout(arg)) {
          builder.setInsertionPointAfterValue(arg);
          auto cast = mlir::cast<mlir::TypedValue<mlir::VectorType>>(arg);
          auto &&[newArg, root] = applyVnniTransform(builder, cast);
          arg.replaceAllUsesExcept(newArg, root);
        }
      }
    }
  }
}

static void updateBlockTypes(mlir::OpBuilder &builder, mlir::Block &block,
                             LayoutAnalysis &analysis) {
  if (!mlir::isa<mlir::RegionBranchOpInterface>(block.getParentOp())) {
    builder.setInsertionPointToStart(&block);
    for (auto &&arg : block.getArguments()) {
      if (analysis.getLayout(arg)) {
        auto cast = mlir::cast<mlir::TypedValue<mlir::VectorType>>(arg);
        auto &&[newArg, root] = applyVnniTransform(builder, cast);
        arg.replaceAllUsesExcept(newArg, root);
      }
    }
  }
}

namespace imex {

struct VnniTransformationPass final
    : public imex::impl::VnniTransformationBase<VnniTransformationPass> {

  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    LayoutAnalysis analysis;
    if (mlir::failed(analysis.run(op)))
      return signalPassFailure();

    mlir::OpBuilder builder(&getContext());
    llvm::SmallVector<mlir::Type> operands;
    // process ops in post-order so that the layout info is
    // used before being destroyed.
    op->walk([&](mlir::Block *block) {
      // Iterate block ops in reverse so op is updated before it's operands.
      for (mlir::Operation &op : llvm::reverse(block->getOperations())) {
        if (auto terminator =
                mlir::dyn_cast<mlir::RegionBranchTerminatorOpInterface>(op)) {
          handleBranchTerminatorOpInterface(builder, terminator, analysis);
          continue;
        }

        if (auto iface = mlir::dyn_cast<mlir::RegionBranchOpInterface>(op)) {
          handleBranchOpInterface(builder, iface, analysis);
          continue;
        }

        if (auto dpas = mlir::dyn_cast<mlir::xegpu::DpasOp>(op)) {
          updateDpasOp(builder, dpas, analysis);
          continue;
        }

        if (auto load = mlir::dyn_cast<mlir::xegpu::LoadNdOp>(op)) {
          updateLoadOp(builder, load, analysis);
          continue;
        }

        if (auto extractStridedSliceOp =
                mlir::dyn_cast<mlir::vector::ExtractStridedSliceOp>(op)) {
          updateExtractStrideSliceOp(builder, extractStridedSliceOp, analysis);
          continue;
        }

        if (auto extractOp = mlir::dyn_cast<mlir::vector::ExtractOp>(op)) {
          updateExtractOp(builder, extractOp, analysis);
          continue;
        }

        // This is for handling all elementwise ops, e.g. arith.addi, etc.
        if (mlir::OpTrait::hasElementwiseMappableTraits(&op)) {
          updateElemenwiseOp(builder, op, analysis);
          continue;
        }

        updateUnknownOp(builder, op, analysis);
      }

      updateBlockTypes(builder, *block, analysis);
    });
  }
};
} // namespace imex

std::unique_ptr<mlir::Pass> imex::createVnniTransformationPass() {
  return std::make_unique<VnniTransformationPass>();
}
