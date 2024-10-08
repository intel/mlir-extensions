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
#include "imex/Utils/DebugUtils.h"

#include <numeric>
#include <optional>

namespace imex {
#define GEN_PASS_DEF_VNNITRANSFORMATION
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

static int getVnniFactor(mlir::Type elemTy) {
  assert(elemTy.isIntOrFloat() && "Only integer and float types supported");
  return 32 / elemTy.getIntOrFloatBitWidth();
}

static bool isVNNIApplicable(mlir::Type type) {
  auto vecTy = mlir::dyn_cast<mlir::VectorType>(type);

  // VNNI transform only available for 2D vectors.
  if (!vecTy || vecTy.getRank() != 2)
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

    if (mlir::OpTrait::hasElementwiseMappableTraits(op)) {
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
      auto src = extractOp.getVector();
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

static mlir::VectorType getPackedType(mlir::VectorType vecTy) {
  auto shape = vecTy.getShape().vec();
  auto factor = getVnniFactor(vecTy.getElementType());
  unsigned axis = shape.size() == 3 ? 1 : 0;

  // Only 2D/3D vector supported and The vector size
  // must be divisible by the factor
  if ((shape.size() != 2 && shape.size() != 3) || !factor ||
      shape[axis] % factor != 0)
    return nullptr;

  shape.emplace_back(factor);
  shape[axis] /= factor;
  return mlir::VectorType::get(shape, vecTy.getElementType());
}

static llvm::SmallVector<int64_t>
getVNNIShuffleIndices(mlir::VectorType srcType) {
  auto numElements = srcType.getNumElements();
  llvm::SmallVector<int64_t> ret(numElements, 0);
  auto dstType = getPackedType(srcType);
  auto dstShape = dstType.getShape();
  // Convert from contiguous layout to VNNI packed, e.g. from
  // `vector<16x16xf16>` to `vector<8x16x2xf16>`.
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
  return ret;
}

static std::pair<mlir::Value, mlir::Operation *>
applyVnniTransform(mlir::OpBuilder &builder,
                   mlir::TypedValue<mlir::VectorType> src) {
  assert(src && "value must be non-null");
  auto loc = src.getLoc();
  auto srcTy = src.getType();
  auto elems = srcTy.getNumElements();
  auto elemTy = srcTy.getElementType();
  auto linearVecTy = mlir::VectorType::get(elems, elemTy);
  auto root = builder.create<mlir::vector::ShapeCastOp>(loc, linearVecTy, src);
  auto mask = getVNNIShuffleIndices(srcTy);
  auto shuffle = builder.create<mlir::vector::ShuffleOp>(loc, root, root, mask);
  auto packedTy = getPackedType(srcTy);
  auto cast = builder.create<mlir::vector::ShapeCastOp>(loc, packedTy, shuffle);
  // for convenience of load+transpose optimization, add packed attribute
  // to indicate these ops are used to do vnni transform.
  root.getOperation()->setAttr("packed", builder.getUnitAttr());
  shuffle.getOperation()->setAttr("packed", builder.getUnitAttr());
  cast.getOperation()->setAttr("packed", builder.getUnitAttr());

  return {cast, root};
}

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
  if (!analysis.getLayout(rhs)) {
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
  auto src = op.getVector();
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
  auto src = op.getVector();
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
      } else {
        auto cast = mlir::cast<mlir::TypedValue<mlir::VectorType>>(arg);
        auto &&[newArg, root] = applyVnniTransform(builder, cast);
        arg.replaceAllUsesExcept(newArg, root);
      }
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
                             LayoutAnalysis &analysis) {
  if (auto iface = mlir::dyn_cast_if_present<mlir::RegionBranchOpInterface>(
          block.getParentOp())) {
    llvm::SmallVector<mlir::Type> types;
    for (auto arg : block.getArguments()) {
      auto argTy = arg.getType();
      if (!analysis.getLayout(arg)) {
        types.push_back(argTy);
      } else {
        auto vecTy = mlir::cast<mlir::VectorType>(argTy);
        auto packedTy = getPackedType(vecTy);
        types.push_back(packedTy);
      }
    }
    return handleBranchOpInterface(builder, block, iface, types);
  }

  builder.setInsertionPointToStart(&block);
  for (auto &&arg : block.getArguments()) {
    if (analysis.getLayout(arg)) {
      auto cast = mlir::cast<mlir::TypedValue<mlir::VectorType>>(arg);
      auto &&[newArg, root] = applyVnniTransform(builder, cast);
      arg.replaceAllUsesExcept(newArg, root);
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
    op->walk<mlir::WalkOrder::PreOrder>([&](mlir::Block *block) {
      // Iterate block ops in reverse so op is updated before it's operands.
      for (mlir::Operation &op : llvm::reverse(block->getOperations())) {
        // Ignore shape casts as they are generated by the conversion itself.
        // Ignore RegionBranchOpInterface as it handled in `updateBlockTypes`.
        if (mlir::isa<mlir::vector::ShapeCastOp, mlir::RegionBranchOpInterface,
                      mlir::RegionBranchTerminatorOpInterface>(op))
          continue;

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
