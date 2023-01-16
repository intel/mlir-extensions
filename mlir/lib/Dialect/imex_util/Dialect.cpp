// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Dialect/plier/Dialect.hpp"
#include "imex/Dialect/imex_util/Dialect.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/InliningUtils.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>

#include <llvm/ADT/TypeSwitch.h>

namespace {
struct ImexUtilInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                       mlir::BlockAndValueMapping &) const final override {
    return true;
  }
  bool isLegalToInline(mlir::Operation *op, mlir::Region *, bool,
                       mlir::BlockAndValueMapping &) const final override {
    return true;
  }
};
} // namespace

llvm::StringRef imex::util::attributes::getFastmathName() {
  return "imex.fastmath";
}

llvm::StringRef imex::util::attributes::getJumpMarkersName() {
  return "imex.pipeline_jump_markers";
}

llvm::StringRef imex::util::attributes::getParallelName() {
  return "imex.parallel";
}

llvm::StringRef imex::util::attributes::getMaxConcurrencyName() {
  return "imex.max_concurrency";
}

llvm::StringRef imex::util::attributes::getForceInlineName() {
  return "imex.force_inline";
}

llvm::StringRef imex::util::attributes::getOptLevelName() {
  return "imex.opt_level";
}

llvm::StringRef imex::util::attributes::getShapeRangeName() {
  return "imex.shape_range";
}

namespace imex {
namespace util {

void ImexUtilDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "imex/Dialect/imex_util/ImexUtilOps.cpp.inc"
      >();

  addInterfaces<ImexUtilInlinerInterface>();

  addTypes<
#define GET_TYPEDEF_LIST
#include "imex/Dialect/imex_util/ImexUtilOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "imex/Dialect/imex_util/ImexUtilOpsAttributes.cpp.inc"
      >();
}

mlir::Operation *ImexUtilDialect::materializeConstant(mlir::OpBuilder &builder,
                                                      mlir::Attribute value,
                                                      mlir::Type type,
                                                      mlir::Location loc) {
  if (mlir::arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<mlir::arith::ConstantOp>(loc, type, value);

  if (type.isa<mlir::IndexType>())
    if (auto val = mlir::getConstantIntValue(value))
      return builder.create<mlir::arith::ConstantIndexOp>(loc, *val);

  return nullptr;
}

namespace {
template <typename DimOp, typename ExpandOp>
struct DimExpandShape : public mlir::OpRewritePattern<DimOp> {
  using mlir::OpRewritePattern<DimOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(DimOp op, mlir::PatternRewriter &rewriter) const override {
    auto es = op.getSource().template getDefiningOp<ExpandOp>();
    if (!es)
      return mlir::failure();

    auto indexAttr = mlir::getConstantIntValue(op.getIndex());
    if (!indexAttr)
      return mlir::failure();

    auto dstIndex = *indexAttr;
    auto type = es.getType().template cast<mlir::ShapedType>();
    if (!type.isDynamicDim(dstIndex))
      return mlir::failure();

    auto reassoc = es.getReassociationIndices();
    auto srcIndexAttr = [&]() -> llvm::Optional<unsigned> {
      for (auto &it : llvm::enumerate(reassoc))
        for (auto i : it.value())
          if (i == dstIndex)
            return it.index();

      return std::nullopt;
    }();

    if (!srcIndexAttr)
      return mlir::failure();

    auto shape = type.getShape();
    auto srcIndex = *srcIndexAttr;
    for (auto i : reassoc[srcIndex])
      if (i != dstIndex && shape[i] != 1)
        return mlir::failure();

    auto src = es.getSrc();
    rewriter.replaceOpWithNewOp<DimOp>(op, src, srcIndex);
    return mlir::success();
  }
};

struct DimInsertSlice : public mlir::OpRewritePattern<mlir::tensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::DimOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto insertSlice =
        op.getSource().getDefiningOp<mlir::tensor::InsertSliceOp>();
    if (!insertSlice)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::tensor::DimOp>(op, insertSlice.getDest(),
                                                     op.getIndex());
    return mlir::success();
  }
};

struct FillExtractSlice
    : public mlir::OpRewritePattern<mlir::tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ExtractSliceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto fill = op.getSource().getDefiningOp<mlir::linalg::FillOp>();
    if (!fill)
      return mlir::failure();

    auto sizes = op.getMixedSizes();
    llvm::SmallVector<mlir::OpFoldResult> newSizes;
    newSizes.reserve(sizes.size());
    auto droppedDims = op.getDroppedDims();
    for (auto it : llvm::enumerate(sizes))
      if (!droppedDims[it.index()])
        newSizes.emplace_back(it.value());

    auto fillType = fill.result().getType().cast<mlir::ShapedType>();

    auto loc = op->getLoc();
    mlir::Value init = rewriter.create<mlir::tensor::EmptyOp>(
        loc, newSizes, fillType.getElementType());

    auto fillVal = fill.value();
    auto newFill =
        rewriter.create<mlir::linalg::FillOp>(loc, fillVal, init).result();
    rewriter.replaceOp(op, newFill);
    return mlir::success();
  }
};

struct SpirvInputCSE : public mlir::OpRewritePattern<mlir::spirv::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::spirv::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ptr = op.getPtr();
    if (ptr.getType().cast<mlir::spirv::PointerType>().getStorageClass() !=
        mlir::spirv::StorageClass::Input)
      return mlir::failure();

    auto func = op->getParentOfType<mlir::spirv::FuncOp>();
    if (!func)
      return mlir::failure();

    mlir::DominanceInfo dom;
    mlir::spirv::LoadOp prevLoad;
    func->walk([&](mlir::spirv::LoadOp load) -> mlir::WalkResult {
      if (load == op)
        return mlir::WalkResult::interrupt();

      if (load->getOperands() == op->getOperands() &&
          load->getResultTypes() == op->getResultTypes() &&
          dom.properlyDominates(load.getOperation(), op)) {
        prevLoad = load;
        return mlir::WalkResult::interrupt();
      }

      return mlir::WalkResult::advance();
    });

    if (!prevLoad)
      return mlir::failure();

    rewriter.replaceOp(op, prevLoad.getResult());
    return mlir::success();
  }
};

// TODO: move to separate pass
struct GenGlobalId : public mlir::OpRewritePattern<mlir::arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::AddIOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto getArg = [](auto op, bool rev) -> mlir::Value {
      return rev ? op.getLhs() : op.getRhs();
    };

    mlir::gpu::Dimension dim;
    mlir::arith::MulIOp other;
    for (auto rev : {false, true}) {
      auto arg1 = getArg(op, rev);
      auto arg2 = getArg(op, !rev);
      if (auto tid = arg1.getDefiningOp<mlir::gpu::ThreadIdOp>()) {
        dim = tid.getDimension();
        other = arg2.getDefiningOp<mlir::arith::MulIOp>();
        break;
      }
    }

    if (!other)
      return mlir::failure();

    for (auto rev : {false, true}) {
      auto arg1 = getArg(other, rev).getDefiningOp<mlir::gpu::BlockIdOp>();
      auto arg2 = getArg(other, !rev).getDefiningOp<mlir::gpu::BlockDimOp>();
      if (arg1 && arg2) {
        if (arg1.getDimension() != dim || arg2.getDimension() != dim)
          return mlir::failure();

        rewriter.replaceOpWithNewOp<mlir::gpu::GlobalIdOp>(op, dim);
        return mlir::success();
      }
    }

    return mlir::failure();
  }
};

struct ReshapeAlloca : public mlir::OpRewritePattern<mlir::memref::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ReshapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto shapeOp = op.getShape().getDefiningOp<mlir::memref::AllocOp>();
    if (!shapeOp)
      return mlir::failure();

    for (auto user : shapeOp->getUsers())
      if (!mlir::isa<mlir::memref::StoreOp, mlir::memref::ReshapeOp>(user))
        return mlir::failure();

    if (!shapeOp.getDynamicSizes().empty() ||
        !shapeOp.getSymbolOperands().empty())
      return mlir::failure();

    auto func = op->getParentOfType<mlir::FunctionOpInterface>();
    if (!func || func.isExternal())
      return mlir::failure();

    if (shapeOp->getParentOp() != func) {
      rewriter.setInsertionPointToStart(&func.getBlocks().front());
    } else {
      rewriter.setInsertionPoint(shapeOp);
    }

    auto type = shapeOp.getType().cast<mlir::MemRefType>();
    auto alignment = shapeOp.getAlignmentAttr().cast<mlir::IntegerAttr>();
    rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(shapeOp, type,
                                                        alignment);
    return mlir::success();
  }
};
} // namespace

void ImexUtilDialect::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results) const {
  results.add<DimExpandShape<mlir::tensor::DimOp, mlir::tensor::ExpandShapeOp>,
              DimExpandShape<mlir::memref::DimOp, mlir::memref::ExpandShapeOp>,
              DimInsertSlice, FillExtractSlice, SpirvInputCSE, GenGlobalId,
              ReshapeAlloca>(getContext());
}

void EnforceShapeOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state, mlir::Value value,
                           mlir::ValueRange shape) {
  EnforceShapeOp::build(builder, state, value.getType(), value, shape);
}

mlir::OpFoldResult
EnforceShapeOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  //  operands = operands.drop_front();
  //  auto numDims = static_cast<unsigned>(operands.size());
  //  auto srcType = getType().cast<mlir::ShapedType>();
  //  llvm::SmallVector<int64_t> finalShape(numDims,
  //                                        mlir::ShapedType::kDynamicSize);
  //  if (srcType.hasRank()) {
  //    auto shape = srcType.getShape();
  //    if (shape.size() != numDims)
  //      return nullptr;

  //    finalShape.assign(shape.begin(), shape.end());
  //  }
  //  bool changed = false;
  //  for (unsigned i = 0; i < numDims; ++i) {
  //    if (auto attr = operands[i].dyn_cast_or_null<mlir::IntegerAttr>()) {
  //      auto val = attr.getInt();
  //      if (val != mlir::ShapedType::kDynamicSize) {
  //        if (finalShape[i] != mlir::ShapedType::kDynamicSize) {
  //          if (finalShape[i] != val)
  //            return nullptr;

  //        } else {
  //          changed = true;
  //          finalShape[i] = val;
  //        }
  //      }
  //    }
  //  }

  //  if (changed) {
  //    auto finalType =
  //        mlir::RankedTensorType::get(finalShape, srcType.getElementType());
  //    getResult().setType(finalType);
  //    return getResult();
  //  }
  return nullptr;
}

namespace {
struct EnforceShapeDim
    : public mlir::OpInterfaceRewritePattern<mlir::ShapedDimOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::ShapedDimOpInterface op,
                  mlir::PatternRewriter &rewriter) const override {
    auto enforceOp =
        op.getShapedValue().getDefiningOp<imex::util::EnforceShapeOp>();
    if (!enforceOp)
      return mlir::failure();

    auto constInd = mlir::getConstantIntValue(op.getDimension());
    if (!constInd)
      return mlir::failure();

    auto index = *constInd;
    if (index < 0 || index >= static_cast<int64_t>(enforceOp.getSizes().size()))
      return mlir::failure();

    rewriter.replaceOp(op, enforceOp.getSizes()[static_cast<unsigned>(index)]);
    return mlir::success();
  }
};
} // namespace

void EnforceShapeOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<EnforceShapeDim>(context);
}

/*
mlir::LogicalResult
ParallelOp::moveOutOfLoop(mlir::ArrayRef<mlir::Operation *> ops) {
  for (mlir::Operation *op : ops) {
    op->moveBefore(*this);
  }
  return mlir::success();
}
*/

mlir::Region &ParallelOp::getLoopBody() { return getRegion(); }

/*
bool ParallelOp::isDefinedOutsideOfLoop(mlir::Value value) {
  return !region().isAncestor(value.getParentRegion());
}
*/

void ParallelOp::build(
    mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState,
    mlir::ValueRange lowerBounds, mlir::ValueRange upperBounds,
    mlir::ValueRange steps,
    mlir::function_ref<void(mlir::OpBuilder &, mlir::Location, mlir::ValueRange,
                            mlir::ValueRange, mlir::Value)>
        bodyBuilder) {
  assert(lowerBounds.size() == upperBounds.size());
  assert(lowerBounds.size() == steps.size());
  odsState.addOperands(lowerBounds);
  odsState.addOperands(upperBounds);
  odsState.addOperands(steps);
  odsState.addAttribute(
      ParallelOp::getOperandSegmentSizeAttr(),
      odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBounds.size()),
                                       static_cast<int32_t>(upperBounds.size()),
                                       static_cast<int32_t>(steps.size())}));
  auto bodyRegion = odsState.addRegion();
  auto count = lowerBounds.size();
  mlir::OpBuilder::InsertionGuard guard(odsBuilder);
  llvm::SmallVector<mlir::Type> argTypes(count * 2 + 1,
                                         odsBuilder.getIndexType());
  llvm::SmallVector<mlir::Location> locs(argTypes.size(),
                                         odsBuilder.getUnknownLoc());
  auto *bodyBlock = odsBuilder.createBlock(bodyRegion, {}, argTypes, locs);

  if (bodyBuilder) {
    odsBuilder.setInsertionPointToStart(bodyBlock);
    auto args = bodyBlock->getArguments();
    bodyBuilder(odsBuilder, odsState.location, args.take_front(count),
                args.drop_front(count).take_front(count), args.back());
    ParallelOp::ensureTerminator(*bodyRegion, odsBuilder, odsState.location);
  }
}

void RetainOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     mlir::Value value) {
  RetainOp::build(builder, state, value.getType(), value);
}

static mlir::Value getChangeLayoutParent(mlir::Value val) {
  if (auto parent = val.getDefiningOp<imex::util::ChangeLayoutOp>())
    return parent.getSource();

  return {};
}

mlir::OpFoldResult
ChangeLayoutOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  auto src = getSource();
  auto thisType = getType();
  do {
    if (thisType == src.getType())
      return src;
  } while ((src = getChangeLayoutParent(src)));

  return nullptr;
}

namespace {
static bool canTransformLayoutCast(mlir::Type src, mlir::Type dst) {
  auto srcType = src.dyn_cast<mlir::MemRefType>();
  auto dstType = dst.dyn_cast<mlir::MemRefType>();
  if (!srcType || !dstType ||
      !mlir::memref::CastOp::areCastCompatible(srcType, dstType))
    return false;

  int64_t srcOffset, dstOffset;
  llvm::SmallVector<int64_t> srcStrides, dstStrides;
  if (mlir::failed(mlir::getStridesAndOffset(srcType, srcStrides, srcOffset)) ||
      mlir::failed(mlir::getStridesAndOffset(dstType, dstStrides, dstOffset)))
    return false;

  auto isStrideCompatible = [](int64_t src, int64_t dst) {
    auto isStatic = [](int64_t v) { return !mlir::ShapedType::isDynamic(v); };
    if (isStatic(src) && isStatic(dst)) {
      return src == dst;
    } else if (isStatic(src)) {
      return true;
    } else if (isStatic(dst)) {
      return false;
    } else {
      // Both dynamic
      return true;
    }
  };

  assert(srcStrides.size() == dstStrides.size());
  if (!isStrideCompatible(srcOffset, dstOffset))
    return false;

  auto rank = static_cast<unsigned>(srcStrides.size());
  for (auto i : llvm::seq(0u, rank))
    if (!isStrideCompatible(srcStrides[i], dstStrides[i]))
      return false;

  return true;
}

static mlir::MemRefType getFullyDynamicType(mlir::Type type) {
  auto memrefType = type.dyn_cast<mlir::MemRefType>();
  if (!memrefType)
    return {};

  auto layout = memrefType.getLayout().dyn_cast<mlir::StridedLayoutAttr>();
  if (!layout)
    return {};

  int64_t offset = mlir::ShapedType::kDynamic;
  llvm::SmallVector<int64_t> strides(layout.getStrides().size(), offset);
  auto dynLayout =
      mlir::StridedLayoutAttr::get(type.getContext(), offset, strides);
  if (layout == dynLayout)
    return {};

  return mlir::MemRefType::get(memrefType.getShape(),
                               memrefType.getElementType(), dynLayout,
                               memrefType.getMemorySpace());
}

struct ChangeLayoutIdentity
    : public mlir::OpRewritePattern<imex::util::ChangeLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::ChangeLayoutOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto srcType = src.getType().cast<mlir::MemRefType>();
    auto dstType = op.getType();
    if (!canTransformLayoutCast(srcType, dstType))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(op, dstType, src);
    return mlir::success();
  }
};

struct ChangeLayoutDim : public mlir::OpRewritePattern<mlir::memref::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::DimOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.getSource().getDefiningOp<imex::util::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::DimOp>(op, cl.getSource(),
                                                     op.getIndex());
    return mlir::success();
  }
};

struct ChangeLayoutClone
    : public mlir::OpRewritePattern<mlir::bufferization::CloneOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::bufferization::CloneOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.getInput().getDefiningOp<imex::util::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    auto src = cl.getSource();
    auto dstType = op.getType();

    auto loc = op.getLoc();
    auto res = rewriter.createOrFold<mlir::bufferization::CloneOp>(loc, src);
    rewriter.replaceOpWithNewOp<imex::util::ChangeLayoutOp>(op, dstType, res);
    return mlir::success();
  }
};

struct PropagateCloneType
    : public mlir::OpRewritePattern<mlir::bufferization::CloneOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::bufferization::CloneOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getInput();
    auto srcType = src.getType();
    auto dstType = op.getType();
    if (srcType == dstType)
      return mlir::failure();

    auto loc = op.getLoc();
    auto res = rewriter.createOrFold<mlir::bufferization::CloneOp>(loc, src);
    rewriter.replaceOpWithNewOp<imex::util::ChangeLayoutOp>(op, dstType, res);
    return mlir::success();
  }
};

struct ChangeLayoutCast : public mlir::OpRewritePattern<mlir::memref::CastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::CastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.getSource().getDefiningOp<imex::util::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    auto src = cl.getSource();
    auto srcType = src.getType().cast<mlir::MemRefType>();
    auto dstType = op.getType().cast<mlir::MemRefType>();
    if (srcType == dstType) {
      rewriter.replaceOp(op, src);
      return mlir::success();
    }

    if (canTransformLayoutCast(srcType, dstType)) {
      rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(op, dstType, src);
      return mlir::success();
    }

    auto loc = op->getLoc();
    auto newDstType =
        mlir::MemRefType::get(dstType.getShape(), srcType.getElementType(),
                              srcType.getLayout(), srcType.getMemorySpace());
    mlir::Value newCast =
        rewriter.create<mlir::memref::CastOp>(loc, newDstType, src);
    rewriter.replaceOpWithNewOp<imex::util::ChangeLayoutOp>(op, dstType,
                                                            newCast);
    return mlir::success();
  }
};

struct ChangeLayoutFromCast
    : public mlir::OpRewritePattern<imex::util::ChangeLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::ChangeLayoutOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cast = op.getSource().getDefiningOp<mlir::memref::CastOp>();
    if (!cast)
      return mlir::failure();

    auto src = cast.getSource();
    auto srcType = src.getType();
    auto dstType = op.getType();
    if (srcType == dstType) {
      rewriter.replaceOp(op, src);
      return mlir::success();
    }

    if (canTransformLayoutCast(srcType, dstType)) {
      rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(op, dstType, src);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct ChangeLayoutLoad : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.getMemref().getDefiningOp<imex::util::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, cl.getSource(),
                                                      op.getIndices());
    return mlir::success();
  }
};

struct ChangeLayoutStore
    : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.getMemref().getDefiningOp<imex::util::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(
        op, op.getValue(), cl.getSource(), op.getIndices());
    return mlir::success();
  }
};

struct ChangeLayoutSubview
    : public mlir::OpRewritePattern<mlir::memref::SubViewOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::SubViewOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.getSource().getDefiningOp<imex::util::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    auto offsets = op.getMixedOffsets();
    auto sizes = op.getMixedSizes();
    auto strides = op.getMixedStrides();

    auto src = cl.getSource();
    auto srcType = src.getType().cast<mlir::MemRefType>();
    auto dstType = op.getType().cast<mlir::MemRefType>();
    auto newDstType =
        [&]() {
          auto srcRank = srcType.getRank();
          auto dstRank = dstType.getRank();
          if (srcRank == dstRank)
            return mlir::memref::SubViewOp::inferResultType(srcType, offsets,
                                                            sizes, strides);

          return mlir::memref::SubViewOp::inferRankReducedResultType(
              dstType.getShape(), srcType, offsets, sizes, strides);
        }()
            .cast<mlir::MemRefType>();

    auto loc = op.getLoc();
    auto newSubview = rewriter.createOrFold<mlir::memref::SubViewOp>(
        loc, newDstType, src, offsets, sizes, strides);
    if (newDstType != dstType)
      newSubview = rewriter.createOrFold<imex::util::ChangeLayoutOp>(
          loc, dstType, newSubview);

    rewriter.replaceOp(op, newSubview);
    return mlir::success();
  }
};

struct ChangeLayoutLinalgGeneric
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::GenericOp op,
                  mlir::PatternRewriter &rewriter) const override {
    bool changed = false;
    llvm::SmallVector<mlir::Value> newOperands;

    for (bool isInputs : {true, false}) {
      mlir::ValueRange args = (isInputs ? op.getInputs() : op.getOutputs());
      auto count = static_cast<unsigned>(args.size());
      newOperands.resize(count);
      bool needUpdate = false;
      for (auto i : llvm::seq(0u, count)) {
        auto arg = args[i];
        auto cl = arg.getDefiningOp<imex::util::ChangeLayoutOp>();
        if (cl) {
          assert(arg.getType().isa<mlir::MemRefType>());
          assert(cl.getSource().getType().isa<mlir::MemRefType>());
          newOperands[i] = cl.getSource();
          needUpdate = true;
          changed = true;
        } else {
          newOperands[i] = arg;
        }
      }

      if (needUpdate) {
        rewriter.updateRootInPlace(op, [&]() {
          (isInputs ? op.getInputsMutable() : op.getOutputsMutable())
              .assign(newOperands);
        });
      }
    }

    return mlir::success(changed);
  }
};

struct ChangeLayoutLinalgFill
    : public mlir::OpRewritePattern<mlir::linalg::FillOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::FillOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto output = op.output();
    auto clOutput = output.getDefiningOp<imex::util::ChangeLayoutOp>();
    if (!clOutput)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::linalg::FillOp>(op, op.value(),
                                                      clOutput.getSource());
    return mlir::success();
  }
};

struct ChangeLayoutIf : public mlir::OpRewritePattern<mlir::scf::YieldOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getResults().empty())
      return mlir::failure();

    auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op->getParentOp());
    if (!ifOp)
      return mlir::failure();

    auto trueYield = mlir::cast<mlir::scf::YieldOp>(
        ifOp.getThenRegion().front().getTerminator());
    auto falseYield = mlir::cast<mlir::scf::YieldOp>(
        ifOp.getElseRegion().front().getTerminator());
    mlir::OpBuilder::InsertionGuard g(rewriter);
    auto count = static_cast<unsigned>(trueYield.getResults().size());
    llvm::SmallVector<mlir::Type> newResultTypes(count);
    bool changed = false;
    for (auto i : llvm::seq(0u, count)) {
      auto origType = ifOp.getResult(i).getType();

      mlir::Type newType;
      for (bool reverse : {true, false}) {
        auto clYield = (reverse ? falseYield : trueYield);
        auto otherYield = (reverse ? trueYield : falseYield);

        auto arg = clYield.getResults()[i];
        if (!arg.getType().isa<mlir::MemRefType>())
          continue;

        auto cl = arg.getDefiningOp<imex::util::ChangeLayoutOp>();
        if (!cl)
          continue;

        auto src = cl.getSource();
        auto srcType = src.getType();

        auto otherArg = otherYield.getResults()[i];

        if (auto otherCl =
                otherArg.getDefiningOp<imex::util::ChangeLayoutOp>()) {
          auto otherSrc = otherCl.getSource();
          if (otherSrc.getType() == srcType) {
            rewriter.updateRootInPlace(
                otherYield, [&]() { otherYield.setOperand(i, otherSrc); });
            newType = srcType;
            break;
          }
        }

        bool outerBreak = false;
        for (auto dstType : {srcType, getFullyDynamicType(srcType)}) {
          if (!dstType)
            continue;

          if (canTransformLayoutCast(origType, dstType)) {
            if (srcType != dstType) {
              rewriter.setInsertionPoint(clYield);
              src = rewriter.create<mlir::memref::CastOp>(clYield.getLoc(),
                                                          dstType, src);
            }

            rewriter.updateRootInPlace(clYield,
                                       [&]() { clYield.setOperand(i, src); });

            rewriter.setInsertionPoint(otherYield);
            auto otherRes = rewriter.createOrFold<mlir::memref::CastOp>(
                otherYield.getLoc(), dstType, otherArg);

            rewriter.updateRootInPlace(
                otherYield, [&]() { otherYield.setOperand(i, otherRes); });
            newType = dstType;
            outerBreak = true;
            break;
          }
        }

        if (outerBreak)
          break;
      }

      if (!newType) {
        newResultTypes[i] = origType;
      } else {
        newResultTypes[i] = newType;
        changed = true;
      }
    }

    if (changed) {
      rewriter.setInsertionPointAfter(ifOp);
      rewriter.updateRootInPlace(ifOp, [&]() {
        auto loc = ifOp.getLoc();
        for (auto i : llvm::seq(0u, count)) {
          auto res = ifOp.getResult(i);
          auto origType = res.getType();
          auto newType = newResultTypes[i];
          if (origType != newType) {
            res.setType(newType);
            auto cl =
                rewriter.create<imex::util::ChangeLayoutOp>(loc, origType, res);
            res.replaceAllUsesExcept(cl, cl);
          }
        }
      });
    }
    return mlir::success(changed);
  }
};

static llvm::Optional<unsigned> getSingleDynamicDim(mlir::ShapedType type) {
  if (!type.hasRank())
    return std::nullopt;

  int dimIndex = -1;
  for (auto it : llvm::enumerate(type.getShape())) {
    auto i = static_cast<int>(it.index());
    auto dim = it.value();
    if (dim == mlir::ShapedType::kDynamic) {
      if (dimIndex != -1)
        return std::nullopt;

      dimIndex = i;
    } else if (dim != 1) {
      return std::nullopt;
    }
  }

  if (dimIndex != -1)
    return static_cast<unsigned>(dimIndex);

  return std::nullopt;
}

struct ChangeLayout1DReshape
    : public mlir::OpRewritePattern<mlir::memref::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ReshapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto source = op.getSource();
    auto shape = op.getShape();
    auto srcType = source.getType().cast<mlir::MemRefType>();
    auto dstType = op.getType().cast<mlir::MemRefType>();
    if (dstType.getRank() != 1)
      return mlir::failure();

    auto srcDimIndex = getSingleDynamicDim(srcType);
    if (!srcDimIndex)
      return mlir::failure();

    auto srcRank = static_cast<unsigned>(srcType.getRank());
    assert(*srcDimIndex < srcRank);
    auto loc = op.getLoc();
    auto zero =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
    using ArrayType = llvm::SmallVector<mlir::OpFoldResult>;
    ArrayType offsets(srcRank, rewriter.getIndexAttr(0));
    ArrayType sizes(srcRank, rewriter.getIndexAttr(1));
    sizes[*srcDimIndex] =
        rewriter.createOrFold<mlir::memref::LoadOp>(loc, shape, zero);
    ArrayType strides(srcRank, rewriter.getIndexAttr(1));
    auto view = rewriter.createOrFold<mlir::memref::SubViewOp>(
        loc, source, offsets, sizes, strides);
    auto dstRank = dstType.getRank();
    if (srcRank != dstRank) {
      assert(dstRank < srcRank);
      llvm::SmallVector<mlir::OpFoldResult> newOfsets(srcRank,
                                                      rewriter.getIndexAttr(0));
      llvm::SmallVector<mlir::OpFoldResult> newStrides(
          srcRank, rewriter.getIndexAttr(1));
      auto viewType = view.getType().cast<mlir::MemRefType>();
      auto reducedType =
          mlir::memref::SubViewOp::inferRankReducedResultType(
              dstType.getShape(), viewType, newOfsets, sizes, newStrides)
              .cast<mlir::MemRefType>();
      view = rewriter.create<mlir::memref::SubViewOp>(
          loc, reducedType, view, newOfsets, sizes, newStrides);
    }
    rewriter.replaceOpWithNewOp<imex::util::ChangeLayoutOp>(op, dstType, view);
    return mlir::success();
  }
};

struct ChangeLayoutSliceGetItem
    : public mlir::OpRewritePattern<plier::SliceGetItemOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::SliceGetItemOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.getArray().getDefiningOp<imex::util::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<plier::SliceGetItemOp>(
        op, op.getType(), op.getSlice(), cl.getSource(), op.getIndex(),
        op.getDim());
    return mlir::success();
  }
};

struct ChangeLayoutCopy : public mlir::OpRewritePattern<mlir::memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::CopyOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto input = op.getSource();
    auto output = op.getTarget();
    auto clInput = input.getDefiningOp<imex::util::ChangeLayoutOp>();
    auto clOutput = output.getDefiningOp<imex::util::ChangeLayoutOp>();
    if (!clInput && !clOutput)
      return mlir::failure();

    if (clInput)
      input = clInput.getSource();

    if (clOutput)
      output = clOutput.getSource();

    rewriter.replaceOpWithNewOp<mlir::memref::CopyOp>(op, input, output);
    return mlir::success();
  }
};

struct ChangeLayoutExpandShape
    : public mlir::OpRewritePattern<mlir::memref::ExpandShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ExpandShapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.getSrc().getDefiningOp<imex::util::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    auto dstType = op.getType().cast<mlir::MemRefType>();
    if (!dstType.getLayout().isIdentity())
      return mlir::failure();

    auto src = cl.getSource();
    auto srcType = src.getType().cast<mlir::MemRefType>();
    if (!mlir::isStrided(srcType))
      return mlir::failure();

    auto reassoc = op.getReassociationIndices();
    auto newDstType = mlir::memref::ExpandShapeOp::computeExpandedType(
        srcType, dstType.getShape(), reassoc);
    if (mlir::failed(newDstType))
      return mlir::failure();

    auto loc = op->getLoc();
    mlir::Value newOp = rewriter.create<mlir::memref::ExpandShapeOp>(
        loc, *newDstType, src, reassoc);
    rewriter.replaceOpWithNewOp<imex::util::ChangeLayoutOp>(op, dstType, newOp);
    return mlir::success();
  }
};

/// Propagates ChangeLayoutOp through SelectOp.
///
/// Example:
/// %0 = imex_util.change_layout %arg1 : memref<?xi32, #map> to memref<?xi32>
/// %res = arith.select %arg3, %0, %arg2 : memref<?xi32>
///
/// Becomes:
/// %0 = memref.cast %arg2 : memref<?xi32> to memref<?xi32, #map>
/// %1 = arith.select %arg3, %arg1, %0 : memref<?xi32, #map>
/// %res  = imex_util.change_layout %1 : memref<?xi32, #map> to memref<?xi32>
struct ChangeLayoutSelect
    : public mlir::OpRewritePattern<mlir::arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.getResult().getType().isa<mlir::MemRefType>())
      return mlir::failure();

    auto trueArg = op.getTrueValue();
    auto falseArg = op.getFalseValue();
    for (bool reverse : {false, true}) {
      auto arg = reverse ? falseArg : trueArg;
      auto cl = arg.getDefiningOp<imex::util::ChangeLayoutOp>();
      if (!cl)
        continue;

      auto srcType = cl.getSource().getType().cast<mlir::MemRefType>();
      auto dstType = arg.getType().cast<mlir::MemRefType>();

      auto otherArg = reverse ? trueArg : falseArg;

      auto otherArgType = otherArg.getType().cast<mlir::MemRefType>();

      arg = cl.getSource();
      if (!canTransformLayoutCast(otherArgType, srcType)) {
        auto dynStride = mlir::ShapedType::kDynamic;
        llvm::SmallVector<int64_t> strides(srcType.getRank(), dynStride);
        auto dynStrides =
            mlir::StridedLayoutAttr::get(op->getContext(), dynStride, strides);
        auto dynStridesMemref =
            mlir::MemRefType::get(srcType.getShape(), srcType.getElementType(),
                                  dynStrides, srcType.getMemorySpace());
        if (!canTransformLayoutCast(otherArgType, dynStridesMemref))
          continue;

        srcType = dynStridesMemref;
        arg = rewriter.create<mlir::memref::CastOp>(op->getLoc(), srcType, arg);
      }

      auto loc = op->getLoc();
      otherArg = rewriter.create<mlir::memref::CastOp>(loc, srcType, otherArg);

      if (reverse) {
        trueArg = otherArg;
        falseArg = arg;
      } else {
        trueArg = arg;
        falseArg = otherArg;
      }

      auto cond = op.getCondition();
      auto result =
          rewriter.create<mlir::arith::SelectOp>(loc, cond, trueArg, falseArg);
      rewriter.replaceOpWithNewOp<imex::util::ChangeLayoutOp>(op, dstType,
                                                              result);

      return mlir::success();
    }

    return mlir::failure();
  }
};

struct ChangeLayoutEnvRegion
    : public mlir::OpRewritePattern<imex::util::EnvironmentRegionYieldOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::EnvironmentRegionYieldOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto args = op.getResults();
    llvm::SmallVector<mlir::Value> updatedArgs(args.begin(), args.end());

    bool changed = false;
    for (auto [i, arg] : llvm::enumerate(args)) {
      auto cl = arg.getDefiningOp<imex::util::ChangeLayoutOp>();
      if (!cl)
        continue;

      updatedArgs[i] = cl.getSource();
      changed = true;
    }

    if (!changed)
      return mlir::failure();

    auto region =
        mlir::cast<imex::util::EnvironmentRegionOp>(op->getParentOp());
    rewriter.updateRootInPlace(
        op, [&]() { op.getResultsMutable().assign(updatedArgs); });

    rewriter.updateRootInPlace(region, [&]() {
      auto loc = region.getLoc();
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointAfter(region);
      for (auto [arg, result] : llvm::zip(updatedArgs, region.getResults())) {
        auto oldType = result.getType();
        auto newType = arg.getType();
        if (newType == oldType)
          continue;

        auto cast =
            rewriter.create<imex::util::ChangeLayoutOp>(loc, oldType, result);
        mlir::Value newResult = cast.getResult();
        for (auto &use : llvm::make_early_inc_range(result.getUses())) {
          auto owner = use.getOwner();
          if (owner == cast)
            continue;

          rewriter.updateRootInPlace(owner, [&]() { use.set(newResult); });
        }
        result.setType(newType);
      }
    });
    return mlir::success();
  }
};

} // namespace

void ChangeLayoutOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results
      .insert<ChangeLayoutIdentity, ChangeLayoutDim, ChangeLayoutClone,
              PropagateCloneType, ChangeLayoutCast, ChangeLayoutFromCast,
              ChangeLayoutLoad, ChangeLayoutStore, ChangeLayoutSubview,
              ChangeLayoutLinalgGeneric, ChangeLayoutLinalgFill, ChangeLayoutIf,
              ChangeLayout1DReshape, ChangeLayoutSliceGetItem, ChangeLayoutCopy,
              ChangeLayoutExpandShape, ChangeLayoutSelect,
              ChangeLayoutEnvRegion>(context);
}

bool ChangeLayoutOp::areCastCompatible(mlir::TypeRange inputs,
                                       mlir::TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;

  mlir::Type a = inputs.front(), b = outputs.front();
  auto aT = a.dyn_cast<mlir::MemRefType>();
  auto bT = b.dyn_cast<mlir::MemRefType>();
  if (!aT || !bT)
    return false;

  if (aT.getElementType() != bT.getElementType() ||
      mlir::failed(mlir::verifyCompatibleShape(aT, bT)) ||
      aT.getMemorySpace() != bT.getMemorySpace())
    return false;

  return true;
}

static mlir::Value propagateCasts(mlir::Value val, mlir::Type thisType);

template <typename T>
static mlir::Value foldPrevCast(mlir::Value val, mlir::Type thisType) {
  if (auto prevOp = val.getDefiningOp<T>()) {
    auto prevArg = prevOp->getOperand(0);
    if (prevArg.getType() == thisType)
      return prevArg;

    auto res = propagateCasts(prevArg, thisType);
    if (res)
      return res;
  }
  return {};
}

static mlir::Value propagateCasts(mlir::Value val, mlir::Type thisType) {
  using fptr = mlir::Value (*)(mlir::Value, mlir::Type);
  const fptr handlers[] = {
      &foldPrevCast<imex::util::SignCastOp>,
      &foldPrevCast<plier::CastOp>,
      &foldPrevCast<mlir::UnrealizedConversionCastOp>,
  };

  for (auto h : handlers) {
    auto res = h(val, thisType);
    if (res)
      return res;
  }

  return {};
}

mlir::OpFoldResult SignCastOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  assert(operands.size() == 1);
  auto thisType = getType();
  auto attrOperand = operands.front().dyn_cast_or_null<mlir::TypedAttr>();
  if (attrOperand && attrOperand.getType() == thisType)
    return attrOperand;

  auto arg = getSource();
  if (arg.getType() == thisType)
    return arg;

  if (auto res = propagateCasts(arg, thisType))
    return res;

  return nullptr;
}

namespace {
template <typename Op>
struct SignCastDimPropagate : public mlir::OpRewritePattern<Op> {
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    auto castOp =
        op.getSource().template getDefiningOp<imex::util::SignCastOp>();
    if (!castOp)
      return mlir::failure();

    auto val = castOp.getSource();
    rewriter.replaceOpWithNewOp<Op>(op, val, op.getIndex());
    return mlir::success();
  }
};

struct SignCastUndefPropagate
    : public mlir::OpRewritePattern<imex::util::SignCastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::SignCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto undefOp = op.getSource().getDefiningOp<imex::util::UndefOp>();
    if (!undefOp)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<imex::util::UndefOp>(op, op.getType());
    return mlir::success();
  }
};

template <typename CastOp>
struct SignCastCastPropagate : public mlir::OpRewritePattern<CastOp> {
  using mlir::OpRewritePattern<CastOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(CastOp op, mlir::PatternRewriter &rewriter) const override {
    auto signCast =
        op.getSource().template getDefiningOp<imex::util::SignCastOp>();
    if (!signCast)
      return mlir::failure();

    auto srcType = signCast.getType().template cast<mlir::ShapedType>();
    auto dstType = op.getType().template cast<mlir::ShapedType>();
    if (srcType.getElementType() != dstType.getElementType() ||
        !srcType.hasRank() || !dstType.hasRank())
      return mlir::failure();

    auto src = signCast.getSource();
    auto finalType = src.getType().template cast<mlir::ShapedType>();
    auto finalElemType = finalType.getElementType();

    auto newDstType = dstType.clone(finalElemType);
    if (!CastOp::areCastCompatible(src.getType(), newDstType))
      return mlir::failure();

    auto loc = op.getLoc();
    mlir::Value cast = rewriter.create<CastOp>(loc, newDstType, src);
    rewriter.replaceOpWithNewOp<imex::util::SignCastOp>(op, dstType, cast);

    return mlir::success();
  }
};

struct SignCastReinterpretPropagate
    : public mlir::OpRewritePattern<mlir::memref::ReinterpretCastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ReinterpretCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto signCast = op.getSource().getDefiningOp<imex::util::SignCastOp>();
    if (!signCast)
      return mlir::failure();

    auto srcType = signCast.getType().cast<mlir::ShapedType>();
    auto dstType = op.getType().cast<mlir::MemRefType>();
    if (srcType.getElementType() != dstType.getElementType())
      return mlir::failure();

    auto src = signCast.getSource();
    auto finalType = src.getType().cast<mlir::MemRefType>();

    auto newDstType =
        mlir::MemRefType::get(dstType.getShape(), dstType.getElementType(),
                              dstType.getLayout(), finalType.getMemorySpace());

    auto loc = op.getLoc();
    auto offset = op.getMixedOffsets().front();
    auto sizes = op.getMixedSizes();
    auto strides = op.getMixedStrides();
    auto cast = rewriter.createOrFold<mlir::memref::ReinterpretCastOp>(
        loc, newDstType, src, offset, sizes, strides);
    rewriter.replaceOpWithNewOp<imex::util::SignCastOp>(op, dstType, cast);

    return mlir::success();
  }
};

struct SignCastLoadPropagate
    : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto signCast = op.getMemref().getDefiningOp<imex::util::SignCastOp>();
    if (!signCast)
      return mlir::failure();

    auto loc = op.getLoc();
    auto src = signCast.getSource();
    auto newOp =
        rewriter.createOrFold<mlir::memref::LoadOp>(loc, src, op.getIndices());

    if (newOp.getType() != op.getType())
      newOp = rewriter.create<imex::util::SignCastOp>(loc, op.getType(), newOp);

    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

struct SignCastStorePropagate
    : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto signCast = op.getMemref().getDefiningOp<imex::util::SignCastOp>();
    if (!signCast)
      return mlir::failure();

    auto src = signCast.getSource();
    auto srcElemType = src.getType().cast<mlir::MemRefType>().getElementType();
    auto val = op.getValue();
    if (val.getType() != srcElemType)
      val = rewriter.create<imex::util::SignCastOp>(op.getLoc(), srcElemType,
                                                    val);

    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, val, src,
                                                       op.getIndices());
    return mlir::success();
  }
};

template <typename Op>
struct SignCastAllocPropagate
    : public mlir::OpRewritePattern<imex::util::SignCastOp> {
  using mlir::OpRewritePattern<imex::util::SignCastOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::SignCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto alloc = op.getSource().getDefiningOp<Op>();
    if (!alloc || !alloc->hasOneUse())
      return mlir::failure();

    auto dstType = op.getType().cast<mlir::MemRefType>();
    rewriter.replaceOpWithNewOp<Op>(op, dstType, alloc.getDynamicSizes(),
                                    alloc.getSymbolOperands(),
                                    alloc.getAlignmentAttr());
    rewriter.eraseOp(alloc);
    return mlir::success();
  }
};

struct SignCastTensorFromElementsPropagate
    : public mlir::OpRewritePattern<imex::util::SignCastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::SignCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto fromElements =
        op.getSource().getDefiningOp<mlir::tensor::FromElementsOp>();
    if (!fromElements)
      return mlir::failure();

    auto loc = fromElements->getLoc();
    auto dstType = op.getType().cast<mlir::TensorType>();
    auto elemType = dstType.getElementType();
    auto elements = fromElements.getElements();
    auto count = static_cast<unsigned>(elements.size());
    llvm::SmallVector<mlir::Value> castedVals(count);
    for (auto i : llvm::seq(0u, count))
      castedVals[i] =
          rewriter.create<imex::util::SignCastOp>(loc, elemType, elements[i]);

    rewriter.replaceOpWithNewOp<mlir::tensor::FromElementsOp>(op, castedVals);
    return mlir::success();
  }
};

struct SignCastTensorCollapseShapePropagate
    : public mlir::OpRewritePattern<imex::util::SignCastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::SignCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto prevOp = op.getSource().getDefiningOp<mlir::tensor::CollapseShapeOp>();
    if (!prevOp)
      return mlir::failure();

    auto src = prevOp.getSrc();
    auto srcType = src.getType().cast<mlir::TensorType>();
    auto dstType = op.getType().cast<mlir::TensorType>();

    auto newSrcType = srcType.clone(dstType.getElementType());
    auto newDstType = dstType.clone(dstType.getElementType());

    auto loc = prevOp->getLoc();
    auto newSrc = rewriter.create<imex::util::SignCastOp>(loc, newSrcType, src);
    rewriter.replaceOpWithNewOp<mlir::tensor::CollapseShapeOp>(
        op, newDstType, newSrc, prevOp.getReassociation());
    return mlir::success();
  }
};

template <typename BuffOp>
struct SignCastBuferizationPropagate : public mlir::OpRewritePattern<BuffOp> {
  using mlir::OpRewritePattern<BuffOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(BuffOp op, mlir::PatternRewriter &rewriter) const override {
    auto signCast =
        op->getOperand(0).template getDefiningOp<imex::util::SignCastOp>();
    if (!signCast)
      return mlir::failure();

    auto src = signCast.getSource();
    auto srcType = src.getType().template cast<mlir::ShapedType>();
    auto dstType = op.getType().template cast<mlir::ShapedType>();
    auto newDstType = dstType.clone(srcType.getElementType());

    auto loc = op.getLoc();
    auto res = rewriter.create<BuffOp>(loc, newDstType, src);
    rewriter.replaceOpWithNewOp<imex::util::SignCastOp>(op, dstType, res);
    return mlir::success();
  }
};

template <typename ViewOp, typename ArrType>
struct SignCastSubviewPropagate : public mlir::OpRewritePattern<ViewOp> {
  using mlir::OpRewritePattern<ViewOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ViewOp op, mlir::PatternRewriter &rewriter) const override {
    auto signCast =
        op.getSource().template getDefiningOp<imex::util::SignCastOp>();
    if (!signCast)
      return mlir::failure();

    auto src = signCast.getSource();
    auto srcType = src.getType().template cast<ArrType>();
    auto dstType = op.getType().template cast<ArrType>();
    auto newDstType =
        dstType.clone(srcType.getElementType()).template cast<ArrType>();

    auto loc = op.getLoc();
    auto res =
        rewriter.create<ViewOp>(loc, newDstType, src, op.getMixedOffsets(),
                                op.getMixedSizes(), op.getMixedStrides());
    rewriter.replaceOpWithNewOp<imex::util::SignCastOp>(op, dstType, res);
    return mlir::success();
  }
};

struct SignCastForPropagate : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto &body = op.getLoopBody().front();
    auto term = mlir::cast<mlir::scf::YieldOp>(body.getTerminator());
    auto termResults = term.getResults();
    auto initArgs = op.getInitArgs();
    auto count = static_cast<unsigned>(initArgs.size());
    assert(termResults.size() == count);

    auto loc = op->getLoc();
    llvm::SmallVector<mlir::Value> newInitArgs(count);
    bool needUpdate = false;
    for (auto i : llvm::seq(0u, count)) {
      auto initArg = initArgs[i];
      auto yieldArg = termResults[i];
      assert(initArg.getType() == yieldArg.getType());
      auto yieldCast = yieldArg.getDefiningOp<imex::util::SignCastOp>();
      if (yieldCast) {
        auto newType = yieldCast.getSource().getType();
        newInitArgs[i] =
            rewriter.create<imex::util::SignCastOp>(loc, newType, initArg);
        needUpdate = true;
      } else {
        newInitArgs[i] = initArg;
      }
    }

    if (!needUpdate)
      return mlir::failure();

    auto bodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value iter, mlir::ValueRange iterVals) {
      assert(iterVals.size() == count);
      mlir::BlockAndValueMapping mapping;
      mapping.map(body.getArguments()[0], iter);
      auto oldIterVals = body.getArguments().drop_front(1);
      for (auto i : llvm::seq(0u, count)) {
        auto iterVal = iterVals[i];
        auto oldIterVal = oldIterVals[i];
        auto oldType = oldIterVal.getType();
        if (iterVal.getType() != oldType) {
          auto newIterVal =
              builder.create<imex::util::SignCastOp>(loc, oldType, iterVal);
          mapping.map(oldIterVal, newIterVal.getResult());
        } else {
          mapping.map(oldIterVal, iterVal);
        }
      }

      for (auto &bodyOp : body.without_terminator())
        builder.clone(bodyOp, mapping);

      llvm::SmallVector<mlir::Value> newYieldArgs(count);
      for (auto i : llvm::seq(0u, count)) {
        auto val = mapping.lookupOrDefault(termResults[i]);
        auto newType = newInitArgs[i].getType();
        if (val.getType() != newType)
          val = val.getDefiningOp<imex::util::SignCastOp>().getSource();

        assert(val.getType() == newType);
        newYieldArgs[i] = val;
      }
      builder.create<mlir::scf::YieldOp>(loc, newYieldArgs);
    };

    auto newOp = rewriter.create<mlir::scf::ForOp>(
        loc, op.getLowerBound(), op.getUpperBound(), op.getStep(), newInitArgs,
        bodyBuilder);
    newOp->setAttrs(op->getAttrs());
    auto newResults = newOp.getResults();

    for (auto i : llvm::seq(0u, count)) {
      auto oldRersultType = initArgs[i].getType();
      mlir::Value newResult = newResults[i];
      if (newResult.getType() != oldRersultType)
        newResult = rewriter.create<imex::util::SignCastOp>(loc, oldRersultType,
                                                            newResult);

      newInitArgs[i] = newResult;
    }

    rewriter.replaceOp(op, newInitArgs);
    return mlir::success();
  }
};

template <typename CastOp>
struct SignCastIfPropagate : public mlir::OpRewritePattern<mlir::scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumResults() == 0)
      return mlir::failure();

    auto thenYield = op.thenYield();
    auto elseYield = op.elseYield();

    unsigned idx;
    CastOp castOp;
    imex::util::UndefOp undefOp;
    for (auto [i, args] : llvm::enumerate(
             llvm::zip(thenYield.getResults(), elseYield.getResults()))) {
      auto [thenArg, elseArg] = args;
      auto cast = thenArg.template getDefiningOp<CastOp>();
      auto undef = elseArg.template getDefiningOp<imex::util::UndefOp>();
      if (cast && undef) {
        idx = static_cast<unsigned>(i);
        castOp = cast;
        undefOp = undef;
        break;
      }
    }

    if (!castOp)
      return mlir::failure();

    auto dstType = castOp.getType();

    auto src = castOp.getSource();
    auto srcType = src.getType();

    rewriter.updateRootInPlace(thenYield,
                               [&]() { thenYield->setOperand(idx, src); });

    mlir::OpBuilder::InsertionGuard g(rewriter);

    rewriter.setInsertionPoint(elseYield);
    auto newUndef =
        rewriter.create<imex::util::UndefOp>(undefOp->getLoc(), srcType);

    rewriter.updateRootInPlace(
        elseYield, [&]() { elseYield->setOperand(idx, newUndef.getResult()); });

    auto res = op.getResult(idx);

    rewriter.setInsertionPointAfter(op);
    auto newRes = rewriter.create<CastOp>(castOp->getLoc(), dstType, res);

    for (auto &use : llvm::make_early_inc_range(res.getUses())) {
      auto owner = use.getOwner();
      if (owner == newRes)
        continue;

      rewriter.updateRootInPlace(owner, [&]() { use.set(newRes); });
    }

    rewriter.updateRootInPlace(op,
                               [&]() { op->getResult(idx).setType(srcType); });

    return mlir::success();
  }
};

} // namespace

void SignCastOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                             ::mlir::MLIRContext *context) {
  results.insert<
      SignCastDimPropagate<mlir::tensor::DimOp>,
      SignCastDimPropagate<mlir::memref::DimOp>, SignCastUndefPropagate,
      SignCastCastPropagate<mlir::tensor::CastOp>,
      SignCastCastPropagate<mlir::memref::CastOp>,
      SignCastCastPropagate<imex::util::ChangeLayoutOp>,
      SignCastReinterpretPropagate, SignCastLoadPropagate,
      SignCastStorePropagate, SignCastAllocPropagate<mlir::memref::AllocOp>,
      SignCastAllocPropagate<mlir::memref::AllocaOp>,
      SignCastTensorFromElementsPropagate, SignCastTensorCollapseShapePropagate,
      SignCastBuferizationPropagate<mlir::bufferization::ToMemrefOp>,
      SignCastBuferizationPropagate<mlir::bufferization::ToTensorOp>,
      SignCastSubviewPropagate<mlir::tensor::ExtractSliceOp,
                               mlir::RankedTensorType>,
      SignCastSubviewPropagate<mlir::memref::SubViewOp, mlir::MemRefType>,
      SignCastForPropagate, SignCastIfPropagate<imex::util::SignCastOp>,
      SignCastIfPropagate<mlir::memref::CastOp>>(context);
}

void TakeContextOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                          mlir::SymbolRefAttr initFunc,
                          mlir::SymbolRefAttr releaseFunc,
                          mlir::TypeRange resultTypes) {
  llvm::SmallVector<mlir::Type> allTypes;
  allTypes.emplace_back(imex::util::OpaqueType::get(b.getContext()));
  allTypes.append(resultTypes.begin(), resultTypes.end());
  build(b, result, allTypes, initFunc, releaseFunc);
}

llvm::Optional<int64_t> TupleExtractOp::getConstantIndex() {
  if (auto constantOp = getIndex().getDefiningOp<mlir::arith::ConstantOp>())
    return constantOp.getValue().cast<mlir::IntegerAttr>().getInt();
  return {};
}

mlir::OpFoldResult
TupleExtractOp::fold(mlir::ArrayRef<mlir::Attribute> operands) {
  // All forms of folding require a known index.
  auto index = operands[1].dyn_cast_or_null<mlir::IntegerAttr>();
  if (!index)
    return {};

  auto parent = getSource().getDefiningOp<BuildTupleOp>();
  if (!parent)
    return {};

  int64_t indexVal = index.getInt();
  mlir::ValueRange args = parent.getArgs();
  if (indexVal < 0 || indexVal >= static_cast<int64_t>(args.size()))
    return {};

  return args[indexVal];
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void EnvironmentRegionOp::getSuccessorRegions(
    llvm::Optional<unsigned> index, mlir::ArrayRef<mlir::Attribute> operands,
    mlir::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  // Branch into body if we came from parent region.
  if (!index) {
    regions.push_back(mlir::RegionSuccessor(&getRegion()));
    return;
  }

  // Branch to parent region from body.
  assert(*index == 0 && "EnvironmentRegionOp must have single region");
  regions.push_back(mlir::RegionSuccessor(getResults()));
}

/// Propagate yielded values, defined outside region.
struct EnvRegionPropagateOutsideValues
    : public mlir::OpRewritePattern<EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EnvironmentRegionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto oldResults = op.getResults();
    auto count = static_cast<unsigned>(oldResults.size());

    mlir::Block *body = &op.getRegion().front();
    auto term = mlir::cast<EnvironmentRegionYieldOp>(body->getTerminator());
    auto termArgs = term.getResults();
    assert(oldResults.size() == termArgs.size());

    // Build list of propagated and new yield args.
    llvm::SmallVector<mlir::Value> newResults(count);
    llvm::SmallVector<mlir::Value> newYieldArgs;
    for (auto i : llvm::seq(0u, count)) {
      auto arg = termArgs[i];
      if (!op.getRegion().isAncestor(arg.getParentRegion())) {
        // Value defined outside op region - use it directly instead of
        // yielding.
        newResults[i] = arg;
      } else {
        newYieldArgs.emplace_back(arg);
      }
    }

    // Same yield results count - nothing changed.
    if (newYieldArgs.size() == count)
      return mlir::failure();

    // Contruct new env region op, only yielding values that weren't propagated.
    mlir::ValueRange newYieldArgsRange(newYieldArgs);
    auto newOp = rewriter.create<EnvironmentRegionOp>(
        op->getLoc(), newYieldArgsRange.getTypes(), op.getEnvironment(),
        op.getArgs());
    mlir::Region &newRegion = newOp.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), newRegion, newRegion.end());
    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<EnvironmentRegionYieldOp>(term, newYieldArgs);
    }

    mlir::ValueRange newOpResults = newOp.getResults();

    // Fill results that weren't propagated with results of new op.
    for (auto i : llvm::seq(0u, count)) {
      if (!newResults[i]) {
        newResults[i] = newOpResults.front();
        newOpResults = newOpResults.drop_front();
      }
    }
    assert(newOpResults.empty() &&
           "Some values weren't consumed - yield args count mismatch?");

    rewriter.replaceOp(op, newResults);
    return mlir::success();
  }
};

/// Merge nested env region if parent have same environment and args.
struct MergeNestedEnvRegion
    : public mlir::OpRewritePattern<EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EnvironmentRegionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto parent = op->getParentOfType<EnvironmentRegionOp>();
    if (!parent)
      return mlir::failure();

    if (parent.getEnvironment() != op.getEnvironment() ||
        parent.getArgs() != op.getArgs())
      return mlir::failure();

    EnvironmentRegionOp::inlineIntoParent(rewriter, op);
    return mlir::success();
  }
};

/// Remove duplicated and unused env region yield args.
struct CleanupRegionYieldArgs
    : public mlir::OpRewritePattern<EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EnvironmentRegionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Block *body = &op.getRegion().front();
    auto term = mlir::cast<EnvironmentRegionYieldOp>(body->getTerminator());

    auto results = op.getResults();
    auto yieldArgs = term.getResults();
    assert(results.size() == yieldArgs.size());
    auto count = static_cast<unsigned>(results.size());

    // Build new yield args list, and mapping between old and new results
    llvm::SmallVector<mlir::Value> newYieldArgs;
    llvm::SmallVector<int> newResultsMapping(count, -1);
    llvm::SmallDenseMap<mlir::Value, int> argsMap;
    for (auto i : llvm::seq(0u, count)) {
      auto res = results[i];

      // Unused result.
      if (res.getUses().empty())
        continue;

      auto arg = yieldArgs[i];
      auto it = argsMap.find_as(arg);
      if (it == argsMap.end()) {
        // Add new result, compute index mapping for it.
        auto ind = static_cast<int>(newYieldArgs.size());
        argsMap.insert({arg, ind});
        newYieldArgs.emplace_back(arg);
        newResultsMapping[i] = ind;
      } else {
        // Duplicated result, reuse prev result index.
        newResultsMapping[i] = it->second;
      }
    }

    // Same yield results count - nothing changed.
    if (newYieldArgs.size() == count)
      return mlir::failure();

    // Contruct new env region op, only yielding values we selected.
    mlir::ValueRange newYieldArgsRange(newYieldArgs);
    auto newOp = rewriter.create<EnvironmentRegionOp>(
        op->getLoc(), newYieldArgsRange.getTypes(), op.getEnvironment(),
        op.getArgs());
    mlir::Region &newRegion = newOp.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), newRegion, newRegion.end());
    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<EnvironmentRegionYieldOp>(term, newYieldArgs);
    }

    // Contruct new result list, using mapping previously constructed.
    auto newResults = newOp.getResults();
    llvm::SmallVector<mlir::Value> newResultsToTeplace(count);
    for (auto i : llvm::seq(0u, count)) {
      auto mapInd = newResultsMapping[i];
      if (mapInd != -1)
        newResultsToTeplace[i] = newResults[mapInd];
    }

    rewriter.replaceOp(op, newResultsToTeplace);
    return mlir::success();
  }
};

/// Merge adjacent env regions.
struct MergeAdjacentRegions
    : public mlir::OpRewritePattern<EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EnvironmentRegionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Get next pos and check if it is also env region op, current op cannot be
    // last as it is not a terminator.
    auto opPos = op->getIterator();
    auto nextOp = mlir::dyn_cast<EnvironmentRegionOp>(*std::next(opPos));
    if (!nextOp)
      return mlir::failure();

    if (nextOp.getEnvironment() != op.getEnvironment() ||
        nextOp.getArgs() != op.getArgs())
      return mlir::failure();

    mlir::Block *body = &op.getRegion().front();
    auto term = mlir::cast<EnvironmentRegionYieldOp>(body->getTerminator());

    auto results = op.getResults();
    auto yieldArgs = term.getResults();
    assert(results.size() == yieldArgs.size());
    auto count = static_cast<unsigned>(results.size());

    // Check if any results from first op are being used in second one, we need
    // to replace them by direct values.
    for (auto i : llvm::seq(0u, count)) {
      auto res = results[i];
      for (auto &use : llvm::make_early_inc_range(res.getUses())) {
        auto *owner = use.getOwner();
        if (nextOp->isProperAncestor(owner)) {
          auto arg = yieldArgs[i];
          rewriter.updateRootInPlace(owner, [&]() { use.set(arg); });
        }
      }
    }

    mlir::Block *nextBody = &nextOp.getRegion().front();
    auto nextTerm =
        mlir::cast<EnvironmentRegionYieldOp>(nextBody->getTerminator());
    auto nextYieldArgs = nextTerm.getResults();

    // Contruct merged yield args list, some of the results may become unused,
    // but they will be cleaned up by other pattern.
    llvm::SmallVector<mlir::Value> newYieldArgs(count + nextYieldArgs.size());
    llvm::copy(nextYieldArgs, llvm::copy(yieldArgs, newYieldArgs.begin()));

    {
      // Merge region from second op into ferst one.
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.mergeBlockBefore(nextBody, term);
      rewriter.setInsertionPoint(term);
      rewriter.create<EnvironmentRegionYieldOp>(term->getLoc(), newYieldArgs);
      rewriter.eraseOp(term);
      rewriter.eraseOp(nextTerm);
    }

    // Contruct new env region op and steal new merged region into it.
    mlir::ValueRange newYieldArgsRange(newYieldArgs);
    auto newOp = rewriter.create<EnvironmentRegionOp>(
        op->getLoc(), newYieldArgsRange.getTypes(), op.getEnvironment(),
        op.getArgs());
    mlir::Region &newRegion = newOp.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), newRegion, newRegion.end());

    auto newResults = newOp.getResults();

    rewriter.replaceOp(op, newResults.take_front(count));
    rewriter.replaceOp(nextOp, newResults.drop_front(count));
    return mlir::success();
  }
};

void EnvironmentRegionOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.insert<EnvRegionPropagateOutsideValues, MergeNestedEnvRegion,
                 CleanupRegionYieldArgs, MergeAdjacentRegions>(context);
}

void EnvironmentRegionOp::inlineIntoParent(mlir::PatternRewriter &builder,
                                           EnvironmentRegionOp op) {
  mlir::Block *block = &op.getRegion().front();
  auto term = mlir::cast<EnvironmentRegionYieldOp>(block->getTerminator());
  auto args = llvm::to_vector(term.getResults());
  builder.eraseOp(term);
  builder.mergeBlockBefore(block, op);
  builder.replaceOp(op, args);
}

void EnvironmentRegionOp::build(
    ::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
    ::mlir::Attribute environment, ::mlir::ValueRange args,
    ::mlir::TypeRange results,
    ::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>
        bodyBuilder) {
  build(odsBuilder, odsState, results, environment, args);
  mlir::Region *bodyRegion = odsState.regions.back().get();
  if (bodyBuilder) {
    bodyRegion->push_back(new mlir::Block);
    mlir::Block &bodyBlock = bodyRegion->front();

    mlir::OpBuilder::InsertionGuard guard(odsBuilder);
    odsBuilder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(odsBuilder, odsState.location);
  }
  ensureTerminator(*bodyRegion, odsBuilder, odsState.location);
}

mlir::LogicalResult BitcastOp::verify() {
  auto srcType = getSource().getType();
  auto dstType = getResult().getType();
  if (srcType.isIntOrFloat() && dstType.isIntOrFloat() &&
      srcType.getIntOrFloatBitWidth() != dstType.getIntOrFloatBitWidth())
    return emitError("Bitcast element size mismatch.");
  return mlir::success();
}

mlir::OpFoldResult
BitcastOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  auto src = getSource();
  auto srcType = src.getType();
  auto dstType = getResult().getType();
  if (srcType == dstType)
    return src;

  return nullptr;
}

mlir::LogicalResult MemrefBitcastOp::verify() {
  auto srcType = getSource().getType().cast<mlir::MemRefType>();
  auto dstType = getResult().getType().cast<mlir::MemRefType>();
  if (srcType.getLayout() != dstType.getLayout())
    return emitError("Bitcast layout mismatch.");
  if (srcType.getMemorySpace() != dstType.getMemorySpace())
    return emitError("Bitcast memory space mismatch.");

  auto srcElem = srcType.getElementType();
  auto dstElem = dstType.getElementType();
  if (srcElem.isIntOrFloat() && dstElem.isIntOrFloat() &&
      srcElem.getIntOrFloatBitWidth() != dstElem.getIntOrFloatBitWidth())
    return emitError("Bitcast element size mismatch.");
  return mlir::success();
}

mlir::OpFoldResult
MemrefBitcastOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  auto src = getSource();
  auto srcType = src.getType();
  auto dstType = getResult().getType();
  if (srcType == dstType)
    return src;

  return nullptr;
}

} // namespace util
} // namespace imex

#include "imex/Dialect/imex_util/ImexUtilOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "imex/Dialect/imex_util/ImexUtilOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "imex/Dialect/imex_util/ImexUtilOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "imex/Dialect/imex_util/ImexUtilOpsTypes.cpp.inc"

#include "imex/Dialect/imex_util/ImexUtilOpsEnums.cpp.inc"
