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

#include "mlir-extensions/dialect/plier/dialect.hpp"
#include "mlir-extensions/dialect/plier_util/dialect.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/InliningUtils.h>

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>

#include <llvm/ADT/TypeSwitch.h>

#include "mlir-extensions/transforms/const_utils.hpp"

namespace MemoryEffects = ::mlir::MemoryEffects;

namespace {
struct PlierUtilInlinerInterface : public mlir::DialectInlinerInterface {
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

namespace plier {
void PlierUtilDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-extensions/dialect/plier_util/PlierUtilOps.cpp.inc"
      >();
  addTypes<OpaqueType>();
  addInterfaces<PlierUtilInlinerInterface>();
}

mlir::Type PlierUtilDialect::parseType(mlir::DialectAsmParser &parser) const {
  parser.emitError(parser.getNameLoc(), "unknown type");
  return mlir::Type();
}

void PlierUtilDialect::printType(mlir::Type type,
                                 mlir::DialectAsmPrinter &os) const {
  llvm::TypeSwitch<mlir::Type>(type)
      .Case<plier::OpaqueType>([&](auto) { os << "OpaqueType"; })
      .Default([](auto) { llvm_unreachable("unexpected type"); });
}

mlir::Operation *PlierUtilDialect::materializeConstant(mlir::OpBuilder &builder,
                                                       mlir::Attribute value,
                                                       mlir::Type type,
                                                       mlir::Location loc) {
  if (mlir::arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<mlir::arith::ConstantOp>(loc, type, value);

  return nullptr;
}

namespace {
template <typename DimOp, typename ExpandOp>
struct DimExpandShape : public mlir::OpRewritePattern<DimOp> {
  using mlir::OpRewritePattern<DimOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(DimOp op, mlir::PatternRewriter &rewriter) const override {
    auto es = op.source().template getDefiningOp<ExpandOp>();
    if (!es)
      return mlir::failure();

    auto indexAttr = mlir::getConstantIntValue(op.index());
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

      return llvm::None;
    }();

    if (!srcIndexAttr)
      return mlir::failure();

    auto shape = type.getShape();
    auto srcIndex = *srcIndexAttr;
    for (auto i : reassoc[srcIndex])
      if (i != dstIndex && shape[i] != 1)
        return mlir::failure();

    auto src = es.src();
    rewriter.replaceOpWithNewOp<DimOp>(op, src, srcIndex);
    return mlir::success();
  }
};

struct DimInsertSlice : public mlir::OpRewritePattern<mlir::tensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::DimOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto insertSlice = op.source().getDefiningOp<mlir::tensor::InsertSliceOp>();
    if (!insertSlice)
      return mlir::failure();

    auto indexAttr = mlir::getConstantIntValue(op.index());
    if (!indexAttr)
      return mlir::failure();

    auto index = *indexAttr;

    auto sizes = insertSlice.getMixedSizes();
    if (index < 0 || static_cast<size_t>(index) >= sizes.size())
      return mlir::failure();

    auto val = [&]() -> mlir::Value {
      auto v = sizes[index];
      if (v.is<mlir::Value>())
        return v.get<mlir::Value>();

      auto attr = v.get<mlir::Attribute>().cast<mlir::IntegerAttr>();
      return rewriter.create<mlir::arith::ConstantIndexOp>(
          op->getLoc(), attr.getValue().getSExtValue());
    }();

    rewriter.replaceOp(op, val);
    return mlir::success();
  }
};
} // namespace

void PlierUtilDialect::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results) const {
  results.add<DimExpandShape<mlir::tensor::DimOp, mlir::tensor::ExpandShapeOp>,
              DimExpandShape<mlir::memref::DimOp, mlir::memref::ExpandShapeOp>,
              DimInsertSlice>(getContext());
}

OpaqueType OpaqueType::get(mlir::MLIRContext *context) {
  assert(context);
  return Base::get(context);
}

void EnforceShapeOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state, mlir::Value value,
                           mlir::ValueRange shape) {
  EnforceShapeOp::build(builder, state, value.getType(), value, shape);
}

mlir::OpFoldResult
EnforceShapeOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  operands = operands.drop_front();
  auto num_dims = static_cast<unsigned>(operands.size());
  auto src_type = getType().cast<mlir::ShapedType>();
  llvm::SmallVector<int64_t> final_shape(num_dims, -1);
  if (src_type.hasRank()) {
    auto shape = src_type.getShape();
    if (shape.size() != num_dims) {
      return nullptr;
    }
    final_shape.assign(shape.begin(), shape.end());
  }
  bool changed = false;
  for (unsigned i = 0; i < num_dims; ++i) {
    if (auto attr = operands[i].dyn_cast_or_null<mlir::IntegerAttr>()) {
      auto val = attr.getInt();
      if (val != -1) {
        if (final_shape[i] != -1) {
          if (final_shape[i] != val) {
            return nullptr;
          }
        } else {
          changed = true;
          final_shape[i] = val;
        }
      }
    }
  }

  if (changed) {
    auto final_type =
        mlir::RankedTensorType::get(final_shape, src_type.getElementType());
    result().setType(final_type);
    return result();
  }
  return nullptr;
}

namespace {
struct EnforceShapeDim : public mlir::OpRewritePattern<mlir::memref::DimOp> {
  using mlir::OpRewritePattern<mlir::memref::DimOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::DimOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto enforce_op = mlir::dyn_cast_or_null<plier::EnforceShapeOp>(
        op.source().getDefiningOp());
    if (!enforce_op) {
      return mlir::failure();
    }
    auto const_ind = plier::getConstVal<mlir::IntegerAttr>(op.index());
    if (!const_ind) {
      return mlir::failure();
    }
    auto index = const_ind.getInt();
    if (index < 0 || index >= static_cast<int64_t>(enforce_op.sizes().size())) {
      return mlir::failure();
    }

    rewriter.replaceOp(op, enforce_op.sizes()[static_cast<unsigned>(index)]);
    return mlir::success();
  }
};
} // namespace

void EnforceShapeOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<EnforceShapeDim>(context);
}

mlir::LogicalResult
ParallelOp::moveOutOfLoop(mlir::ArrayRef<mlir::Operation *> ops) {
  for (mlir::Operation *op : ops) {
    op->moveBefore(*this);
  }
  return mlir::success();
}

mlir::Region &ParallelOp::getLoopBody() { return region(); }

bool ParallelOp::isDefinedOutsideOfLoop(mlir::Value value) {
  return !region().isAncestor(value.getParentRegion());
}

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
      odsBuilder.getI32VectorAttr({static_cast<int32_t>(lowerBounds.size()),
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
  if (auto parent = val.getDefiningOp<plier::ChangeLayoutOp>())
    return parent.source();

  return {};
}

mlir::OpFoldResult
ChangeLayoutOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  auto src = source();
  auto thisType = getType();
  do {
    if (thisType == src.getType())
      return src;
  } while ((src = getChangeLayoutParent(src)));

  return nullptr;
}

namespace {
static bool canTransformLayoutCast(mlir::MemRefType srcType,
                                   mlir::MemRefType dstType) {
  if (!mlir::memref::CastOp::areCastCompatible(srcType, dstType))
    return false;

  int64_t srcOffset, dstOffset;
  llvm::SmallVector<int64_t> srcStrides, dstStrides;
  if (mlir::failed(mlir::getStridesAndOffset(srcType, srcStrides, srcOffset)) ||
      mlir::failed(mlir::getStridesAndOffset(dstType, dstStrides, dstOffset)))
    return false;

  auto isStrideCompatible = [](int64_t src, int64_t dst) {
    auto isStatic = [](int64_t v) {
      return !mlir::ShapedType::isDynamicStrideOrOffset(v);
    };
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

struct ChangeLayoutIdentity
    : public mlir::OpRewritePattern<plier::ChangeLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::ChangeLayoutOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.source();
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
    auto cl = op.source().getDefiningOp<plier::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::DimOp>(op, cl.source(),
                                                     op.index());
    return mlir::success();
  }
};

struct ChangeLayoutExtractMetadata
    : public mlir::OpRewritePattern<plier::ExtractMemrefMetadataOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::ExtractMemrefMetadataOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.source().getDefiningOp<plier::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<plier::ExtractMemrefMetadataOp>(
        op, cl.source(), op.dimIndex().getSExtValue());
    return mlir::success();
  }
};

struct ChangeLayoutClone
    : public mlir::OpRewritePattern<mlir::bufferization::CloneOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::bufferization::CloneOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.input().getDefiningOp<plier::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    auto src = cl.source();
    auto dstType = op.getType();

    auto loc = op.getLoc();
    auto res = rewriter.createOrFold<mlir::bufferization::CloneOp>(loc, src);
    rewriter.replaceOpWithNewOp<plier::ChangeLayoutOp>(op, dstType, res);
    return mlir::success();
  }
};

struct PropagateCloneType
    : public mlir::OpRewritePattern<mlir::bufferization::CloneOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::bufferization::CloneOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.input();
    auto srcType = src.getType();
    auto dstType = op.getType();
    if (srcType == dstType)
      return mlir::failure();

    auto loc = op.getLoc();
    auto res = rewriter.createOrFold<mlir::bufferization::CloneOp>(loc, src);
    rewriter.replaceOpWithNewOp<plier::ChangeLayoutOp>(op, dstType, res);
    return mlir::success();
  }
};

struct ChangeLayoutCast : public mlir::OpRewritePattern<mlir::memref::CastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::CastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.source().getDefiningOp<plier::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    auto src = cl.source();
    auto srcType = src.getType();
    auto dstType = op.getType();
    if (srcType == dstType) {
      rewriter.replaceOp(op, src);
      return mlir::success();
    }

    if (mlir::memref::CastOp::areCastCompatible(srcType, dstType)) {
      rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(op, dstType, src);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct ChangeLayoutSignCast : public mlir::OpRewritePattern<plier::SignCastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::SignCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.value().getDefiningOp<plier::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    auto src = cl.source();
    auto srcType = src.getType().cast<mlir::MemRefType>();
    auto oldType = op.getType().cast<mlir::MemRefType>();
    auto newType = mlir::MemRefType::get(
        srcType.getShape(), oldType.getElementType(), srcType.getLayout());

    auto loc = op.getLoc();
    auto newOp = rewriter.createOrFold<plier::SignCastOp>(loc, newType, src);

    if (oldType != newType)
      newOp = rewriter.createOrFold<plier::ChangeLayoutOp>(loc, oldType, newOp);

    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

struct ChangeLayoutReduceRank
    : public mlir::OpRewritePattern<plier::ReduceRankOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::ReduceRankOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.source().getDefiningOp<plier::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    auto loc = op.getLoc();
    auto newOp = rewriter.createOrFold<plier::ReduceRankOp>(loc, cl.source(),
                                                            op.getMapping());
    auto oldType = op.getType();
    auto newType = newOp.getType();
    if (oldType != newType)
      newOp = rewriter.createOrFold<plier::ChangeLayoutOp>(loc, oldType, newOp);

    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

struct ChangeLayoutLoad : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.memref().getDefiningOp<plier::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, cl.source(),
                                                      op.indices());
    return mlir::success();
  }
};

struct ChangeLayoutStore
    : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.memref().getDefiningOp<plier::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(
        op, op.value(), cl.source(), op.indices());
    return mlir::success();
  }
};

struct ChangeLayoutSubview
    : public mlir::OpRewritePattern<mlir::memref::SubViewOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::SubViewOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.source().getDefiningOp<plier::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    auto offsets = op.getMixedOffsets();
    auto sizes = op.getMixedSizes();
    auto strides = op.getMixedStrides();

    auto src = cl.source();
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
              dstRank, srcType, offsets, sizes, strides);
        }()
            .cast<mlir::MemRefType>();

    auto loc = op.getLoc();
    auto newSubview = rewriter.createOrFold<mlir::memref::SubViewOp>(
        loc, newDstType, src, offsets, sizes, strides);
    if (newDstType != dstType)
      newSubview = rewriter.createOrFold<plier::ChangeLayoutOp>(loc, dstType,
                                                                newSubview);

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
      mlir::ValueRange args = (isInputs ? op.inputs() : op.outputs());
      auto count = static_cast<unsigned>(args.size());
      newOperands.resize(count);
      bool needUpdate = false;
      for (auto i : llvm::seq(0u, count)) {
        auto arg = args[i];
        auto cl = arg.getDefiningOp<plier::ChangeLayoutOp>();
        if (cl) {
          assert(arg.getType().isa<mlir::MemRefType>());
          assert(cl.source().getType().isa<mlir::MemRefType>());
          newOperands[i] = cl.source();
          needUpdate = true;
          changed = true;
        } else {
          newOperands[i] = arg;
        }
      }

      if (needUpdate) {
        rewriter.updateRootInPlace(op, [&]() {
          (isInputs ? op.inputsMutable() : op.outputsMutable())
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
    auto clOutput = output.getDefiningOp<plier::ChangeLayoutOp>();
    if (!clOutput)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::linalg::FillOp>(op, op.value(),
                                                      clOutput.source());
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

        auto cl = arg.getDefiningOp<plier::ChangeLayoutOp>();
        if (!cl)
          continue;

        auto src = cl.source();
        auto srcType = src.getType();

        if (!mlir::memref::CastOp::areCastCompatible(origType, srcType))
          continue;

        rewriter.updateRootInPlace(clYield,
                                   [&]() { clYield.setOperand(i, src); });

        auto otherArg = otherYield.getResults()[i];
        rewriter.setInsertionPoint(otherYield);
        auto otherRes = rewriter.createOrFold<mlir::memref::CastOp>(
            otherYield.getLoc(), srcType, otherArg);

        rewriter.updateRootInPlace(
            otherYield, [&]() { otherYield.setOperand(i, otherRes); });

        newType = srcType;
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
                rewriter.create<plier::ChangeLayoutOp>(loc, origType, res);
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
    return llvm::None;

  int dimIndex = -1;
  for (auto it : llvm::enumerate(type.getShape())) {
    auto i = static_cast<int>(it.index());
    auto dim = it.value();
    if (dim == mlir::ShapedType::kDynamicSize) {
      if (dimIndex != -1)
        return llvm::None;

      dimIndex = i;
    } else if (dim != 1) {
      return llvm::None;
    }
  }

  if (dimIndex != -1)
    return static_cast<unsigned>(dimIndex);

  return llvm::None;
}

struct ChangeLayout1DReshape
    : public mlir::OpRewritePattern<mlir::memref::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ReshapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto source = op.source();
    auto shape = op.shape();
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
    auto resType = view.getType().cast<mlir::MemRefType>();
    if (resType.getRank() > dstType.getRank()) {
      // TODO: Rank-reducing subview
      const int32_t mapping[1] = {static_cast<int32_t>(*srcDimIndex)};
      view = rewriter.createOrFold<plier::ReduceRankOp>(loc, view, mapping);
    }
    rewriter.replaceOpWithNewOp<plier::ChangeLayoutOp>(op, dstType, view);
    return mlir::success();
  }
};

struct ChangeLayoutSliceGetItem
    : public mlir::OpRewritePattern<plier::SliceGetItemOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::SliceGetItemOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.array().getDefiningOp<plier::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<plier::SliceGetItemOp>(
        op, op.getType(), op.slice(), cl.source(), op.index(), op.dim());
    return mlir::success();
  }
};

struct ChangeLayoutCopy : public mlir::OpRewritePattern<mlir::memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::CopyOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto input = op.source();
    auto output = op.target();
    auto clInput = input.getDefiningOp<plier::ChangeLayoutOp>();
    auto clOutput = output.getDefiningOp<plier::ChangeLayoutOp>();
    if (!clInput && !clOutput)
      return mlir::failure();

    if (clInput)
      input = clInput.source();

    if (clOutput)
      output = clOutput.source();

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
    auto cl = op.src().getDefiningOp<plier::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    auto dstType = op.getType().cast<mlir::MemRefType>();
    if (!dstType.getLayout().isIdentity())
      return mlir::failure();

    auto src = cl.source();
    auto srcType = src.getType().cast<mlir::MemRefType>();
    if (!mlir::isStrided(srcType))
      return mlir::failure();

    llvm::SmallVector<int64_t> strides(
        dstType.getRank(), mlir::ShapedType::kDynamicStrideOrOffset);
    auto newDstMap = mlir::makeStridedLinearLayoutMap(
        strides, mlir::ShapedType::kDynamicStrideOrOffset, op->getContext());
    if (!newDstMap)
      return mlir::failure();

    auto newDstType = mlir::MemRefType::get(
        dstType.getShape(), dstType.getElementType(), newDstMap);

    auto loc = op->getLoc();
    mlir::Value newOp = rewriter.create<mlir::memref::ExpandShapeOp>(
        loc, newDstType, src, op.getReassociationIndices());
    rewriter.replaceOpWithNewOp<plier::ChangeLayoutOp>(op, dstType, newOp);
    return mlir::success();
  }
};

} // namespace

void ChangeLayoutOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results
      .insert<ChangeLayoutIdentity, ChangeLayoutReduceRank, ChangeLayoutDim,
              ChangeLayoutExtractMetadata, ChangeLayoutClone,
              PropagateCloneType, ChangeLayoutCast, ChangeLayoutSignCast,
              ChangeLayoutLoad, ChangeLayoutStore, ChangeLayoutSubview,
              ChangeLayoutLinalgGeneric, ChangeLayoutLinalgFill, ChangeLayoutIf,
              ChangeLayout1DReshape, ChangeLayoutSliceGetItem, ChangeLayoutCopy,
              ChangeLayoutExpandShape>(context);
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
      &foldPrevCast<SignCastOp>,
      &foldPrevCast<CastOp>,
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
  auto attrOperand = operands.front();
  if (attrOperand && attrOperand.getType() == thisType)
    return attrOperand;

  auto arg = value();
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
        mlir::dyn_cast_or_null<plier::SignCastOp>(op.source().getDefiningOp());
    if (!castOp)
      return mlir::failure();

    auto val = castOp.value();
    rewriter.replaceOpWithNewOp<Op>(op, val, op.index());
    return mlir::success();
  }
};

struct SignCastUndefPropagate
    : public mlir::OpRewritePattern<plier::SignCastOp> {
  using mlir::OpRewritePattern<plier::SignCastOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::SignCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto undefOp = op.value().getDefiningOp<plier::UndefOp>();
    if (!undefOp)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<plier::UndefOp>(op, op.getType());
    return mlir::success();
  }
};

struct SignCastTensorCastPropagate
    : public mlir::OpRewritePattern<plier::SignCastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::SignCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto tensorCast = op.value().getDefiningOp<mlir::tensor::CastOp>();
    if (!tensorCast)
      return mlir::failure();

    auto srcType = tensorCast.source().getType().cast<mlir::TensorType>();
    auto dstType = tensorCast.getType().cast<mlir::TensorType>();
    if (srcType.getElementType() != dstType.getElementType() ||
        !srcType.hasRank() || !dstType.hasRank())
      return mlir::failure();

    auto finalType = op.getType().cast<mlir::TensorType>();
    auto finalElemType = finalType.getElementType();

    auto newSrcType = srcType.clone(finalElemType);
    auto newDstType = dstType.clone(finalElemType);

    auto loc = op.getLoc();
    auto casted = rewriter.createOrFold<plier::SignCastOp>(loc, newSrcType,
                                                           tensorCast.source());
    rewriter.replaceOpWithNewOp<mlir::tensor::CastOp>(op, newDstType, casted);

    return mlir::success();
  }
};

struct SignCastMemrefCastPropagate
    : public mlir::OpRewritePattern<plier::SignCastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::SignCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto memrefCast = op.value().getDefiningOp<mlir::memref::CastOp>();
    if (!memrefCast)
      return mlir::failure();

    auto srcType = memrefCast.source().getType().cast<mlir::MemRefType>();
    auto dstType = memrefCast.getType().cast<mlir::MemRefType>();
    if (srcType.getElementType() != dstType.getElementType() ||
        !srcType.hasRank() || !dstType.hasRank())
      return mlir::failure();

    auto finalType = op.getType().cast<mlir::MemRefType>();
    auto finalElemType = finalType.getElementType();

    auto newSrcType = srcType.clone(finalElemType);
    auto newDstType = dstType.clone(finalElemType);

    auto loc = op.getLoc();
    auto casted = rewriter.createOrFold<plier::SignCastOp>(loc, newSrcType,
                                                           memrefCast.source());
    rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(op, newDstType, casted);

    return mlir::success();
  }
};

template <typename Op>
struct SignCastAllocPropagate
    : public mlir::OpRewritePattern<plier::SignCastOp> {
  using mlir::OpRewritePattern<plier::SignCastOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::SignCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto alloc = op.value().getDefiningOp<Op>();
    if (!alloc || !alloc->hasOneUse())
      return mlir::failure();

    auto dstType = op.getType().cast<mlir::MemRefType>();
    rewriter.replaceOpWithNewOp<Op>(op, dstType, alloc.dynamicSizes(),
                                    alloc.symbolOperands(),
                                    alloc.alignmentAttr());
    rewriter.eraseOp(alloc);
    return mlir::success();
  }
};

struct SignCastTensorFromElementsPropagate
    : public mlir::OpRewritePattern<plier::SignCastOp> {
  using mlir::OpRewritePattern<plier::SignCastOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::SignCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto fromElements =
        op.value().getDefiningOp<mlir::tensor::FromElementsOp>();
    if (!fromElements)
      return mlir::failure();

    auto loc = fromElements->getLoc();
    auto dstType = op.getType().cast<mlir::TensorType>();
    auto elemType = dstType.getElementType();
    auto elements = fromElements.elements();
    auto count = static_cast<unsigned>(elements.size());
    llvm::SmallVector<mlir::Value> castedVals(count);
    for (auto i : llvm::seq(0u, count))
      castedVals[i] =
          rewriter.create<plier::SignCastOp>(loc, elemType, elements[i]);

    rewriter.replaceOpWithNewOp<mlir::tensor::FromElementsOp>(op, castedVals);
    return mlir::success();
  }
};

struct SignCastTensorCollapseShapePropagate
    : public mlir::OpRewritePattern<plier::SignCastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::SignCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto prevOp = op.value().getDefiningOp<mlir::tensor::CollapseShapeOp>();
    if (!prevOp)
      return mlir::failure();

    auto src = prevOp.src();
    auto srcType = src.getType().cast<mlir::TensorType>();
    auto dstType = op.getType().cast<mlir::TensorType>();

    auto newSrcType = srcType.clone(dstType.getElementType());
    auto newDstType = dstType.clone(dstType.getElementType());

    auto loc = prevOp->getLoc();
    auto newSrc = rewriter.create<plier::SignCastOp>(loc, newSrcType, src);
    rewriter.replaceOpWithNewOp<mlir::tensor::CollapseShapeOp>(
        op, newDstType, newSrc, prevOp.reassociation());
    return mlir::success();
  }
};

struct SignCastTensorToMemrefPropagate
    : public mlir::OpRewritePattern<plier::SignCastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::SignCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto toMemref = op.value().getDefiningOp<mlir::bufferization::ToMemrefOp>();
    if (!toMemref)
      return mlir::failure();

    auto tensor = toMemref.tensor();
    auto tensorType = tensor.getType().cast<mlir::TensorType>();
    auto dstType = op.getType().cast<mlir::MemRefType>();

    auto newTensorType = tensorType.clone(dstType.getElementType());

    auto loc = toMemref->getLoc();
    auto newTensor =
        rewriter.create<plier::SignCastOp>(loc, newTensorType, tensor);
    rewriter.replaceOpWithNewOp<mlir::bufferization::ToMemrefOp>(op, dstType,
                                                                 newTensor);
    return mlir::success();
  }
};

struct SignCastMemrefToTensorPropagate
    : public mlir::OpRewritePattern<plier::SignCastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::SignCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto toTensor = op.value().getDefiningOp<mlir::bufferization::ToTensorOp>();
    if (!toTensor)
      return mlir::failure();

    auto memref = toTensor.memref();
    auto memrefType = memref.getType().cast<mlir::MemRefType>();
    auto dstType = op.getType().cast<mlir::TensorType>();

    auto newMemrefType = memrefType.clone(dstType.getElementType());

    auto loc = toTensor->getLoc();
    auto newMemref =
        rewriter.create<plier::SignCastOp>(loc, newMemrefType, memref);
    rewriter.replaceOpWithNewOp<mlir::bufferization::ToTensorOp>(op, dstType,
                                                                 newMemref);
    return mlir::success();
  }
};

} // namespace

void SignCastOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                             ::mlir::MLIRContext *context) {
  results.insert<
      SignCastDimPropagate<mlir::tensor::DimOp>,
      SignCastDimPropagate<mlir::memref::DimOp>, SignCastUndefPropagate,
      SignCastTensorCastPropagate, SignCastMemrefCastPropagate,
      SignCastAllocPropagate<mlir::memref::AllocOp>,
      SignCastAllocPropagate<mlir::memref::AllocaOp>,
      SignCastTensorFromElementsPropagate, SignCastTensorCollapseShapePropagate,
      SignCastTensorToMemrefPropagate, SignCastMemrefToTensorPropagate>(
      context);
}

void ReduceRankOp::build(::mlir::OpBuilder &odsBuilder,
                         ::mlir::OperationState &odsState, ::mlir::Value src,
                         ::mlir::ArrayRef<int32_t> mapping) {
  assert(src.getType().isa<mlir::ShapedType>());
  auto srcType = src.getType().cast<mlir::ShapedType>();
  assert(srcType.hasRank());
  auto srcRank = static_cast<unsigned>(srcType.getRank());
  assert(!mapping.empty());
  assert(llvm::all_of(mapping, [&](int32_t val) {
    return val >= 0 && val < static_cast<int32_t>(srcRank);
  }));
  auto mapAttr = odsBuilder.getI32ArrayAttr(mapping);
  auto srcShape = srcType.getShape();
  llvm::SmallVector<int64_t> shape(mapping.size());
  for (auto it : llvm::enumerate(mapping))
    shape[it.index()] = srcShape[static_cast<size_t>(it.value())];

  if (auto tensorType = srcType.dyn_cast<mlir::RankedTensorType>()) {
    auto retType = mlir::RankedTensorType::get(
        shape, tensorType.getElementType(), tensorType.getEncoding());
    build(odsBuilder, odsState, retType, src, mapAttr);
  } else if (auto memrefType = srcType.dyn_cast<mlir::MemRefType>()) {
    auto affineMap = [&]() {
      mlir::AffineMap ret;
      if (!memrefType.getLayout().isIdentity()) {
        auto affineMap = memrefType.getLayout().getAffineMap();
        auto context = odsBuilder.getContext();
        llvm::SmallVector<mlir::AffineExpr> dimReplacements(srcRank);
        llvm::SmallVector<mlir::AffineExpr> symReplacements(srcRank + 1);
        symReplacements[0] = mlir::getAffineSymbolExpr(0, context);
        for (auto i : llvm::seq(0u, srcRank)) {
          auto it = llvm::find(mapping, i);
          if (it != mapping.end()) {
            auto srcIndex = static_cast<unsigned>(it - mapping.begin());
            dimReplacements[i] = mlir::getAffineDimExpr(srcIndex, context);
            symReplacements[i + 1] =
                mlir::getAffineSymbolExpr(srcIndex + 1, context);
          } else {
            dimReplacements[i] = mlir::getAffineConstantExpr(0, context);
            symReplacements[i + 1] = mlir::getAffineConstantExpr(0, context);
          }
        }
        auto dstRank = static_cast<unsigned>(mapping.size());
        auto resMap = affineMap.replaceDimsAndSymbols(
            dimReplacements, symReplacements, dstRank, dstRank + 1);
        ret = mlir::simplifyAffineMap(resMap);
      }
      return ret;
    }();

    auto retType =
        mlir::MemRefType::get(shape, memrefType.getElementType(), affineMap,
                              memrefType.getMemorySpace());
    build(odsBuilder, odsState, retType, src, mapAttr);
  } else {
    llvm_unreachable("ReduceRankOp: Invalid src type");
  }
}

mlir::OpFoldResult
ReduceRankOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  auto src = source();
  if (src.getType() == getType()) {
    return src;
  }
  return nullptr;
}

llvm::SmallVector<int32_t> ReduceRankOp::getMapping() {
  auto m = mapping();
  llvm::SmallVector<int32_t> ret(m.size());
  llvm::transform(m, ret.begin(), [](mlir::Attribute a) {
    return a.cast<mlir::IntegerAttr>().getValue().getSExtValue();
  });
  return ret;
}

namespace {
template <typename Op>
struct ReduceRankDimPropagate : public mlir::OpRewritePattern<Op> {
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    auto index = mlir::getConstantIntValue(op.index());
    if (!index)
      return mlir::failure();

    auto prev = op.source().template getDefiningOp<plier::ReduceRankOp>();
    if (!prev)
      return mlir::failure();

    auto mappedArg = prev.mapping()[*index]
                         .template cast<mlir::IntegerAttr>()
                         .getValue()
                         .getSExtValue();
    rewriter.replaceOpWithNewOp<Op>(op, prev.source(), mappedArg);
    return mlir::success();
  }
};

static auto mapReduceRankIndices(mlir::OpBuilder &builder, mlir::Location loc,
                                 plier::ReduceRankOp src,
                                 mlir::ValueRange srcIndices) {
  auto srcMemref = src.getViewSource();
  auto srcMemrefType = srcMemref.getType().cast<mlir::MemRefType>();
  auto rank = static_cast<unsigned>(srcMemrefType.getRank());
  auto zero = builder.createOrFold<mlir::arith::ConstantIndexOp>(loc, 0);
  auto mapping = src.getMapping();
  llvm::SmallVector<mlir::Value> indices(rank);
  for (auto i : llvm::seq(0u, rank)) {
    auto it = llvm::find(mapping, static_cast<int32_t>(i));
    if (mapping.end() == it) {
      indices[i] = zero;
    } else {
      auto dstIndex = static_cast<size_t>(it - mapping.begin());
      indices[i] = srcIndices[dstIndex];
    }
  }
  return indices;
}

struct ReduceRankLoadPropagate
    : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.memref().getDefiningOp<plier::ReduceRankOp>();
    if (!src)
      return mlir::failure();

    auto indices =
        mapReduceRankIndices(rewriter, op.getLoc(), src, op.indices());
    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, src.getViewSource(),
                                                      indices);
    return mlir::success();
  }
};

struct ReduceRankStorePropagate
    : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.memref().getDefiningOp<plier::ReduceRankOp>();
    if (!src)
      return mlir::failure();

    auto indices =
        mapReduceRankIndices(rewriter, op.getLoc(), src, op.indices());
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(
        op, op.value(), src.getViewSource(), indices);
    return mlir::success();
  }
};
} // namespace

void ReduceRankOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<ReduceRankDimPropagate<mlir::tensor::DimOp>,
                 ReduceRankDimPropagate<mlir::memref::DimOp>,
                 ReduceRankLoadPropagate, ReduceRankStorePropagate>(context);
}

void ExtractMemrefMetadataOp::build(::mlir::OpBuilder &odsBuilder,
                                    ::mlir::OperationState &odsState,
                                    ::mlir::Value src, int64_t dim) {
  assert(dim >= 0 && dim < src.getType().cast<mlir::MemRefType>().getRank());
  ExtractMemrefMetadataOp::build(odsBuilder, odsState,
                                 odsBuilder.getIndexType(), src,
                                 odsBuilder.getIndexAttr(dim));
}

void ExtractMemrefMetadataOp::build(::mlir::OpBuilder &odsBuilder,
                                    ::mlir::OperationState &odsState,
                                    ::mlir::Value src) {
  ExtractMemrefMetadataOp::build(odsBuilder, odsState,
                                 odsBuilder.getIndexType(), src,
                                 odsBuilder.getIndexAttr(-1));
}

mlir::OpFoldResult
ExtractMemrefMetadataOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  auto idx = dimIndex().getSExtValue();
  assert(idx >= -1);
  auto src = source();

  int64_t offset;
  llvm::SmallVector<int64_t> strides;
  if (mlir::succeeded(mlir::getStridesAndOffset(
          src.getType().cast<mlir::MemRefType>(), strides, offset))) {
    mlir::Builder builder(getContext());
    if (idx == -1 && !mlir::ShapedType::isDynamicStrideOrOffset(offset)) {
      return builder.getIndexAttr(offset);
    } else if (idx >= 0 && idx < static_cast<int64_t>(strides.size()) &&
               !mlir::ShapedType::isDynamicStrideOrOffset(
                   strides[static_cast<unsigned>(idx)])) {
      return builder.getIndexAttr(strides[static_cast<unsigned>(idx)]);
    }
  }

  if (auto reintr = src.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
    if (idx == -1) {
      auto offsets = reintr.getMixedOffsets();
      if (offsets.size() == 1)
        return offsets.front();

      return nullptr;
    }

    auto strides = reintr.getMixedStrides();
    if (static_cast<unsigned>(idx) < strides.size())
      return strides[static_cast<unsigned>(idx)];

    return nullptr;
  }

  if (auto cast = src.getDefiningOp<mlir::memref::CastOp>()) {
    auto newSrc = cast.source();
    sourceMutable().assign(newSrc);
    return getResult();
  }

  if (auto reduceRank = src.getDefiningOp<plier::ReduceRankOp>()) {
    auto newSrc = reduceRank.source();
    if (idx == -1) {
      sourceMutable().assign(newSrc);
      return getResult();
    }

    auto mapping = reduceRank.getMapping();
    if (static_cast<unsigned>(idx) < mapping.size()) {
      auto newIdx = mapping[static_cast<unsigned>(idx)];
      assert(newIdx >= 0);
      sourceMutable().assign(newSrc);
      auto type = dimIndexAttr().getType();
      dimIndexAttr(mlir::IntegerAttr::get(type, newIdx));
      return getResult();
    }

    return nullptr;
  }

  if (auto cast = src.getDefiningOp<mlir::memref::CastOp>()) {
    auto castSrc = cast.source();
    auto castSrcType = castSrc.getType().cast<mlir::ShapedType>();
    auto srcType = src.getType().cast<mlir::ShapedType>();
    if (castSrcType.hasRank() && srcType.hasRank() &&
        castSrcType.getRank() == srcType.getRank()) {
      sourceMutable().assign(castSrc);
      return getResult();
    }

    return nullptr;
  }

  return nullptr;
}

namespace {
template <typename Op, typename DelOp>
struct RemoveUnusedOp : public mlir::OpRewritePattern<Op> {
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    for (auto user : op->getUsers()) {
      if (!mlir::isa<DelOp>(user))
        return mlir::failure();
    }

    for (auto user : llvm::make_early_inc_range(op->getUsers())) {
      assert(user->getNumResults() == 0);
      rewriter.eraseOp(user);
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};
} // namespace

void CreateGpuStreamOp::build(::mlir::OpBuilder &odsBuilder,
                              ::mlir::OperationState &odsState) {
  auto ctx = odsBuilder.getContext();
  CreateGpuStreamOp::build(odsBuilder, odsState, plier::OpaqueType::get(ctx));
}

void CreateGpuStreamOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<RemoveUnusedOp<CreateGpuStreamOp, DestroyGpuStreamOp>>(
      context);
}

void LoadGpuModuleOp::build(::mlir::OpBuilder &odsBuilder,
                            ::mlir::OperationState &odsState,
                            ::mlir::Value stream,
                            ::mlir::gpu::GPUModuleOp module) {
  auto ctx = odsBuilder.getContext();
  LoadGpuModuleOp::build(odsBuilder, odsState, plier::OpaqueType::get(ctx),
                         stream,
                         mlir::SymbolRefAttr::get(ctx, module.getName()));
}

void LoadGpuModuleOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<RemoveUnusedOp<LoadGpuModuleOp, DestroyGpuModuleOp>>(context);
}

void GetGpuKernelOp::build(::mlir::OpBuilder &odsBuilder,
                           ::mlir::OperationState &odsState,
                           ::mlir::Value module,
                           ::mlir::gpu::GPUFuncOp kernel) {
  auto ctx = odsBuilder.getContext();
  GetGpuKernelOp::build(odsBuilder, odsState, plier::OpaqueType::get(ctx),
                        module,
                        mlir::SymbolRefAttr::get(ctx, kernel.getName()));
}

void GetGpuKernelOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<RemoveUnusedOp<GetGpuKernelOp, DestroyGpuKernelOp>>(context);
}

void LaunchGpuKernelOp::build(::mlir::OpBuilder &builder,
                              ::mlir::OperationState &result,
                              ::mlir::Value stream, ::mlir::Value kernel,
                              ::mlir::gpu::KernelDim3 gridSize,
                              ::mlir::gpu::KernelDim3 blockSize,
                              ::mlir::ValueRange kernelOperands) {
  result.addOperands(stream);
  result.addOperands(kernel);
  result.addOperands({gridSize.x, gridSize.y, gridSize.z, blockSize.x,
                      blockSize.y, blockSize.z});
  result.addOperands(kernelOperands);
  llvm::SmallVector<int32_t> segmentSizes(10, 1);
  segmentSizes.front() = 0; // Initially no async dependencies.
  segmentSizes.back() = static_cast<int32_t>(kernelOperands.size());
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr(segmentSizes));
}

void GPUSuggestBlockSizeOp::build(::mlir::OpBuilder &odsBuilder,
                                  ::mlir::OperationState &odsState,
                                  ::llvm::Optional<::mlir::Value> stream,
                                  ::mlir::OpFoldResult kernel,
                                  ::mlir::ValueRange gridSize) {
  auto dimCount = gridSize.size();
  assert(dimCount > 0 && dimCount <= 3);
  llvm::SmallVector<mlir::Type, 3> resTypes(dimCount,
                                            odsBuilder.getIndexType());
  mlir::Value kernVal;
  mlir::SymbolRefAttr kernRef;
  if (kernel.is<mlir::Value>())
    kernVal = kernel.get<mlir::Value>();
  else
    kernRef = kernel.get<mlir::Attribute>().cast<mlir::SymbolRefAttr>();

  GPUSuggestBlockSizeOp::build(odsBuilder, odsState, resTypes,
                               stream.getValueOr(mlir::Value{}), kernVal,
                               kernRef, gridSize);
}

mlir::StringAttr GPUSuggestBlockSizeOp::getKernelModuleName() {
  assert(kernelRef());
  return kernelRef()->getRootReference();
}

mlir::StringAttr GPUSuggestBlockSizeOp::getKernelName() {
  assert(kernelRef());
  return kernelRef()->getLeafReference();
}

void ForceCopyOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                        mlir::Value source) {
  build(b, result, source.getType(), source);
}

void TakeContextOp::build(mlir::OpBuilder &b, mlir::OperationState &result,
                          mlir::SymbolRefAttr initFunc,
                          mlir::SymbolRefAttr releaseFunc,
                          mlir::TypeRange resultTypes) {
  llvm::SmallVector<mlir::Type> allTypes;
  allTypes.emplace_back(plier::OpaqueType::get(b.getContext()));
  allTypes.append(resultTypes.begin(), resultTypes.end());
  build(b, result, allTypes, initFunc, releaseFunc);
}

} // namespace plier

#include "mlir-extensions/dialect/plier_util/PlierUtilOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mlir-extensions/dialect/plier_util/PlierUtilOps.cpp.inc"

//#include "mlir-extensions/dialect/plier_util/PlierUtilOpsEnums.cpp.inc"
