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

#include "plier/dialect.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/InliningUtils.h>

#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <llvm/ADT/TypeSwitch.h>

#include "plier/transforms/const_utils.hpp"

namespace MemoryEffects = ::mlir::MemoryEffects;

namespace {
struct PlierInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                       mlir::BlockAndValueMapping &) const final override {
    return true;
  }
  bool isLegalToInline(mlir::Operation *op, mlir::Region *, bool,
                       mlir::BlockAndValueMapping &) const final override {
    return !mlir::isa<plier::ArgOp>(op);
  }
};
} // namespace

namespace plier {

llvm::StringRef attributes::getFastmathName() { return "#plier.fastmath"; }

llvm::StringRef attributes::getJumpMarkersName() {
  return "#plier.pipeline_jump_markers";
}

llvm::StringRef attributes::getParallelName() { return "#plier.parallel"; }

llvm::StringRef attributes::getMaxConcurrencyName() {
  return "#plier.max_concurrency";
}

llvm::StringRef attributes::getForceInlineName() {
  return "#plier.force_inline";
}

llvm::StringRef attributes::getOptLevelName() { return "#plier.opt_level"; }

llvm::StringRef attributes::getGpuRangeName() { return "#plier.gpu_range"; }

namespace detail {
struct PyTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::StringRef;

  PyTypeStorage(mlir::StringRef name) : name(name) {}

  bool operator==(const KeyTy &key) const { return key == name; }

  static PyTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                  const KeyTy &key) {
    return new (allocator.allocate<PyTypeStorage>())
        PyTypeStorage(allocator.copyInto(key));
  }

  mlir::StringRef name;
};

struct LiteralTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Attribute;

  LiteralTypeStorage(mlir::Attribute val) : value(val) {}

  bool operator==(const KeyTy &key) const { return key == value; }

  static LiteralTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<LiteralTypeStorage>())
        LiteralTypeStorage(key);
  }

  mlir::Attribute value;
};

struct SliceTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::tuple<mlir::Type, mlir::Type, mlir::Type>;

  SliceTypeStorage(const KeyTy &t) : types(t) {}

  bool operator==(const KeyTy &key) const { return key == types; }

  static SliceTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<TypeVarStorage>()) SliceTypeStorage(key);
  }

  KeyTy types;
};

struct TypeVarStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  TypeVarStorage(mlir::Type type) : type(type) {}

  bool operator==(const KeyTy &key) const { return key == type; }

  static TypeVarStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    return new (allocator.allocate<TypeVarStorage>()) TypeVarStorage(key);
  }

  mlir::Type type;
};
} // namespace detail

mlir::ArrayRef<detail::OperatorNamePair> getOperators() {
  return llvm::makeArrayRef(detail::OperatorNames);
}

void PlierDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "plier/PlierOps.cpp.inc"
      >();
  addTypes<plier::PyType, plier::LiteralType, SliceType, plier::TypeVar,
           OpaqueType>();
  addInterfaces<PlierInlinerInterface>();
}

mlir::Type PlierDialect::parseType(mlir::DialectAsmParser &parser) const {
  parser.emitError(parser.getNameLoc(), "unknown type");
  return mlir::Type();
}

void PlierDialect::printType(mlir::Type type,
                             mlir::DialectAsmPrinter &os) const {
  llvm::TypeSwitch<mlir::Type>(type)
      .Case<plier::PyType>(
          [&](auto t) { os << "PyType<" << t.getName() << ">"; })
      .Case<plier::LiteralType>([&](auto t) {
        os << "LiteralType<";
        os.printAttribute(t.getValue());
        os << ">";
      })
      .Case<plier::SliceType>([&](auto t) {
        os << "SliceType<";
        os.printType(t.getBegin());
        os << ", ";
        os.printType(t.getEnd());
        os << ", ";
        os.printType(t.getStride());
        os << ">";
      })
      .Case<plier::TypeVar>([&](auto t) {
        os << "TypeVar<";
        os.printType(t.getType());
        os << ">";
      })
      .Case<plier::OpaqueType>([&](auto) { os << "OpaqueType"; })
      .Default([](auto) { llvm_unreachable("unexpected type"); });
}

PyType PyType::get(mlir::MLIRContext *context, llvm::StringRef name) {
  assert(!name.empty());
  return Base::get(context, name);
}

PyType PyType::getUndefined(mlir::MLIRContext *context) {
  return Base::get(context, "");
}

llvm::StringRef PyType::getName() const { return getImpl()->name; }

LiteralType LiteralType::get(mlir::Attribute value) {
  assert(value);
  return Base::get(value.getContext(), value);
}

mlir::Attribute LiteralType::getValue() const { return getImpl()->value; }

SliceType SliceType::get(mlir::Type begin, mlir::Type end, mlir::Type stride) {
  assert(begin);
  assert(end);
  assert(stride);
  return Base::get(begin.getContext(), std::make_tuple(begin, end, stride));
}

mlir::Type SliceType::getBegin() const { return std::get<0>(getImpl()->types); }

mlir::Type SliceType::getEnd() const { return std::get<1>(getImpl()->types); }

mlir::Type SliceType::getStride() const {
  return std::get<2>(getImpl()->types);
}

std::array<mlir::Type, 3> SliceType::getTypes() const {
  return {getBegin(), getEnd(), getStride()};
}

TypeVar TypeVar::get(mlir::Type type) {
  assert(type);
  return Base::get(type.getContext(), type);
}

OpaqueType OpaqueType::get(mlir::MLIRContext *context) {
  assert(context);
  return Base::get(context);
}

mlir::Type TypeVar::getType() const { return getImpl()->type; }

void ArgOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  unsigned index, mlir::StringRef name) {
  ArgOp::build(builder, state, PyType::getUndefined(state.getContext()), index,
               name);
}

mlir::OpFoldResult ArgOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  auto func = getOperation()->getParentOfType<mlir::FuncOp>();
  if (func) {
    auto ind = index();
    if (ind < func.getNumArguments() &&
        func.getArgument(ind).getType() == getType()) {
      return func.getArgument(ind);
    }
  }
  return nullptr;
}

void ConstOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,

                    mlir::Attribute val) {
  ConstOp::build(builder, state, PyType::getUndefined(state.getContext()), val);
}

void GlobalOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     mlir::StringRef name) {
  GlobalOp::build(builder, state, PyType::getUndefined(state.getContext()),
                  name);
}

void BinOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs, mlir::StringRef op) {
  BinOp::build(builder, state, PyType::getUndefined(state.getContext()), lhs,
               rhs, op);
}

void UnaryOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Value value, mlir::StringRef op) {
  UnaryOp::build(builder, state, PyType::getUndefined(state.getContext()),
                 value, op);
}

mlir::OpFoldResult CastOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  auto opType = getOperand().getType();
  auto retType = getType();
  if (opType == retType && opType != PyType::getUndefined(getContext()))
    return getOperand();

  if (auto prevCast = getOperand().getDefiningOp<plier::CastOp>()) {
    auto prevValue = prevCast.value();
    if (prevValue.getType() == retType)
      return prevValue;
  }

  return nullptr;
}

void PyCallOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value func,
    llvm::StringRef func_name, mlir::ValueRange args, mlir::Value varargs,
    mlir::ArrayRef<std::pair<std::string, mlir::Value>> kwargs) {
  auto ctx = builder.getContext();

  llvm::SmallVector<mlir::Value> kwArgsVals(kwargs.size());
  llvm::copy(llvm::make_second_range(kwargs), kwArgsVals.begin());

  llvm::SmallVector<mlir::Attribute> kwNames;
  kwNames.reserve(kwargs.size());
  for (auto &a : kwargs)
    kwNames.push_back(mlir::StringAttr::get(ctx, a.first));

  PyCallOp::build(builder, state, PyType::getUndefined(state.getContext()),
                  func, args, varargs, kwArgsVals, func_name,
                  mlir::ArrayAttr::get(ctx, kwNames));
}

void BuildTupleOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         ::mlir::ValueRange args) {
  BuildTupleOp::build(builder, state, PyType::getUndefined(state.getContext()),
                      args);
}

// mlir::LogicalResult BuildTupleOp::fold(
//    llvm::ArrayRef<mlir::Attribute> /*operands*/,
//    llvm::SmallVectorImpl<mlir::OpFoldResult> &results)
//{
//    auto res_types = getResultTypes();
//    auto args = getOperands();
//    if (res_types.size() == args.size())
//    {
//        std::copy(args.begin(), args.end(), std::back_inserter(results));
//        return mlir::success();
//    }
//    return mlir::failure();
//}

mlir::Value foldBuildTupleGetitem(mlir::Value val, mlir::Type type,
                                  llvm::ArrayRef<mlir::Attribute> operands) {
  auto buildTuple = val.getDefiningOp<plier::BuildTupleOp>();
  if (buildTuple) {
    if (auto val = operands[1].dyn_cast_or_null<mlir::IntegerAttr>()) {
      auto index = val.getInt();
      if (index >= 0 && index < buildTuple.getNumOperands()) {
        auto op = buildTuple.getOperand(static_cast<unsigned>(index));
        if (op.getType() == type) {
          return op;
        }
      }
    }
  }
  return {};
}

void GetItemOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      ::mlir::Value value, ::mlir::Value index) {
  GetItemOp::build(builder, state, PyType::getUndefined(state.getContext()),
                   value, index);
}

mlir::OpFoldResult GetItemOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  if (auto val = foldBuildTupleGetitem(value(), getType(), operands))
    return val;

  return nullptr;
}

void GetiterOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      ::mlir::Value value) {
  GetiterOp::build(builder, state, PyType::getUndefined(state.getContext()),
                   value);
}

void IternextOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       ::mlir::Value value) {
  IternextOp::build(builder, state, PyType::getUndefined(state.getContext()),
                    value);
}

void PairfirstOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        ::mlir::Value value) {
  PairfirstOp::build(builder, state, PyType::getUndefined(state.getContext()),
                     value);
}

void PairsecondOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         ::mlir::Value value) {
  PairsecondOp::build(builder, state, PyType::getUndefined(state.getContext()),
                      value);
}

void GetattrOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      mlir::Value value, mlir::StringRef name) {
  GetattrOp::build(builder, state, PyType::getUndefined(state.getContext()),
                   value, name);
}

void ExhaustIterOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          mlir::Value value, int64_t count) {
  ExhaustIterOp::build(builder, state, PyType::getUndefined(state.getContext()),
                       value, builder.getI64IntegerAttr(count));
}

mlir::OpFoldResult
ExhaustIterOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  if (getType() == getOperand().getType() &&
      getType() != plier::PyType::getUndefined(getContext())) {
    return getOperand();
  }
  return nullptr;
}

namespace {
struct GetattrGlobalRewrite : public mlir::OpRewritePattern<GetattrOp> {
  using mlir::OpRewritePattern<GetattrOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(GetattrOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto prev_op = mlir::dyn_cast_or_null<plier::GlobalOp>(
        op.getOperand().getDefiningOp());
    if (prev_op) {
      auto new_name = llvm::Twine(prev_op.name() + "." + op.name()).str();
      auto new_op =
          rewriter.create<plier::GlobalOp>(op.getLoc(), op.getType(), new_name);
      rewriter.replaceOp(op, new_op.getResult());
      return mlir::success();
    }
    return mlir::failure();
  }
};
} // namespace

void GetattrOp::getCanonicalizationPatterns(
    ::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context) {
  results.insert<GetattrGlobalRewrite>(context);
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
    ::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context) {
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
  auto *bodyBlock = odsBuilder.createBlock(bodyRegion, {}, argTypes);

  if (bodyBuilder) {
    odsBuilder.setInsertionPointToStart(bodyBlock);
    auto args = bodyBlock->getArguments();
    bodyBuilder(odsBuilder, odsState.location, args.take_front(count),
                args.drop_front(count).take_front(count), args.back());
    ParallelOp::ensureTerminator(*bodyRegion, odsBuilder, odsState.location);
  }
}

void BuildSliceOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::Value begin, mlir::Value end,
                         mlir::Value stride) {
  auto type = SliceType::get(begin.getType(), end.getType(), stride.getType());
  BuildSliceOp::build(builder, state, type, begin, end, stride);
}

void RetainOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     mlir::Value value) {
  RetainOp::build(builder, state, value.getType(), value);
}

mlir::OpFoldResult SignCastOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  assert(operands.size() == 1);
  auto thisType = getType();
  auto attrOperand = operands.front();
  if (attrOperand && attrOperand.getType() == thisType) {
    return attrOperand;
  }

  auto arg = getOperand();
  if (arg.getType() == thisType) {
    return arg;
  }
  if (auto prevOp = arg.getDefiningOp<SignCastOp>()) {
    auto prevArg = prevOp.getOperand();
    if (prevArg.getType() == thisType) {
      return prevArg;
    }
  }
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
  using mlir::OpRewritePattern<plier::SignCastOp>::OpRewritePattern;

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

    auto newSrcType =
        mlir::RankedTensorType::get(srcType.getShape(), finalElemType);
    auto newDstType =
        mlir::RankedTensorType::get(dstType.getShape(), finalElemType);

    auto loc = op.getLoc();
    auto casted = rewriter.createOrFold<plier::SignCastOp>(loc, newSrcType,
                                                           tensorCast.source());
    rewriter.replaceOpWithNewOp<mlir::tensor::CastOp>(op, newDstType, casted);

    return mlir::success();
  }
};

} // namespace

void SignCastOp::getCanonicalizationPatterns(
    ::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context) {
  results.insert<SignCastDimPropagate<mlir::tensor::DimOp>,
                 SignCastDimPropagate<mlir::memref::DimOp>,
                 SignCastUndefPropagate, SignCastTensorCastPropagate>(context);
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
  for (auto it : llvm::enumerate(mapping)) {
    shape[it.index()] = srcShape[static_cast<size_t>(it.value())];
  }

  if (auto tensorType = srcType.dyn_cast<mlir::RankedTensorType>()) {
    auto retType = mlir::RankedTensorType::get(
        shape, tensorType.getElementType(), tensorType.getEncoding());
    build(odsBuilder, odsState, retType, src, mapAttr);
  } else if (auto memrefType = srcType.dyn_cast<mlir::MemRefType>()) {
    auto affineMaps = [&]() {
      llvm::SmallVector<mlir::AffineMap, 1> ret;
      auto affineMaps = memrefType.getAffineMaps();
      assert(affineMaps.size() < 2);
      if (!affineMaps.empty() && !affineMaps[0].isIdentity()) {
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
                mlir::getAffineSymbolExpr(srcIndex, context);
          } else {
            dimReplacements[i] = mlir::getAffineConstantExpr(0, context);
            symReplacements[i + 1] = mlir::getAffineConstantExpr(0, context);
          }
        }
        auto srcMap = memrefType.getAffineMaps().front();
        auto dstRank = static_cast<unsigned>(mapping.size());
        auto resMap = srcMap.replaceDimsAndSymbols(
            dimReplacements, symReplacements, dstRank, dstRank + 1);
        ret.emplace_back(mlir::simplifyAffineMap(resMap));
      }
      return ret;
    }();

    auto retType =
        mlir::MemRefType::get(shape, memrefType.getElementType(), affineMaps,
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
  auto zero = builder.createOrFold<mlir::ConstantIndexOp>(loc, 0);
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
    ::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context) {
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

} // namespace plier

#include "plier/PlierOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "plier/PlierOps.cpp.inc"

//#include "plier/PlierOpsEnums.cpp.inc"
