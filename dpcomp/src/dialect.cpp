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

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>

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
      .Case<plier::SliceType>([&](auto) { os << "SliceType"; })
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

SliceType SliceType::get(mlir::MLIRContext *context) {
  assert(context);
  return Base::get(context);
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

mlir::OpFoldResult CastOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  auto arg = value();
  auto opType = arg.getType();
  auto retType = getType();
  if (opType == retType && opType != PyType::getUndefined(getContext()))
    return arg;

  if (auto res = propagateCasts(arg, retType))
    return res;

  return nullptr;
}

namespace {
struct PropagateCasts
    : public mlir::OpRewritePattern<mlir::UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::UnrealizedConversionCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto inputs = op.inputs();
    if (inputs.size() != 1 || op->getNumResults() != 1)
      return mlir::failure();

    auto thisType = op.getType(0);
    auto arg = inputs[0];
    auto res = propagateCasts(arg, thisType);
    if (!res)
      return mlir::failure();

    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};
} // namespace

void CastOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                         ::mlir::MLIRContext *context) {
  results.insert<PropagateCasts>(context);
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

static mlir::Value
foldBuildTupleGetitem(mlir::Value val, mlir::Type type,
                      llvm::ArrayRef<mlir::Attribute> operands) {
  auto getCastArg = [](mlir::Value arg) -> mlir::Value {
    if (auto cast = arg.getDefiningOp<plier::CastOp>())
      return cast.value();

    if (auto cast = arg.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      auto inputs = cast.inputs();
      if (inputs.size() == 1)
        return inputs.front();
    }

    return {};
  };

  while (auto arg = getCastArg(val))
    val = arg;

  auto buildTuple = val.getDefiningOp<plier::BuildTupleOp>();
  if (buildTuple) {
    if (auto val = operands[1].dyn_cast_or_null<mlir::IntegerAttr>()) {
      auto index = val.getInt();
      auto numArgs = static_cast<unsigned>(buildTuple.args().size());
      if (index >= 0 && index < numArgs) {
        auto op = buildTuple.args()[static_cast<unsigned>(index)];
        if (op.getType() == type)
          return op;
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
  auto type = SliceType::get(builder.getContext());
  BuildSliceOp::build(builder, state, type, begin, end, stride);
}

namespace {
struct SliceGetitemPropagate
    : public mlir::OpRewritePattern<plier::SliceGetItemOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::SliceGetItemOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.array().getType().isa<mlir::TensorType>())
      return mlir::failure();

    auto index = mlir::getConstantIntValue(op.index());
    if (!index)
      return mlir::failure();

    auto i = *index;
    if (i < 0 || i >= 3)
      return mlir::failure();

    auto buildSlice = op.slice().getDefiningOp<plier::BuildSliceOp>();
    if (!buildSlice)
      return mlir::failure();

    auto loc = op.getLoc();
    auto getInd = [&](int64_t val) -> mlir::Value {
      return rewriter.create<mlir::arith::ConstantIndexOp>(loc, val);
    };

    auto src = buildSlice.getOperand(static_cast<unsigned>(i));
    auto srcType = src.getType();
    if (srcType.isa<mlir::NoneType>()) {
      if (i == 0) {
        rewriter.replaceOp(op, getInd(0));
      } else if (i == 1) {
        auto size =
            rewriter.create<mlir::tensor::DimOp>(loc, op.array(), op.dim());
        rewriter.replaceOp(op, size.getResult());
      } else { // i == 2
        rewriter.replaceOp(op, getInd(1));
      }
    } else {
      if (auto intType = srcType.dyn_cast<mlir::IntegerType>()) {
        if (!intType.isSignless()) {
          auto signless =
              mlir::IntegerType::get(intType.getContext(), intType.getWidth());
          src = rewriter.create<plier::SignCastOp>(loc, signless, src);
        }
        auto indexType = rewriter.getIndexType();
        src = rewriter.create<mlir::arith::IndexCastOp>(loc, src, indexType);
      } else if (srcType.isa<mlir::IndexType>()) {
        // Nothing
      } else {
        return mlir::failure();
      }
      rewriter.replaceOp(op, src);
    }

    return mlir::success();
  }
};
} // namespace

void SliceGetItemOp::getCanonicalizationPatterns(
    ::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context) {
  results.insert<SliceGetitemPropagate>(context);
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
  for (auto i : llvm::seq(0u, rank)) {
    if (!isStrideCompatible(srcStrides[i], dstStrides[i]))
      return false;
  }
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

    rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(op, src, dstType);
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
    : public mlir::OpRewritePattern<mlir::memref::CloneOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::CloneOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.input().getDefiningOp<plier::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::CloneOp>(op, cl.source());
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
      rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(op, src, dstType);
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

    auto loc = op.getLoc();
    auto newSubview = rewriter.createOrFold<mlir::memref::SubViewOp>(
        loc, cl.source(), op.getMixedOffsets(), op.getMixedSizes(),
        op.getMixedStrides());
    auto oldType = op.getType();
    auto newType = newSubview.getType();
    if (newType != oldType)
      newSubview = rewriter.createOrFold<plier::ChangeLayoutOp>(loc, oldType,
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
          if (!isInputs) {
            for (auto i : llvm::seq(0u, count)) {
              auto newType = newOperands[i].getType();
              auto res = op.getResult(i);
              if (newType != res.getType()) {
                assert(newType.isa<mlir::MemRefType>());
                assert(res.use_empty());
                res.setType(newType);
              }
            }
          }
        });
      }
    }

    return mlir::success(changed);
  }
};

struct ChangeLayoutLinalgCopy
    : public mlir::OpRewritePattern<mlir::linalg::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::CopyOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto input = op.input();
    auto output = op.output();
    auto clInput = input.getDefiningOp<plier::ChangeLayoutOp>();
    auto clOutput = output.getDefiningOp<plier::ChangeLayoutOp>();
    if (!clInput && !clOutput)
      return mlir::failure();

    if (clInput)
      input = clInput.source();

    if (clOutput)
      output = clOutput.source();

    rewriter.replaceOpWithNewOp<mlir::linalg::CopyOp>(
        op, input, output, op.inputPermutation().getValueOr(mlir::AffineMap{}),
        op.outputPermutation().getValueOr(mlir::AffineMap{}));
    return mlir::success();
  }
};

struct ChangeLayoutIf : public mlir::OpRewritePattern<mlir::scf::YieldOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.results().empty())
      return mlir::failure();

    auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op->getParentOp());
    if (!ifOp)
      return mlir::failure();

    auto trueYield = mlir::cast<mlir::scf::YieldOp>(
        ifOp.thenRegion().front().getTerminator());
    auto falseYield = mlir::cast<mlir::scf::YieldOp>(
        ifOp.elseRegion().front().getTerminator());
    mlir::OpBuilder::InsertionGuard g(rewriter);
    auto count = static_cast<unsigned>(trueYield.results().size());
    llvm::SmallVector<mlir::Type> newResultTypes(count);
    bool changed = false;
    for (auto i : llvm::seq(0u, count)) {
      auto origType = ifOp.getResult(i).getType();

      mlir::Type newType;
      for (bool reverse : {true, false}) {
        auto clYield = (reverse ? falseYield : trueYield);
        auto otherYield = (reverse ? trueYield : falseYield);

        auto arg = clYield.results()[i];
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

        auto otherArg = otherYield.results()[i];
        rewriter.setInsertionPoint(otherYield);
        auto otherRes = rewriter.createOrFold<mlir::memref::CastOp>(
            otherYield.getLoc(), otherArg, srcType);

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

} // namespace

void ChangeLayoutOp::getCanonicalizationPatterns(
    ::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context) {
  results.insert<ChangeLayoutIdentity, ChangeLayoutDim,
                 ChangeLayoutExtractMetadata, ChangeLayoutClone,
                 ChangeLayoutCast, ChangeLayoutLoad, ChangeLayoutStore,
                 ChangeLayoutSubview, ChangeLayoutLinalgGeneric,
                 ChangeLayoutLinalgCopy, ChangeLayoutIf, ChangeLayout1DReshape>(
      context);
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
    auto affineMap = [&]() {
      mlir::AffineMap ret;
      auto affineMap = memrefType.getLayout().getAffineMap();
      if (affineMap && !affineMap.isIdentity()) {
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
