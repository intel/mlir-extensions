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

#include "mlir-extensions/Dialect/plier/dialect.hpp"
#include "mlir-extensions/Dialect/plier_util/dialect.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/InliningUtils.h>

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>

#include <llvm/ADT/TypeSwitch.h>

#include "mlir-extensions/Transforms/const_utils.hpp"

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
#include "mlir-extensions/Dialect/plier/PlierOps.cpp.inc"
      >();
  addTypes<plier::PyType, plier::LiteralType, SliceType, plier::TypeVar>();
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
      .Default([](auto) { llvm_unreachable("unexpected type"); });
}

mlir::Operation *PlierDialect::materializeConstant(mlir::OpBuilder &builder,
                                                   mlir::Attribute value,
                                                   mlir::Type type,
                                                   mlir::Location loc) {
  if (mlir::arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<mlir::arith::ConstantOp>(loc, type, value);

  return nullptr;
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

mlir::Type TypeVar::getType() const { return getImpl()->type; }

void ArgOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  unsigned index, mlir::StringRef name) {
  ArgOp::build(builder, state, PyType::getUndefined(state.getContext()), index,
               name);
}

mlir::OpFoldResult ArgOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  auto func = getOperation()->getParentOfType<mlir::func::FuncOp>();
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
    auto inputs = op.getInputs();
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
  PyCallOp::build(builder, state,
                  plier::PyType::getUndefined(builder.getContext()), func,
                  func_name, args, varargs, kwargs);
}

void PyCallOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type type,
    mlir::Value func, llvm::StringRef func_name, mlir::ValueRange args,
    mlir::Value varargs,
    mlir::ArrayRef<std::pair<std::string, mlir::Value>> kwargs) {
  auto ctx = builder.getContext();

  llvm::SmallVector<mlir::Value> kwArgsVals(kwargs.size());
  llvm::copy(llvm::make_second_range(kwargs), kwArgsVals.begin());

  llvm::SmallVector<mlir::Attribute> kwNames;
  kwNames.reserve(kwargs.size());
  for (auto &a : kwargs)
    kwNames.push_back(mlir::StringAttr::get(ctx, a.first));

  PyCallOp::build(builder, state, type, func, args, varargs, kwArgsVals,
                  func_name, mlir::ArrayAttr::get(ctx, kwNames));
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
      auto inputs = cast.getInputs();
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

void GetattrOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                            ::mlir::MLIRContext *context) {
  results.insert<GetattrGlobalRewrite>(context);
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
    if (!op.array().getType().isa<mlir::ShapedType>())
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
        auto size = [&]() -> mlir::Value {
          if (op.array().getType().isa<mlir::TensorType>())
            return rewriter.create<mlir::tensor::DimOp>(loc, op.array(),
                                                        op.dim());
          return rewriter.create<mlir::memref::DimOp>(loc, op.array(),
                                                      op.dim());
        }();
        rewriter.replaceOp(op, size);
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
        src = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, src);
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
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<SliceGetitemPropagate>(context);
}
} // namespace plier

#include "mlir-extensions/Dialect/plier/PlierOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mlir-extensions/Dialect/plier/PlierOps.cpp.inc"

//#include "mlir-extensions/Dialect/plier/PlierOpsEnums.cpp.inc"
