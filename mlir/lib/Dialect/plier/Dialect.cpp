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

#include "imex/Dialect/plier/Dialect.hpp"
#include "imex/Dialect/imex_util/Dialect.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/InliningUtils.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/FunctionInterfaces.h>

#include <llvm/ADT/TypeSwitch.h>

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

mlir::ArrayRef<detail::OperatorNamePair> getOperators() {
  return llvm::makeArrayRef(detail::OperatorNames);
}

void PlierDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "imex/Dialect/plier/PlierOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "imex/Dialect/plier/PlierOpsTypes.cpp.inc"
      >();

  addInterfaces<PlierInlinerInterface>();
}

mlir::Operation *PlierDialect::materializeConstant(mlir::OpBuilder &builder,
                                                   mlir::Attribute value,
                                                   mlir::Type type,
                                                   mlir::Location loc) {
  if (mlir::arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<mlir::arith::ConstantOp>(loc, type, value);

  return nullptr;
}

void ArgOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  unsigned index, mlir::StringRef name) {
  ArgOp::build(builder, state, PyType::getUndefined(state.getContext()), index,
               name);
}

mlir::OpFoldResult ArgOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  auto func = getOperation()->getParentOfType<mlir::FunctionOpInterface>();
  if (func) {
    auto ind = getIndex();
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
      &foldPrevCast<imex::util::SignCastOp>,
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
  auto arg = getValue();
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

void GetItemOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      ::mlir::Value value, ::mlir::Value index) {
  GetItemOp::build(builder, state, PyType::getUndefined(state.getContext()),
                   value, index);
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
    auto prevOp = mlir::dyn_cast_or_null<plier::GlobalOp>(
        op.getOperand().getDefiningOp());
    if (prevOp) {
      auto newName = llvm::Twine(prevOp.getName() + "." + op.getName()).str();
      auto newOp =
          rewriter.create<plier::GlobalOp>(op.getLoc(), op.getType(), newName);
      rewriter.replaceOp(op, newOp.getResult());
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
} // namespace plier

#include "imex/Dialect/plier/PlierOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "imex/Dialect/plier/PlierOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "imex/Dialect/plier/PlierOpsTypes.cpp.inc"

//#include "imex/Dialect/plier/PlierOpsEnums.cpp.inc"
