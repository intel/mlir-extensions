// Copyright 2022 Intel Corporation
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

#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/InliningUtils.h>

#include <llvm/ADT/TypeSwitch.h>

namespace {
struct NTensorInlinerInterface : public mlir::DialectInlinerInterface {
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

void imex::ntensor::NTensorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "imex/Dialect/ntensor/IR/NTensorOps.cpp.inc"
      >();

  addInterfaces<NTensorInlinerInterface>();

  addTypes<
#define GET_TYPEDEF_LIST
#include "imex/Dialect/ntensor/IR/NTensorOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "imex/Dialect/ntensor/IR/NTensorOpsAttributes.cpp.inc"
      >();
}

mlir::Operation *imex::ntensor::NTensorDialect::materializeConstant(
    mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type type,
    mlir::Location loc) {
  if (mlir::arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<mlir::arith::ConstantOp>(loc, type, value);

  if (type.isa<mlir::IndexType>())
    if (auto val = mlir::getConstantIntValue(value))
      return builder.create<mlir::arith::ConstantIndexOp>(loc, *val);

  return nullptr;
}

void imex::ntensor::NTensorType::walkImmediateSubElements(
    llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
    llvm::function_ref<void(mlir::Type)> walkTypesFn) const {
  walkTypesFn(getElementType());
  if (mlir::Attribute env = getEnvironment())
    walkAttrsFn(env);
}

bool imex::ntensor::NTensorBase::hasRank() const { return true; }

llvm::ArrayRef<int64_t> imex::ntensor::NTensorBase::getShape() const {
  return cast<NTensorType>().getShape();
}

imex::ntensor::NTensorBase imex::ntensor::NTensorBase::cloneWith(
    llvm::Optional<llvm::ArrayRef<int64_t>> shape, Type elementType) const {
  auto t = cast<NTensorType>();
  return NTensorType::get(shape.value_or(getShape()), elementType,
                          t.getEnvironment());
}

mlir::Type imex::ntensor::NTensorType::replaceImmediateSubElements(
    llvm::ArrayRef<mlir::Attribute> replAttrs,
    llvm::ArrayRef<mlir::Type> replTypes) const {
  return get(getShape(), replTypes.front(),
             replAttrs.empty() ? mlir::Attribute() : replAttrs.back());
}

namespace {
struct ResolveSlicePropagate
    : public mlir::OpRewritePattern<imex::ntensor::ResolveSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::ResolveSliceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto buildSlice =
        op.getSlice().getDefiningOp<imex::ntensor::BuildSliceOp>();
    if (!buildSlice)
      return mlir::failure();

    auto loc = op->getLoc();

    mlir::Value zero;
    auto getZero = [&]() {
      if (!zero)
        zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);

      return zero;
    };

    auto size = op.getSize();
    auto handleNegativeVal = [&](mlir::Value val) -> mlir::Value {
      auto isNeg = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::slt, val, getZero());
      auto negIndex = rewriter.create<mlir::arith::AddIOp>(loc, size, val);
      auto posIndex =
          rewriter.create<mlir::arith::SelectOp>(loc, isNeg, negIndex, val);
      auto isOutOfRange = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sgt, val, size);
      return rewriter.create<mlir::arith::SelectOp>(loc, isOutOfRange, size,
                                                    posIndex);
    };

    mlir::Value results[3];
    if (auto begin = buildSlice.getBegin()) {
      results[0] = handleNegativeVal(begin);
    } else {
      results[0] = getZero();
    }

    if (auto end = buildSlice.getEnd()) {
      results[1] = handleNegativeVal(end);
    } else {
      results[1] = size;
    }

    if (auto step = buildSlice.getStep()) {
      results[2] = step;
    } else {
      results[2] = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    }

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};
} // namespace

void imex::ntensor::ResolveSliceOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<ResolveSlicePropagate>(context);
}

static mlir::LogicalResult
parseShape(mlir::AsmParser &parser,
           mlir::FailureOr<llvm::SmallVector<int64_t>> &shape,
           mlir::FailureOr<mlir::Type> &type) {
  llvm::SmallVector<int64_t> dimensions;
  if (parser.parseDimensionList(dimensions))
    return mlir::failure();

  mlir::Type t;
  if (parser.parseType(t))
    return mlir::failure();

  shape = std::move(dimensions);
  type = std::move(t);
  return mlir::success();
}

static void printShape(mlir::AsmPrinter &printer, llvm::ArrayRef<int64_t> shape,
                       mlir::Type type) {
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim))
      printer << '?';
    else
      printer << dim;
    printer << 'x';
  }
  printer << type;
}

static bool parseArgList(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &argsOperands,
    mlir::ArrayAttr &args_namesAttr) {
  if (parser.parseLParen())
    return true;

  auto *context = parser.getContext();
  llvm::SmallVector<mlir::Attribute> names;
  if (parser.parseOptionalRParen()) {
    std::string name;
    while (true) {
      name.clear();
      if (!parser.parseOptionalKeywordOrString(&name)) {
        if (parser.parseColon())
          return true;
      }
      names.push_back(mlir::StringAttr::get(context, name));

      argsOperands.push_back({});
      if (parser.parseOperand(argsOperands.back()))
        return true;

      if (!parser.parseOptionalRParen())
        break;

      if (parser.parseComma())
        return true;
    }
  }

  assert(names.size() == argsOperands.size());
  args_namesAttr = mlir::ArrayAttr::get(context, names);
  return false;
}

static void printArgList(mlir::OpAsmPrinter &printer,
                         imex::ntensor::CallOp call, mlir::ValueRange args,
                         mlir::ArrayAttr args_names) {
  assert(args.size() == args_names.size());
  printer << '(';
  bool first = true;
  for (auto it : llvm::zip(args, args_names)) {
    if (first) {
      first = false;
    } else {
      printer << ", ";
    }
    auto arg = std::get<0>(it);
    auto name = std::get<1>(it);
    auto nameStr =
        (name ? name.cast<mlir::StringAttr>().getValue() : llvm::StringRef());
    if (!nameStr.empty())
      printer << nameStr << ':';
    printer.printOperand(arg);
  }
  printer << ')';
}

#include "imex/Dialect/ntensor/IR/NTensorOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "imex/Dialect/ntensor/IR/NTensorOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "imex/Dialect/ntensor/IR/NTensorOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "imex/Dialect/ntensor/IR/NTensorOpsTypes.cpp.inc"

#include "imex/Dialect/ntensor/IR/NTensorOpsEnums.cpp.inc"
