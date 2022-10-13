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

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
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

bool imex::ntensor::NTensorBase::isValidElementType(Type type) {
  return type.isIntOrIndexOrFloat();
}

mlir::Type imex::ntensor::NTensorType::replaceImmediateSubElements(
    llvm::ArrayRef<mlir::Attribute> replAttrs,
    llvm::ArrayRef<mlir::Type> replTypes) const {
  return get(getShape(), replTypes.front(),
             replAttrs.empty() ? mlir::Attribute() : replAttrs.back());
}

static mlir::Value handleSliceIndexVars(mlir::OpBuilder &builder,
                                        mlir::Location loc, mlir::Value source,
                                        mlir::Value size) {
  auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto isNeg = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, source, zero);
  auto negIndex = builder.create<mlir::arith::AddIOp>(loc, size, source);
  auto posIndex =
      builder.create<mlir::arith::SelectOp>(loc, isNeg, negIndex, source);
  auto isOutOfRange = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sge, posIndex, size);
  return builder.create<mlir::arith::SelectOp>(loc, isOutOfRange, size,
                                               posIndex);
}

static mlir::Value computeCount(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value begin, mlir::Value end,
                                mlir::Value step) {
  auto size = builder.createOrFold<mlir::arith::SubIOp>(loc, end, begin);
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  size = builder.createOrFold<mlir::arith::SubIOp>(loc, size, one);
  size = builder.createOrFold<mlir::arith::AddIOp>(loc, size, step);
  size = builder.createOrFold<mlir::arith::DivUIOp>(loc, size, step);
  return size;
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
    auto size = op.getSize();
    std::array<mlir::Value, 4> results;
    if (auto begin = buildSlice.getBegin()) {
      results[0] = handleSliceIndexVars(rewriter, loc, begin, size);
    } else {
      results[0] = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    }

    if (auto end = buildSlice.getEnd()) {
      results[1] = handleSliceIndexVars(rewriter, loc, end, size);
    } else {
      results[1] = size;
    }

    if (auto step = buildSlice.getStep()) {
      results[2] = step;
    } else {
      results[2] = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    }

    results[3] =
        computeCount(rewriter, loc, results[0], results[1], results[2]);

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};
} // namespace

void imex::ntensor::ResolveSliceOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<ResolveSlicePropagate>(context);
}

namespace {
struct ResolveIndexPropagate
    : public mlir::OpRewritePattern<imex::ntensor::ResolveIndexOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ntensor::ResolveIndexOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto res = handleSliceIndexVars(rewriter, op->getLoc(), op.getIndex(),
                                    op.getSize());
    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};
} // namespace

void imex::ntensor::DimOp::build(mlir::OpBuilder &builder,
                                 mlir::OperationState &result,
                                 mlir::Value source, int64_t index) {
  auto loc = result.location;
  auto indexValue = builder.create<mlir::arith::ConstantIndexOp>(loc, index);
  build(builder, result, source, indexValue);
}

void imex::ntensor::DimOp::build(mlir::OpBuilder &builder,
                                 mlir::OperationState &result,
                                 mlir::Value source, mlir::Value index) {
  auto indexTy = builder.getIndexType();
  build(builder, result, indexTy, source, index);
}

imex::ntensor::NTensorType imex::ntensor::SubviewOp::inferResultType(
    imex::ntensor::NTensorType sourceType,
    mlir::ArrayRef<int64_t> staticOffsets, mlir::ArrayRef<int64_t> staticSizes,
    mlir::ArrayRef<int64_t> staticStrides) {
  unsigned rank = sourceType.getRank();
  (void)rank;
  assert(staticOffsets.size() == rank && "staticOffsets length mismatch");
  assert(staticSizes.size() == rank && "staticSizes length mismatch");
  assert(staticStrides.size() == rank && "staticStrides length mismatch");
  return imex::ntensor::NTensorType::get(
      staticSizes, sourceType.getElementType(), sourceType.getEnvironment());
}

imex::ntensor::NTensorType imex::ntensor::SubviewOp::inferResultType(
    imex::ntensor::NTensorType sourceShapedTensorType,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides) {
  mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  mlir::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets,
                             mlir::ShapedType::kDynamicStrideOrOffset);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes,
                             mlir::ShapedType::kDynamicSize);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides,
                             mlir::ShapedType::kDynamicStrideOrOffset);
  return SubviewOp::inferResultType(sourceShapedTensorType, staticOffsets,
                                    staticSizes, staticStrides);
}

imex::ntensor::NTensorType imex::ntensor::SubviewOp::inferRankReducedResultType(
    mlir::ArrayRef<int64_t> resultShape, imex::ntensor::NTensorType sourceType,
    mlir::ArrayRef<int64_t> offsets, mlir::ArrayRef<int64_t> sizes,
    mlir::ArrayRef<int64_t> strides) {
  auto inferredType = inferResultType(sourceType, offsets, sizes, strides);
  assert(inferredType.getRank() >= static_cast<int64_t>(resultShape.size()) &&
         "expected ");
  if (inferredType.getRank() == static_cast<int64_t>(resultShape.size()))
    return inferredType;

  assert(mlir::computeRankReductionMask(inferredType.getShape(), resultShape)
             .has_value() &&
         "invalid rank reduction");

  return imex::ntensor::NTensorType::get(
      resultShape, sourceType.getElementType(), sourceType.getEnvironment());
}

imex::ntensor::NTensorType imex::ntensor::SubviewOp::inferRankReducedResultType(
    mlir::ArrayRef<int64_t> resultShape, imex::ntensor::NTensorType sourceType,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides) {
  mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  mlir::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets,
                             mlir::ShapedType::kDynamicStrideOrOffset);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes,
                             mlir::ShapedType::kDynamicSize);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides,
                             mlir::ShapedType::kDynamicStrideOrOffset);
  return SubviewOp::inferRankReducedResultType(
      resultShape, sourceType, staticOffsets, staticSizes, staticStrides);
}

// Build a SubViewOp with mixed static and dynamic entries and custom result
// type. If the type passed is nullptr, it is inferred.
void imex::ntensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result,
    imex::ntensor::NTensorType resultType, mlir::Value source,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  mlir::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets,
                             mlir::ShapedType::kDynamicStrideOrOffset);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes,
                             mlir::ShapedType::kDynamicSize);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides,
                             mlir::ShapedType::kDynamicStrideOrOffset);
  auto sourceType = source.getType().cast<imex::ntensor::NTensorType>();
  // Structuring implementation this way avoids duplication between builders.
  if (!resultType) {
    resultType = imex::ntensor::SubviewOp::inferResultType(
        sourceType, staticOffsets, staticSizes, staticStrides);
  }
  build(b, result, resultType, source, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getI64ArrayAttr(staticOffsets),
        b.getI64ArrayAttr(staticSizes), b.getI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

// Build a SubViewOp with mixed static and dynamic entries and inferred result
// type.
void imex::ntensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result, mlir::Value source,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  build(b, result, imex::ntensor::NTensorType(), source, offsets, sizes,
        strides, attrs);
}

// Build a SubViewOp with static entries and inferred result type.
void imex::ntensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result, mlir::Value source,
    mlir::ArrayRef<int64_t> offsets, mlir::ArrayRef<int64_t> sizes,
    mlir::ArrayRef<int64_t> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<mlir::OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  mlir::SmallVector<mlir::OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  mlir::SmallVector<mlir::OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  build(b, result, source, offsetValues, sizeValues, strideValues, attrs);
}

// Build a SubViewOp with dynamic entries and custom result type. If the
// type passed is nullptr, it is inferred.
void imex::ntensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result,
    imex::ntensor::NTensorType resultType, mlir::Value source,
    mlir::ArrayRef<int64_t> offsets, mlir::ArrayRef<int64_t> sizes,
    mlir::ArrayRef<int64_t> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<mlir::OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  mlir::SmallVector<mlir::OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  mlir::SmallVector<mlir::OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  build(b, result, resultType, source, offsetValues, sizeValues, strideValues,
        attrs);
}

// Build a SubViewOp with dynamic entries and custom result type. If the type
// passed is nullptr, it is inferred.
void imex::ntensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result,
    imex::ntensor::NTensorType resultType, mlir::Value source,
    mlir::ValueRange offsets, mlir::ValueRange sizes, mlir::ValueRange strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<mlir::OpFoldResult> offsetValues =
      llvm::to_vector<4>(llvm::map_range(
          offsets, [](mlir::Value v) -> mlir::OpFoldResult { return v; }));
  mlir::SmallVector<mlir::OpFoldResult> sizeValues =
      llvm::to_vector<4>(llvm::map_range(
          sizes, [](mlir::Value v) -> mlir::OpFoldResult { return v; }));
  mlir::SmallVector<mlir::OpFoldResult> strideValues =
      llvm::to_vector<4>(llvm::map_range(
          strides, [](mlir::Value v) -> mlir::OpFoldResult { return v; }));
  build(b, result, resultType, source, offsetValues, sizeValues, strideValues);
}

// Build a SubViewOp with dynamic entries and inferred result type.
void imex::ntensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result, mlir::Value source,
    mlir::ValueRange offsets, mlir::ValueRange sizes, mlir::ValueRange strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  build(b, result, imex::ntensor::NTensorType(), source, offsets, sizes,
        strides, attrs);
}

void imex::ntensor::ResolveIndexOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<ResolveIndexPropagate>(context);
}

// Copypasted from upstream tensor.
llvm::SmallBitVector imex::ntensor::SubviewOp::getDroppedDims() {
  mlir::ArrayRef<int64_t> resultShape = getType().getShape();
  mlir::SmallVector<mlir::OpFoldResult> mixedSizes = getMixedSizes();
  llvm::SmallBitVector droppedDims(mixedSizes.size());
  unsigned shapePos = 0;
  for (const auto &size : enumerate(mixedSizes)) {
    llvm::Optional<int64_t> sizeVal = getConstantIntValue(size.value());
    // If the size is not 1, or if the current matched dimension of the result
    // is the same static shape as the size value (which is 1), then the
    // dimension is preserved.
    if (!sizeVal || *sizeVal != 1 ||
        (shapePos < resultShape.size() && resultShape[shapePos] == 1)) {
      shapePos++;
      continue;
    }
    droppedDims.set(size.index());
  }
  return droppedDims;
}

mlir::OpFoldResult imex::ntensor::FromTensorOp::fold(
    llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  if (auto to = getTensor().getDefiningOp<imex::ntensor::ToTensorOp>()) {
    auto array = to.getArray();
    if (getType() == array.getType())
      return array;
  }
  return nullptr;
}

mlir::OpFoldResult
imex::ntensor::ToTensorOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  if (auto from = getArray().getDefiningOp<imex::ntensor::FromTensorOp>()) {
    auto val = from.getTensor();
    auto haveOtherOps = [](mlir::Operation *op) -> bool {
      for (auto user : op->getUsers()) {
        if (!mlir::isa<ToTensorOp>(user))
          return true;
      }
      return false;
    };

    // This folding only safe if we don't have writes to the other fromTensor
    // results. Conservatively check there are no other ops except ToTensorOp.
    if (getType() == val.getType() && !haveOtherOps(from))
      return val;
  }
  return nullptr;
}

mlir::OpFoldResult imex::ntensor::FromMemrefOp::fold(
    llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  if (auto to = getMemref().getDefiningOp<imex::ntensor::ToMemrefOp>()) {
    auto array = to.getArray();
    if (getType() == array.getType())
      return array;
  }
  return nullptr;
}

mlir::OpFoldResult
imex::ntensor::ToMemrefOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  if (auto from = getArray().getDefiningOp<imex::ntensor::FromMemrefOp>()) {
    auto val = from.getMemref();
    if (getType() == val.getType())
      return val;
  }
  return nullptr;
}

// Copypasted from upstream tensor.
mlir::LogicalResult imex::ntensor::SubviewOp::reifyResultShapes(
    mlir::OpBuilder &builder,
    mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0].reserve(getType().getRank());
  mlir::SmallVector<mlir::OpFoldResult> mixedSizes = getMixedSizes();
  llvm::SmallBitVector droppedDims = getDroppedDims();
  mlir::Location loc = getLoc();
  for (const auto &size : enumerate(mixedSizes)) {
    if (droppedDims.test(size.index()))
      continue;
    if (auto attr = size.value().dyn_cast<mlir::Attribute>()) {
      reifiedReturnShapes[0].push_back(
          builder.create<mlir::arith::ConstantIndexOp>(
              loc, attr.cast<mlir::IntegerAttr>().getInt()));
      continue;
    }
    reifiedReturnShapes[0].push_back(size.value().get<mlir::Value>());
  }
  return mlir::success();
}

bool imex::ntensor::CastOp::areCastCompatible(mlir::TypeRange inputs,
                                              mlir::TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  mlir::Type a = inputs.front(), b = outputs.front();
  auto aT = a.dyn_cast<imex::ntensor::NTensorType>();
  auto bT = b.dyn_cast<imex::ntensor::NTensorType>();
  if (!aT || !bT)
    return false;

  if (aT.getElementType() != bT.getElementType())
    return false;

  return succeeded(mlir::verifyCompatibleShape(aT, bT));
}

void imex::ntensor::ElementwiseOp::build(
    ::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
    ::mlir::Type resultType, ::mlir::Value source,
    ::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location,
                              ::mlir::Value)>
        bodyBuilder) {
  assert(source.getType().isa<imex::ntensor::NTensorType>() &&
         "Expected ntensor type");
  build(odsBuilder, odsState, mlir::TypeRange(resultType), source);
  if (bodyBuilder) {
    mlir::Region *bodyRegion = odsState.regions.back().get();
    bodyRegion->push_back(new mlir::Block);
    mlir::Block &bodyBlock = bodyRegion->front();
    auto srcType = source.getType().cast<imex::ntensor::NTensorType>();
    bodyBlock.addArgument(srcType.getElementType(), odsState.location);

    mlir::OpBuilder::InsertionGuard guard(odsBuilder);
    odsBuilder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(odsBuilder, odsState.location, bodyBlock.getArgument(0));
  }
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
