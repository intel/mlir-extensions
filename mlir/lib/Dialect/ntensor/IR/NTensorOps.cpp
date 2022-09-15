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

#include "mlir-extensions/Dialect/ntensor/IR/NTensorOps.hpp"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
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

namespace imex {
namespace ntensor {

void NTensorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-extensions/Dialect/ntensor/IR/NTensorOps.cpp.inc"
      >();

  addInterfaces<NTensorInlinerInterface>();

  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-extensions/Dialect/ntensor/IR/NTensorOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir-extensions/Dialect/ntensor/IR/NTensorOpsAttributes.cpp.inc"
      >();
}

mlir::Operation *NTensorDialect::materializeConstant(mlir::OpBuilder &builder,
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

void NTensorType::walkImmediateSubElements(
    llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
    llvm::function_ref<void(mlir::Type)> walkTypesFn) const {
  walkTypesFn(getElementType());
  if (mlir::Attribute env = getEnvironment())
    walkAttrsFn(env);
}

bool NTensorBase::hasRank() const { return true; }

llvm::ArrayRef<int64_t> NTensorBase::getShape() const {
  return cast<NTensorType>().getShape();
}

NTensorBase
NTensorBase::cloneWith(llvm::Optional<llvm::ArrayRef<int64_t>> shape,
                       Type elementType) const {
  auto t = cast<NTensorType>();
  return NTensorType::get(shape.value_or(getShape()), elementType,
                          t.getEnvironment());
}

mlir::Type NTensorType::replaceImmediateSubElements(
    llvm::ArrayRef<mlir::Attribute> replAttrs,
    llvm::ArrayRef<mlir::Type> replTypes) const {
  return get(getShape(), replTypes.front(),
             replAttrs.empty() ? mlir::Attribute() : replAttrs.back());
}

} // namespace ntensor
} // namespace imex

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

#include "mlir-extensions/Dialect/ntensor/IR/NTensorOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mlir-extensions/Dialect/ntensor/IR/NTensorOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir-extensions/Dialect/ntensor/IR/NTensorOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir-extensions/Dialect/ntensor/IR/NTensorOpsTypes.cpp.inc"

#include "mlir-extensions/Dialect/ntensor/IR/NTensorOpsEnums.cpp.inc"
