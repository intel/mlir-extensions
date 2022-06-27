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

#include "mlir-extensions/Dialect/intel_gpu/IR/intel_gpu_ops.hpp"

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

#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/TypeSwitch.h>

#include "mlir-extensions/Transforms/const_utils.hpp"

namespace MemoryEffects = ::mlir::MemoryEffects;

namespace {
struct IntelGpuInlinerInterface : public mlir::DialectInlinerInterface {
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

namespace intel_gpu {
void IntelGpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-extensions/Dialect/intel_gpu/IR/IntelGpuOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir-extensions/Dialect/intel_gpu/IR/IntelGpuOpsAttributes.cpp.inc"
      >();
  addTypes<OpaqueType>();
  addInterfaces<IntelGpuInlinerInterface>();
}

mlir::Type IntelGpuDialect::parseType(mlir::DialectAsmParser &parser) const {
  parser.emitError(parser.getNameLoc(), "unknown type");
  return mlir::Type();
}

void IntelGpuDialect::printType(mlir::Type type,
                                mlir::DialectAsmPrinter &os) const {
  llvm::TypeSwitch<mlir::Type>(type)
      .Case<intel_gpu::OpaqueType>([&](auto) { os << "OpaqueType"; })
      .Default([](auto) { llvm_unreachable("unexpected type"); });
}

mlir::Operation *IntelGpuDialect::materializeConstant(mlir::OpBuilder &builder,
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

OpaqueType OpaqueType::get(mlir::MLIRContext *context) {
  assert(context);
  return Base::get(context);
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

void GetStreamOp::build(::mlir::OpBuilder &odsBuilder,
                        ::mlir::OperationState &odsState) {
  auto ctx = odsBuilder.getContext();
  GetStreamOp::build(odsBuilder, odsState, intel_gpu::OpaqueType::get(ctx));
}

void GetStreamOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<RemoveUnusedOp<GetStreamOp, DestroyStreamOp>>(context);
}

mlir::StringRef getGpuAccessibleAttrName() { return "gpu.gpu_accessible"; }

} // namespace intel_gpu

#include "mlir-extensions/Dialect/intel_gpu/IR/IntelGpuOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir-extensions/Dialect/intel_gpu/IR/IntelGpuOpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir-extensions/Dialect/intel_gpu/IR/IntelGpuOps.cpp.inc"

#include "mlir-extensions/Dialect/intel_gpu/IR/IntelGpuOpsEnums.cpp.inc"
