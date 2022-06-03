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

#include "mlir-extensions/Dialect/gpu_runtime/IR/gpu_runtime_ops.hpp"

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
struct GpuRuntimeInlinerInterface : public mlir::DialectInlinerInterface {
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

namespace gpu_runtime {
void GpuRuntimeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-extensions/Dialect/gpu_runtime/IR/GpuRuntimeOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir-extensions/Dialect/gpu_runtime/IR/GpuRuntimeOpsAttributes.cpp.inc"
      >();
  addTypes<OpaqueType>();
  addInterfaces<GpuRuntimeInlinerInterface>();
}

mlir::Type GpuRuntimeDialect::parseType(mlir::DialectAsmParser &parser) const {
  parser.emitError(parser.getNameLoc(), "unknown type");
  return mlir::Type();
}

void GpuRuntimeDialect::printType(mlir::Type type,
                                  mlir::DialectAsmPrinter &os) const {
  llvm::TypeSwitch<mlir::Type>(type)
      .Case<gpu_runtime::OpaqueType>([&](auto) { os << "OpaqueType"; })
      .Default([](auto) { llvm_unreachable("unexpected type"); });
}

mlir::Operation *
GpuRuntimeDialect::materializeConstant(mlir::OpBuilder &builder,
                                       mlir::Attribute value, mlir::Type type,
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

void CreateGpuStreamOp::build(::mlir::OpBuilder &odsBuilder,
                              ::mlir::OperationState &odsState) {
  auto ctx = odsBuilder.getContext();
  CreateGpuStreamOp::build(odsBuilder, odsState,
                           gpu_runtime::OpaqueType::get(ctx));
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
  LoadGpuModuleOp::build(odsBuilder, odsState,
                         gpu_runtime::OpaqueType::get(ctx), stream,
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
  GetGpuKernelOp::build(odsBuilder, odsState, gpu_runtime::OpaqueType::get(ctx),
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

mlir::StringRef getAllocSharedAttrName() { return "gpu.alloc_shared"; }

mlir::StringRef getGpuAccessibleAttrName() { return "gpu.gpu_accessible"; }

} // namespace gpu_runtime

#include "mlir-extensions/Dialect/gpu_runtime/IR/GpuRuntimeOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir-extensions/Dialect/gpu_runtime/IR/GpuRuntimeOpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir-extensions/Dialect/gpu_runtime/IR/GpuRuntimeOps.cpp.inc"

#include "mlir-extensions/Dialect/gpu_runtime/IR/GpuRuntimeOpsEnums.cpp.inc"
