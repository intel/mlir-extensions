// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/InliningUtils.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/TypeSwitch.h>

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
#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOpsAttributes.cpp.inc"
      >();

  addInterfaces<GpuRuntimeInlinerInterface>();
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

void CreateGpuStreamOp::build(mlir::OpBuilder &odsBuilder,
                              mlir::OperationState &odsState,
                              mlir::Attribute device) {
  auto ctx = odsBuilder.getContext();
  CreateGpuStreamOp::build(odsBuilder, odsState,
                           gpu_runtime::StreamType::get(ctx), device);
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
                      builder.getDenseI32ArrayAttr(segmentSizes));
}

void GPUSuggestBlockSizeOp::build(::mlir::OpBuilder &odsBuilder,
                                  ::mlir::OperationState &odsState,
                                  ::llvm::Optional<::mlir::Value> stream,
                                  ::mlir::ValueRange gridSize,
                                  ::mlir::Value kernel) {
  auto dimCount = gridSize.size();
  assert(dimCount > 0 && dimCount <= 3);
  llvm::SmallVector<mlir::Type, 3> resTypes(dimCount,
                                            odsBuilder.getIndexType());

  GPUSuggestBlockSizeOp::build(odsBuilder, odsState, resTypes,
                               stream.value_or(mlir::Value{}), kernel,
                               mlir::SymbolRefAttr{}, gridSize);
}

void GPUSuggestBlockSizeOp::build(::mlir::OpBuilder &odsBuilder,
                                  ::mlir::OperationState &odsState,
                                  ::llvm::Optional<::mlir::Value> stream,
                                  ::mlir::ValueRange gridSize,
                                  ::mlir::SymbolRefAttr kernel) {
  auto dimCount = gridSize.size();
  assert(dimCount > 0 && dimCount <= 3);
  llvm::SmallVector<mlir::Type, 3> resTypes(dimCount,
                                            odsBuilder.getIndexType());

  GPUSuggestBlockSizeOp::build(odsBuilder, odsState, resTypes,
                               stream.value_or(mlir::Value{}), mlir::Value{},
                               kernel, gridSize);
}

void GPUSuggestBlockSizeOp::build(::mlir::OpBuilder &odsBuilder,
                                  ::mlir::OperationState &odsState,
                                  ::llvm::Optional<::mlir::Value> stream,
                                  ::mlir::ValueRange gridSize) {
  auto dimCount = gridSize.size();
  assert(dimCount > 0 && dimCount <= 3);
  llvm::SmallVector<mlir::Type, 3> resTypes(dimCount,
                                            odsBuilder.getIndexType());

  GPUSuggestBlockSizeOp::build(odsBuilder, odsState, resTypes,
                               stream.value_or(mlir::Value{}), mlir::Value{},
                               mlir::SymbolRefAttr{}, gridSize);
}

mlir::StringAttr GPUSuggestBlockSizeOp::getKernelModuleName() {
  assert(getKernelRef());
  return getKernelRef()->getRootReference();
}

mlir::StringAttr GPUSuggestBlockSizeOp::getKernelName() {
  assert(getKernelRef());
  return getKernelRef()->getLeafReference();
}

mlir::StringRef getGpuAccessibleAttrName() { return "gpu.gpu_accessible"; }

} // namespace gpu_runtime

// TODO: unify with upstream
/// Parses an optional list of async operands with an optional leading keyword.
/// (`async`)? (`[` ssa-id-list `]`)?
///
/// This method is used by the tablegen assembly format for async ops as well.
static mlir::ParseResult parseAsyncDependencies(
    mlir::OpAsmParser &parser, mlir::Type &asyncTokenType,
    mlir::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>
        &asyncDependencies) {
  auto loc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword("async"))) {
    if (parser.getNumResults() == 0)
      return parser.emitError(loc, "needs to be named when marked 'async'");
    asyncTokenType = parser.getBuilder().getType<mlir::gpu::AsyncTokenType>();
  }
  return parser.parseOperandList(asyncDependencies,
                                 mlir::OpAsmParser::Delimiter::OptionalSquare);
}

/// Prints optional async dependencies with its leading keyword.
///   (`async`)? (`[` ssa-id-list `]`)?
// Used by the tablegen assembly format for several async ops.
static void printAsyncDependencies(mlir::OpAsmPrinter &printer,
                                   mlir::Operation *op,
                                   mlir::Type asyncTokenType,
                                   mlir::OperandRange asyncDependencies) {
  if (asyncTokenType)
    printer << "async";
  if (asyncDependencies.empty())
    return;
  if (asyncTokenType)
    printer << ' ';
  printer << '[';
  llvm::interleaveComma(asyncDependencies, printer);
  printer << ']';
}

#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOpsTypes.cpp.inc"

#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOpsEnums.cpp.inc"
