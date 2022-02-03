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

#include "mlir-extensions/dialect/gpu_runtime/IR/gpu_runtime_ops.hpp"

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
#include <mlir/Dialect/StandardOps/Utils/Utils.h>

#include <llvm/ADT/TypeSwitch.h>

#include "mlir-extensions/transforms/const_utils.hpp"

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
#include "mlir-extensions/dialect/gpu_runtime/IR/GpuRuntimeOps.cpp.inc"
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

  return builder.create<mlir::ConstantOp>(loc, type, value);
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
  for (auto it : llvm::enumerate(mapping))
    shape[it.index()] = srcShape[static_cast<size_t>(it.value())];

  if (auto tensorType = srcType.dyn_cast<mlir::RankedTensorType>()) {
    auto retType = mlir::RankedTensorType::get(
        shape, tensorType.getElementType(), tensorType.getEncoding());
    build(odsBuilder, odsState, retType, src, mapAttr);
  } else if (auto memrefType = srcType.dyn_cast<mlir::MemRefType>()) {
    auto affineMap = [&]() {
      mlir::AffineMap ret;
      if (!memrefType.getLayout().isIdentity()) {
        auto affineMap = memrefType.getLayout().getAffineMap();
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
                mlir::getAffineSymbolExpr(srcIndex + 1, context);
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

    auto prev = op.source().template getDefiningOp<gpu_runtime::ReduceRankOp>();
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
                                 gpu_runtime::ReduceRankOp src,
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
    auto src = op.memref().getDefiningOp<gpu_runtime::ReduceRankOp>();
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
    auto src = op.memref().getDefiningOp<gpu_runtime::ReduceRankOp>();
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

} // namespace gpu_runtime

#include "mlir-extensions/dialect/gpu_runtime/IR/GpuRuntimeOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mlir-extensions/dialect/gpu_runtime/IR/GpuRuntimeOps.cpp.inc"
