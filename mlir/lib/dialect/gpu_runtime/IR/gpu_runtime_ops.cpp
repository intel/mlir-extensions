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

#include "gpu_runtime/dialect.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/InliningUtils.h>

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>

#include <llvm/ADT/TypeSwitch.h>

#include "gpu_runtime/transforms/const_utils.hpp"

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
    return !mlir::isa<gpu_runtime::ArgOp>(op);
  }
};
} // namespace

namespace gpu_runtime {

llvm::StringRef attributes::getFastmathName() { return "#gpu_runtime.fastmath"; }

llvm::StringRef attributes::getJumpMarkersName() {
  return "#gpu_runtime.pipeline_jump_markers";
}

llvm::StringRef attributes::getParallelName() { return "#gpu_runtime.parallel"; }

llvm::StringRef attributes::getMaxConcurrencyName() {
  return "#gpu_runtime.max_concurrency";
}

llvm::StringRef attributes::getForceInlineName() {
  return "#gpu_runtime.force_inline";
}

llvm::StringRef attributes::getOptLevelName() { return "#gpu_runtime.opt_level"; }

llvm::StringRef attributes::getGpuRangeName() { return "#gpu_runtime.gpu_range"; }

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

void GpuRuntimeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "gpu_runtime/GpuRuntimeOps.cpp.inc"
      >();
  addTypes<gpu_runtime::PyType, gpu_runtime::LiteralType, SliceType, gpu_runtime::TypeVar,
           OpaqueType>();
  addInterfaces<GpuRuntimeInlinerInterface>();
}

mlir::Type GpuRuntimeDialect::parseType(mlir::DialectAsmParser &parser) const {
  parser.emitError(parser.getNameLoc(), "unknown type");
  return mlir::Type();
}

void GpuRuntimeDialect::printType(mlir::Type type,
                             mlir::DialectAsmPrinter &os) const {
  llvm::TypeSwitch<mlir::Type>(type)
      .Case<gpu_runtime::PyType>(
          [&](auto t) { os << "PyType<" << t.getName() << ">"; })
      .Case<gpu_runtime::LiteralType>([&](auto t) {
        os << "LiteralType<";
        os.printAttribute(t.getValue());
        os << ">";
      })
      .Case<gpu_runtime::SliceType>([&](auto) { os << "SliceType"; })
      .Case<gpu_runtime::TypeVar>([&](auto t) {
        os << "TypeVar<";
        os.printType(t.getType());
        os << ">";
      })
      .Case<gpu_runtime::OpaqueType>([&](auto) { os << "OpaqueType"; })
      .Default([](auto) { llvm_unreachable("unexpected type"); });
}

mlir::Operation *GpuRuntimeDialect::materializeConstant(mlir::OpBuilder &builder,
                                                   mlir::Attribute value,
                                                   mlir::Type type,
                                                   mlir::Location loc) {
  if (mlir::arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<mlir::arith::ConstantOp>(loc, type, value);

  return builder.create<mlir::ConstantOp>(loc, type, value);
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
  CreateGpuStreamOp::build(odsBuilder, odsState, gpu_runtime::OpaqueType::get(ctx));
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
  LoadGpuModuleOp::build(odsBuilder, odsState, gpu_runtime::OpaqueType::get(ctx),
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

#include "gpu_runtime/GpuRuntimeOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "gpu_runtime/GpuRuntimeOps.cpp.inc"
