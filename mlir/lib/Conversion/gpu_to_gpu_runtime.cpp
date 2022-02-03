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

#include "mlir-extensions/Conversion/gpu_to_gpu_runtime.hpp"

static const char *kGpuAllocShared = "gpu.alloc_shared";

struct InsertGPUAllocs
    : public mlir::PassWrapper<InsertGPUAllocs, mlir::FunctionPass> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnFunction() override {
    auto func = getFunction();
    auto &funcBody = func.getBody();
    if (funcBody.empty()) {
      return;
    } else if (!llvm::hasSingleElement(funcBody)) {
      func.emitError("Function must have exactly one block");
      signalPassFailure();
      return;
    }

    struct AccessType {
      bool hostRead = false;
      bool hostWrite = false;
      bool deviceRead = false;
      bool deviceWrite = false;
    };

    llvm::SmallMapVector<mlir::Operation *, AccessType, 8> gpuBufferAllocs;
    llvm::SmallMapVector<unsigned, AccessType, 8> gpuBufferParams;
    auto &aliases = getAnalysis<mlir::BufferViewFlowAnalysis>();

    auto getMemref = [](mlir::Operation *op)
        -> llvm::Optional<mlir::SmallVector<mlir::Value, 4>> {
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
        return {{load.memref()}};
      } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
        return {{store.memref()}};
      } else if (auto call = mlir::dyn_cast<mlir::CallOp>(op)) {
        mlir::SmallVector<mlir::Value, 4> ret;
        for (auto arg : call.operands()) {
          if (arg.getType().isa<mlir::MemRefType>())
            ret.emplace_back(arg);
        }
        return std::move(ret);
      } else {
        op->emitError("Uhhandled mem op in gpu region");
        return llvm::None;
      }
    };

    auto scfDialect = getContext().getOrLoadDialect<mlir::scf::SCFDialect>();

    auto hasMemAccess = [](mlir::Operation *op) -> bool {
      if (auto memInterface =
              mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
        if (memInterface.hasEffect<mlir::MemoryEffects::Read>() ||
            memInterface.hasEffect<mlir::MemoryEffects::Write>())
          return true;
      }
      if (auto call = mlir::dyn_cast<mlir::CallOp>(op)) {
        for (auto arg : call.operands()) {
          if (arg.getType().isa<mlir::MemRefType>())
            return true;
        }
      }
      return false;
    };

    if (func.walk([&](mlir::Operation *op) {
              if (!op->getParentOfType<mlir::gpu::LaunchOp>())
                return mlir::WalkResult::advance();

              if (!hasMemAccess(op))
                return mlir::WalkResult::advance();

              auto memref = getMemref(op);
              if (!memref)
                return mlir::WalkResult::interrupt();

              for (auto mem : *memref) {
                while (auto parentView =
                           mem.getDefiningOp<mlir::ViewLikeOpInterface>())
                  mem = parentView.getViewSource();

                for (auto alias : aliases.resolve(mem)) {
                  auto op = alias.getDefiningOp();
                  if (op) {
                    if (op->getDialect() == scfDialect ||
                        mlir::isa<mlir::ViewLikeOpInterface>(op))
                      continue;

                    auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op);
                    if (!allocOp) {
                      op->emitError("Unhandled memref producer");
                      return mlir::WalkResult::interrupt();
                    }

                    gpuBufferAllocs.insert({allocOp, {}});
                  } else {
                    auto block = alias.getParentBlock();
                    auto blockArgs = block->getArguments();
                    auto it = llvm::find(blockArgs, alias);
                    assert(it != blockArgs.end());
                    auto index = it - blockArgs.begin();
                    gpuBufferParams.insert({static_cast<unsigned>(index), {}});
                  }
                }
              }

              return mlir::WalkResult::advance();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }

    auto getAccessType = [&](mlir::Value memref) {
      AccessType ret;
      for (auto mem : aliases.resolve(memref)) {
        for (auto user : mem.getUsers()) {
          if (mlir::isa<mlir::ReturnOp>(user)) {
            ret.hostRead = true;
            ret.hostWrite = true;
            continue;
          }

          if (auto copy = mlir::dyn_cast<mlir::memref::CopyOp>(user)) {
            if (copy.source() == mem)
              ret.hostRead = true;

            if (copy.target() == mem)
              ret.hostWrite = true;

            continue;
          }

          if (auto memInterface =
                  mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            bool onDevice = user->getParentOfType<mlir::gpu::LaunchOp>();
            if (memInterface.hasEffect<mlir::MemoryEffects::Read>())
              (onDevice ? ret.deviceRead : ret.hostRead) = true;

            if (memInterface.hasEffect<mlir::MemoryEffects::Write>())
              (onDevice ? ret.deviceWrite : ret.hostWrite) = true;

            continue;
          }
          if (mlir::isa<mlir::CallOp>(user)) {
            bool onDevice = user->getParentOfType<mlir::gpu::LaunchOp>();
            (onDevice ? ret.deviceRead : ret.hostRead) = true;
            (onDevice ? ret.deviceWrite : ret.hostWrite) = true;
            continue;
          }
        }
      }
      return ret;
    };

    for (auto &it : gpuBufferAllocs) {
      auto alloc = mlir::cast<mlir::memref::AllocOp>(it.first);
      it.second = getAccessType(alloc);
    }

    auto &block = funcBody.front();
    for (auto &it : gpuBufferParams) {
      auto param = block.getArgument(it.first);
      it.second = getAccessType(param);

      it.second.hostRead = true;
      it.second.hostWrite = true;
    }

    mlir::OpBuilder builder(func);
    for (auto it : gpuBufferAllocs) {
      auto alloc = mlir::cast<mlir::memref::AllocOp>(it.first);
      auto access = it.second;
      auto loc = alloc.getLoc();
      builder.setInsertionPoint(alloc);
      auto gpuAlloc = builder.create<mlir::gpu::AllocOp>(
          loc, alloc.getType(), /*asyncToken*/ nullptr,
          /*asyncDependencies*/ llvm::None, alloc.dynamicSizes(),
          alloc.symbolOperands());
      alloc->replaceAllUsesWith(gpuAlloc);
      alloc.erase();
      if (access.hostRead || access.hostWrite)
        gpuAlloc->setAttr(kGpuAllocShared, builder.getUnitAttr());
    }

    auto term = block.getTerminator();
    assert(term);

    llvm::SmallVector<mlir::Value> dims;
    llvm::SmallPtrSet<mlir::Operation *, 8> filter;
    for (auto it : gpuBufferParams) {
      auto param = block.getArgument(it.first);
      auto access = it.second;
      auto loc = param.getLoc();
      builder.setInsertionPointToStart(&block);
      auto memrefType = param.getType().cast<mlir::MemRefType>();
      auto rank = static_cast<unsigned>(memrefType.getRank());
      dims.resize(rank);
      filter.clear();
      for (auto i : llvm::seq(0u, rank)) {
        auto op = builder.create<mlir::memref::DimOp>(loc, param, i);
        dims[i] = op;
        filter.insert(op);
      }
      auto allocType = mlir::MemRefType::get(
          memrefType.getShape(), memrefType.getElementType(),
          mlir::MemRefLayoutAttrInterface{}, memrefType.getMemorySpace());
      auto gpuAlloc = builder.create<mlir::gpu::AllocOp>(
          loc, allocType, /*asyncToken*/ nullptr,
          /*asyncDependencies*/ llvm::None, dims,
          /*symbolOperands*/ llvm::None);
      auto allocResult = gpuAlloc.getResult(0);

      if (access.hostRead || access.hostWrite)
        gpuAlloc->setAttr(kGpuAllocShared, builder.getUnitAttr());

      if (access.hostWrite && access.deviceRead) {
        auto copy =
            builder.create<mlir::memref::CopyOp>(loc, param, allocResult);
        filter.insert(copy);
      }

      if (allocType != memrefType) {
        allocResult =
            builder.create<mlir::memref::CastOp>(loc, allocResult, memrefType);
      }

      param.replaceAllUsesExcept(allocResult, filter);
      builder.setInsertionPoint(term);
      if (access.hostRead && access.deviceWrite)
        builder.create<mlir::memref::CopyOp>(loc, allocResult, param);

      builder.create<mlir::memref::DeallocOp>(loc, allocResult);
    }
  }
};

static mlir::LogicalResult processAllocUser(mlir::Operation *user,
                                            mlir::Operation *allocParent,
                                            mlir::DominanceInfo &dom,
                                            mlir::Operation *&lastUser) {
  auto origUser = user;
  if (user->hasTrait<mlir::OpTrait::IsTerminator>())
    return mlir::failure();

  auto parent = user->getParentOp();
  while (parent != allocParent) {
    user = parent;
    parent = user->getParentOp();
    if (parent == nullptr)
      return mlir::failure();
  }

  if (dom.properlyDominates(lastUser, user))
    lastUser = user;

  for (auto resUser : origUser->getUsers())
    if (mlir::failed(processAllocUser(resUser, allocParent, dom, lastUser)))
      return mlir::failure();

  return mlir::success();
}

template <typename AllocOp, typename DeallocOp>
struct CreateDeallocOp : public mlir::OpRewritePattern<AllocOp> {
  using mlir::OpRewritePattern<AllocOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AllocOp op, mlir::PatternRewriter &rewriter) const override {
    auto allocParent = op->getParentOp();
    mlir::Operation *lastUser = op;
    mlir::DominanceInfo dom;
    for (auto user : op->getUsers())
      if (mlir::isa<DeallocOp>(user) ||
          mlir::failed(processAllocUser(user, allocParent, dom, lastUser)))
        return mlir::failure();

    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(lastUser);
    rewriter.create<DeallocOp>(lastUser->getLoc(), op);
    return mlir::success();
  }
};

static llvm::Optional<mlir::Value> getGpuStream(mlir::OpBuilder &builder,
                                                mlir::Operation *op) {
  assert(op);
  auto func = op->getParentOfType<mlir::FuncOp>();
  if (!func)
    return {};

  if (!llvm::hasSingleElement(func.getBody()))
    return {};

  auto &block = func.getBody().front();
  auto ops = block.getOps<gpu_runtime::CreateGpuStreamOp>();
  if (!ops.empty())
    return (*ops.begin()).getResult();

  mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(&block);
  auto loc = builder.getUnknownLoc();
  auto stream = builder.create<gpu_runtime::CreateGpuStreamOp>(loc).getResult();
  builder.setInsertionPoint(block.getTerminator());
  builder.create<gpu_runtime::DestroyGpuStreamOp>(loc, stream);
  return stream;
}

class ConvertSubviewOp
    : public mlir::OpConversionPattern<mlir::memref::SubViewOp> {
public:
  using mlir::OpConversionPattern<mlir::memref::SubViewOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::SubViewOp op,
                  mlir::memref::SubViewOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dstType = op.getType().cast<mlir::MemRefType>();
    if (!dstType.hasRank() || dstType.getRank() != 1)
      return mlir::failure();

    auto intType = getTypeConverter()->convertType(rewriter.getIndexType());
    if (!intType)
      return mlir::failure();

    auto loc = op.getLoc();
    auto getValue = [&](mlir::OpFoldResult src) -> mlir::Value {
      if (auto val = src.dyn_cast<mlir::Value>())
        return val;

      auto attr = src.get<mlir::Attribute>();
      return rewriter.create<mlir::spirv::ConstantOp>(loc, intType, attr);
    };

    auto offset =
        getValue(op.isDynamicOffset(0)
                     ? mlir::OpFoldResult(adaptor.offsets()[0])
                     : mlir::OpFoldResult(adaptor.static_offsets()[0]));
    auto stride =
        getValue(op.isDynamicStride(0)
                     ? mlir::OpFoldResult(adaptor.strides()[0])
                     : mlir::OpFoldResult(adaptor.static_strides()[0]));
    auto finalOffset = rewriter.createOrFold<mlir::spirv::IMulOp>(
        loc, intType, offset, stride);

    auto ptr = rewriter
                   .create<mlir::spirv::InBoundsPtrAccessChainOp>(
                       loc, adaptor.source(), finalOffset, llvm::None)
                   .getResult();

    rewriter.replaceOp(op, ptr);
    return mlir::success();
  }
};

template <typename T>
class ConvertCastOp : public mlir::OpConversionPattern<T> {
public:
  using mlir::OpConversionPattern<T>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.source());
    return mlir::success();
  }
};

class ConvertLoadOp : public mlir::OpConversionPattern<mlir::memref::LoadOp> {
public:
  using mlir::OpConversionPattern<mlir::memref::LoadOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::memref::LoadOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memrefType = op.memref().getType().cast<mlir::MemRefType>();
    if (!memrefType.hasRank() || memrefType.getRank() != 1)
      return mlir::failure();

    auto loc = op.getLoc();
    auto ptr = rewriter.create<mlir::spirv::InBoundsPtrAccessChainOp>(
        loc, adaptor.memref(), adaptor.indices().front(), llvm::None);

    auto memoryAccess = mlir::spirv::MemoryAccessAttr::get(
        op.getContext(), mlir::spirv::MemoryAccess::Aligned);
    auto alignment =
        rewriter.getI32IntegerAttr(memrefType.getElementTypeBitWidth() / 8);
    rewriter.replaceOpWithNewOp<mlir::spirv::LoadOp>(op, ptr, memoryAccess,
                                                     alignment);

    return mlir::success();
  }
};

class ConvertStoreOp : public mlir::OpConversionPattern<mlir::memref::StoreOp> {
public:
  using mlir::OpConversionPattern<mlir::memref::StoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::memref::StoreOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memrefType = op.memref().getType().cast<mlir::MemRefType>();
    if (!memrefType.hasRank() || memrefType.getRank() != 1)
      return mlir::failure();

    auto loc = op.getLoc();
    auto ptr = rewriter.create<mlir::spirv::InBoundsPtrAccessChainOp>(
        loc, adaptor.memref(), adaptor.indices().front(), llvm::None);

    auto memoryAccess = mlir::spirv::MemoryAccessAttr::get(
        op.getContext(), mlir::spirv::MemoryAccess::Aligned);
    auto alignment =
        rewriter.getI32IntegerAttr(memrefType.getElementTypeBitWidth() / 8);
    rewriter.replaceOpWithNewOp<mlir::spirv::StoreOp>(op, ptr, adaptor.value(),
                                                      memoryAccess, alignment);

    return mlir::success();
  }
};

class ConvertReduceRank
    : public mlir::OpConversionPattern<gpu_runtime::ReduceRankOp> {
public:
  using mlir::OpConversionPattern<
      gpu_runtime::ReduceRankOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::ReduceRankOp op,
                  gpu_runtime::ReduceRankOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.source());
    return mlir::success();
  }
};

template <typename Op>
static mlir::Value lowerIntAtomic(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value ptr, mlir::Value val) {
  return builder.create<Op>(loc, ptr, mlir::spirv::Scope::Device,
                            mlir::spirv::MemorySemantics::None, val);
}

static mlir::Value lowerFloatAddAtomic(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value ptr,
                                       mlir::Value val) {
  return builder.create<mlir::spirv::AtomicFAddEXTOp>(
      loc, val.getType(), ptr, mlir::spirv::Scope::Device,
      mlir::spirv::MemorySemantics::None, val);
}

static mlir::Value lowerFloatSubAtomic(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value ptr,
                                       mlir::Value val) {
  auto neg = builder.create<mlir::spirv::FNegateOp>(loc, val).getResult();
  return builder.create<mlir::spirv::AtomicFAddEXTOp>(
      loc, neg.getType(), ptr, mlir::spirv::Scope::Device,
      mlir::spirv::MemorySemantics::None, neg);
}

class ConvertAtomicOps : public mlir::OpConversionPattern<mlir::CallOp> {
public:
  using mlir::OpConversionPattern<mlir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::CallOp op, mlir::CallOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.operands();
    if (operands.size() != 2)
      return mlir::failure();

    if (op.getNumResults() != 1)
      return mlir::failure();

    auto ptr = operands[0];
    auto ptrType = ptr.getType().dyn_cast<mlir::spirv::PointerType>();
    if (!ptrType)
      return mlir::failure();

    auto val = operands[1];
    auto valType = val.getType();
    if (ptrType.getPointeeType() != valType)
      return mlir::failure();

    bool isInt;
    if (valType.isSignlessInteger())
      isInt = true;
    else if (valType.isa<mlir::FloatType>())
      isInt = false;
    else
      return mlir::failure();

    auto funcName = op.getCallee();

    using func_t = mlir::Value (*)(mlir::OpBuilder &, mlir::Location,
                                   mlir::Value, mlir::Value);

    struct Desc {
      mlir::StringRef name;
      func_t intOp;
      func_t floatOp;
    };

    const Desc handlers[] = {
        {"atomic_add", &lowerIntAtomic<mlir::spirv::AtomicIAddOp>,
         &lowerFloatAddAtomic},
        {"atomic_sub", &lowerIntAtomic<mlir::spirv::AtomicISubOp>,
         &lowerFloatSubAtomic},
    };

    auto handler = [&]() -> func_t {
      for (auto &h : handlers) {
        if (funcName.consume_front(h.name))
          return (isInt ? h.intOp : h.floatOp);
      }
      return nullptr;
    }();

    if (handler) {
      auto res = handler(rewriter, op.getLoc(), ptr, val);
      if (res) {
        rewriter.replaceOp(op, res);
        return mlir::success();
      }
    }

    return mlir::failure();
  }
};

// TODO: something better
class ConvertFunc : public mlir::OpConversionPattern<mlir::FuncOp> {
public:
  using mlir::OpConversionPattern<mlir::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::FuncOp op, mlir::FuncOp::Adaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.body().empty())
      return mlir::failure();

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct GPUToSpirvPass
    : public mlir::PassWrapper<GPUToSpirvPass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    auto module = getOperation();

    llvm::SmallVector<mlir::Operation *, 1> kernelModules;
    mlir::OpBuilder builder(context);
    module.walk([&builder, &kernelModules](mlir::gpu::GPUModuleOp moduleOp) {
      // For each kernel module (should be only 1 for now, but that is not a
      // requirement here), clone the module for conversion because the
      // gpu.launch function still needs the kernel module.
      builder.setInsertionPoint(moduleOp.getOperation());
      kernelModules.push_back(builder.clone(*moduleOp.getOperation()));
    });

    auto targetAttr = mlir::spirv::lookupTargetEnvOrDefault(module);
    auto target = mlir::SPIRVConversionTarget::get(targetAttr);

    mlir::SPIRVTypeConverter::Options options;
    options.use64bitIndex = true;

    mlir::SPIRVTypeConverter typeConverter(targetAttr, options);
    mlir::RewritePatternSet patterns(context);

    typeConverter.addConversion(
        [](mlir::MemRefType type) -> llvm::Optional<mlir::Type> {
          if (type.hasRank() && type.getElementType().isIntOrFloat())
            return mlir::spirv::PointerType::get(
                type.getElementType(),
                mlir::spirv::StorageClass::CrossWorkgroup);
          return mlir::Type(nullptr);
        });

    mlir::ScfToSPIRVContext scfToSpirvCtx;
    mlir::populateSCFToSPIRVPatterns(typeConverter, scfToSpirvCtx, patterns);
    mlir::populateGPUToSPIRVPatterns(typeConverter, patterns);
    mlir::populateStandardToSPIRVPatterns(typeConverter, patterns);
    mlir::arith::populateArithmeticToSPIRVPatterns(typeConverter, patterns);
    mlir::populateMathToSPIRVPatterns(typeConverter, patterns);

    patterns.insert<ConvertSubviewOp, ConvertCastOp<mlir::memref::CastOp>,
                    ConvertCastOp<mlir::memref::ReinterpretCastOp>,
                    ConvertLoadOp, ConvertStoreOp, ConvertReduceRank,
                    ConvertAtomicOps, ConvertFunc>(typeConverter, context);

    if (failed(
            applyFullConversion(kernelModules, *target, std::move(patterns))))
      return signalPassFailure();
  }
};

template <typename Op, typename F>
static mlir::LogicalResult createGpuKernelLoad(mlir::PatternRewriter &builder,
                                               Op &&op, F &&func) {
  auto mod = op->template getParentOfType<mlir::ModuleOp>();
  if (!mod)
    return mlir::failure();

  auto gpuMod = mod.template lookupSymbol<mlir::gpu::GPUModuleOp>(
      op.getKernelModuleName());
  if (!gpuMod)
    return mlir::failure();

  auto gpuKernel =
      gpuMod.template lookupSymbol<mlir::gpu::GPUFuncOp>(op.getKernelName());
  if (!gpuKernel)
    return mlir::failure();

  auto stream = getGpuStream(builder, op);
  if (!stream)
    return mlir::failure();

  auto loc = op.getLoc();
  auto module =
      builder.create<gpu_runtime::LoadGpuModuleOp>(loc, *stream, gpuMod);
  auto kernel =
      builder.create<gpu_runtime::GetGpuKernelOp>(loc, module, gpuKernel);
  auto newOp = func(builder, loc, *stream, kernel);
  builder.replaceOp(op, newOp.getResults());
  return mlir::success();
}

struct ExpandLaunchOp : public mlir::OpRewritePattern<mlir::gpu::LaunchFuncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::LaunchFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    return createGpuKernelLoad(
        rewriter, op,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value stream,
            mlir::Value kernel) {
          return builder.create<gpu_runtime::LaunchGpuKernelOp>(
              loc, stream, kernel, op.getGridSizeOperandValues(),
              op.getBlockSizeOperandValues(), op.operands());
        });
  }
};

struct ExpandAllocOp : public mlir::OpRewritePattern<mlir::gpu::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::AllocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto stream = getGpuStream(rewriter, op);
    if (!stream)
      return mlir::failure();

    auto shared = op->hasAttr(kGpuAllocShared);

    mlir::Type token = op.asyncToken() ? op.asyncToken().getType() : nullptr;
    auto res = rewriter.replaceOpWithNewOp<gpu_runtime::GPUAllocOp>(
        op, op.getType(), token, op.asyncDependencies(), *stream,
        op.dynamicSizes(), op.symbolOperands());

    if (shared)
      res->setAttr(kGpuAllocShared, rewriter.getUnitAttr());

    return mlir::success();
  }
};

struct ExpandSuggestBlockSizeOp
    : public mlir::OpRewritePattern<gpu_runtime::GPUSuggestBlockSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUSuggestBlockSizeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.kernel())
      return mlir::failure();

    assert(op.kernelRef());
    return createGpuKernelLoad(
        rewriter, op,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value stream,
            mlir::Value kernel) {
          return builder.create<gpu_runtime::GPUSuggestBlockSizeOp>(
              loc, stream, kernel, op.gridSize());
        });
  }
};

struct SetSPIRVCapabilitiesPass
    : public mlir::PassWrapper<SetSPIRVCapabilitiesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    namespace spirv = mlir::spirv;
    auto context = &getContext();
    spirv::Capability caps[] = {
        // clang-format off
        spirv::Capability::Addresses,
        spirv::Capability::Float16Buffer,
        spirv::Capability::Int64,
        spirv::Capability::Int16,
        spirv::Capability::Int8,
        spirv::Capability::Kernel,
        spirv::Capability::Linkage,
        spirv::Capability::Vector16,
        spirv::Capability::GenericPointer,
        spirv::Capability::Groups,
        spirv::Capability::Float16,
        spirv::Capability::Float64,
        spirv::Capability::AtomicFloat32AddEXT,
        // clang-format on
    };
    spirv::Extension exts[] = {
        spirv::Extension::SPV_EXT_shader_atomic_float_add};
    auto triple =
        spirv::VerCapExtAttr::get(spirv::Version::V_1_0, caps, exts, context);
    auto attr = spirv::TargetEnvAttr::get(
        triple, spirv::Vendor::Unknown, spirv::DeviceType::Unknown,
        spirv::TargetEnvAttr::kUnknownDeviceID,
        spirv::getDefaultResourceLimits(context));
    auto module = getOperation();
    module->setAttr(spirv::getTargetEnvAttrName(), attr);
  }
};

struct SerializeSPIRVPass
    : public mlir::PassWrapper<SerializeSPIRVPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto mod = getOperation();

    namespace gpu = mlir::gpu;
    namespace spirv = mlir::spirv;
    llvm::SmallVector<uint32_t, 0> spvBinary;
    for (auto gpuMod : mod.getOps<gpu::GPUModuleOp>()) {
      auto name = gpuMod.getName();
      auto isSameMod = [&](spirv::ModuleOp spvMod) -> bool {
        auto spvModName = spvMod.getName();
        return spvModName->consume_front("__spv__") && spvModName == name;
      };
      auto spvMods = mod.getOps<spirv::ModuleOp>();
      auto it = llvm::find_if(spvMods, isSameMod);
      if (it == spvMods.end()) {
        gpuMod.emitError() << "Unable to find corresponding SPIR-V module";
        signalPassFailure();
        return;
      }
      auto spvMod = *it;

      spvBinary.clear();
      if (mlir::failed(spirv::serialize(spvMod, spvBinary))) {
        spvMod.emitError() << "Failed to serialize SPIR-V module";
        signalPassFailure();
        return;
      }

      auto spvData =
          llvm::StringRef(reinterpret_cast<const char *>(spvBinary.data()),
                          spvBinary.size() * sizeof(uint32_t));
      auto spvAttr = mlir::StringAttr::get(&getContext(), spvData);
      gpuMod->setAttr(gpu::getDefaultGpuBinaryAnnotation(), spvAttr);
      spvMod->erase();
    }
  }
};

struct GPUExPass : public mlir::PassWrapper<GPUExPass, mlir::FunctionPass> {

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<gpu_runtime::GpuRuntimeDialect>();
  }

  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns(&getContext());

    patterns.insert<ExpandLaunchOp, ExpandAllocOp, ExpandSuggestBlockSizeOp>(
        &getContext());

    (void)mlir::applyPatternsAndFoldGreedily(getFunction(),
                                             std::move(patterns));
  }
};

// Expose the passes to the outside world
std::unique_ptr<mlir::Pass> gpu_runtime::runSetSPIRVCapabilitiesPass() {
  return std::make_unique<SetSPIRVCapabilitiesPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::runGPUToSpirvPass() {
  return std::make_unique<GPUToSpirvPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::runInsertGPUAllocsPass() {
  return std::make_unique<InsertGPUAllocs>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::runSerializeSPIRVPass() {
  return std::make_unique<SerializeSPIRVPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::runGPUExPass() {
  return std::make_unique<GPUExPass>();
}
