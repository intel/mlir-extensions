//===- InsertGpuAllocs.cpp - InsertGpuAllocs Pass  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file converts the memref.allocs for device side to gpu.allocs to
/// distinguish between host & device side memory allocations.
///
//===----------------------------------------------------------------------===//

#include <imex/Transforms/Transforms.h>

namespace imex {

mlir::StringRef getAllocSharedAttrName() { return "gpu.alloc_shared"; }

struct InsertGPUAllocs
    : public mlir::PassWrapper<InsertGPUAllocs,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  InsertGPUAllocs() = default;
  InsertGPUAllocs(bool gpuDealloc) { useGpuDealloc = gpuDealloc; }
  InsertGPUAllocs(const InsertGPUAllocs &pass) : PassWrapper(pass) {}

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::arith::ArithmeticDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
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
    llvm::SmallMapVector<mlir::Operation *, AccessType, 8>
        gpuGetMemrefGlobalParams;
    auto &aliases = getAnalysis<mlir::BufferViewFlowAnalysis>();

    auto getMemref = [](mlir::Operation *op)
        -> llvm::Optional<mlir::SmallVector<mlir::Value, 4>> {
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
        return {{load.memref()}};
      } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
        return {{store.memref()}};
      } else if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
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
      if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
        for (auto arg : call.operands()) {
          if (arg.getType().isa<mlir::MemRefType>())
            return true;
        }
      }
      return false;
    };

    // Traverse through all the memory access ops under GPU launch Op
    // and add device memory allocation appropriately
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
                    if (mlir::isa<mlir::memref::GetGlobalOp>(op)) {
                      gpuGetMemrefGlobalParams.insert({op, {}});
                      continue;
                    }
                    if (op->getDialect() == scfDialect ||
                        mlir::isa<mlir::ViewLikeOpInterface>(op))
                      continue;
                    if (mlir::isa<mlir::memref::AllocOp>(op)) {
                      gpuBufferAllocs.insert({op, {}});
                    } else if (mlir::isa<mlir::func::CallOp>(op)) {
                      // Ignore
                    } else {
                      op->emitError("Unhandled memref producer");
                      return mlir::WalkResult::interrupt();
                    }

                  } else {
                    auto block = alias.getParentBlock();
                    auto blockArgs = block->getArguments();
                    auto it = llvm::find(blockArgs, alias);
                    assert(it != blockArgs.end());
                    auto index = static_cast<unsigned>(it - blockArgs.begin());
                    gpuBufferParams.insert({index, {}});
                  }
                }
              }

              return mlir::WalkResult::advance();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Checks the access type of the OP
    auto getAccessType = [&](mlir::Value memref) {
      AccessType ret;
      for (auto mem : aliases.resolve(memref)) {
        for (auto user : mem.getUsers()) {
          if (mlir::isa<mlir::func::ReturnOp>(user)) {
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
          if (mlir::isa<mlir::func::CallOp>(user)) {
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

    // GetMemrefGlobal Op Case:
    // This is the case where the inputs are globals contants and accessed using
    // memref.get_global op. This code will add the IR for memeory allocation on
    // the device with gpu.alloc and insert a memref.copy from host to device
    mlir::OpBuilder builder(func);
    llvm::SmallVector<mlir::Value> dims;
    llvm::SmallPtrSet<mlir::Operation *, 8> filter;
    auto term = block.getTerminator();
    assert(term);

    for (auto &it : gpuGetMemrefGlobalParams) {
      auto getGlobalOp = mlir::cast<mlir::memref::GetGlobalOp>(it.first);
      it.second = getAccessType(getGlobalOp);
      it.second.hostRead = true;
      it.second.hostWrite = true;

      auto loc = getGlobalOp.getLoc();
      auto access = it.second;
      builder.setInsertionPointAfter(getGlobalOp);
      auto getGlobalOpResult = getGlobalOp.getResult();
      auto memrefType = getGlobalOp.getType().cast<mlir::MemRefType>();
      auto rank = static_cast<unsigned>(memrefType.getRank());
      filter.clear();
      dims.clear();
      for (auto i : llvm::seq(0u, rank)) {
        if (memrefType.isDynamicDim(i)) {
          auto op =
              builder.create<mlir::memref::DimOp>(loc, getGlobalOpResult, i);
          dims.push_back(op);
          filter.insert(op);
        }
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
        gpuAlloc->setAttr(imex::getAllocSharedAttrName(),
                          builder.getUnitAttr());

      if (access.hostWrite && access.deviceRead) {
        auto copy =
            builder.create<mlir::memref::CopyOp>(loc, getGlobalOp, allocResult);
        filter.insert(copy);
      }

      if (allocType != memrefType)
        allocResult =
            builder.create<mlir::memref::CastOp>(loc, memrefType, allocResult);

      getGlobalOpResult.replaceAllUsesExcept(allocResult, filter);

      builder.setInsertionPoint(term);
      if (access.hostRead && access.deviceWrite) {
        builder.create<mlir::memref::CopyOp>(loc, allocResult, getGlobalOp);
      }

      if (useGpuDealloc)
        builder.create<mlir::gpu::DeallocOp>(loc, llvm::None, allocResult);
      else
        builder.create<mlir::memref::DeallocOp>(loc, allocResult);
    }

    // This is the case where a memref.alloc op is directly converted to
    // gpu.alloc
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
        gpuAlloc->setAttr(imex::getAllocSharedAttrName(),
                          builder.getUnitAttr());
    }

    // This is the case where the inputs are passed as arguments to the
    // function. This code will add the IR for memeory allocation on the device
    // with gpu.alloc and insert a memref.copy from host to device
    for (auto it : gpuBufferParams) {
      auto param = block.getArgument(it.first);
      auto access = it.second;
      auto loc = param.getLoc();
      builder.setInsertionPointToStart(&block);
      auto memrefType = param.getType().cast<mlir::MemRefType>();
      auto rank = static_cast<unsigned>(memrefType.getRank());
      filter.clear();
      dims.clear();
      for (auto i : llvm::seq(0u, rank)) {
        if (memrefType.isDynamicDim(i)) {
          auto op = builder.create<mlir::memref::DimOp>(loc, param, i);
          dims.push_back(op);
          filter.insert(op);
        }
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
        gpuAlloc->setAttr(imex::getAllocSharedAttrName(),
                          builder.getUnitAttr());

      if (access.hostWrite && access.deviceRead) {
        auto copy =
            builder.create<mlir::memref::CopyOp>(loc, param, allocResult);
        filter.insert(copy);
      }

      if (allocType != memrefType)
        allocResult =
            builder.create<mlir::memref::CastOp>(loc, memrefType, allocResult);

      param.replaceAllUsesExcept(allocResult, filter);
      builder.setInsertionPoint(term);
      if (access.hostRead && access.deviceWrite)
        builder.create<mlir::memref::CopyOp>(loc, allocResult, param);

      if (useGpuDealloc)
        builder.create<mlir::gpu::DeallocOp>(loc, llvm::None, allocResult);
      else
        builder.create<mlir::memref::DeallocOp>(loc, allocResult);
    }
  }

  Option<bool> useGpuDealloc{
      *this, "use-gpu-dealloc",
      llvm::cl::desc("use gpu.dealloc for gpu allocated memrefs, "
                     "memref.dealloc will be used otherwise"),
      llvm::cl::init(true)};
};

} // namespace imex

std::unique_ptr<mlir::Pass>
imex::createInsertGPUAllocsPass(bool useGpuDealloc) {
  return std::make_unique<InsertGPUAllocs>(useGpuDealloc);
}
