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



#include <imex/Transforms/Transforms.hpp>

namespace imex {

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
                    if (mlir::isa<mlir::memref::AllocOp,
                                  mlir::memref::GetGlobalOp>(op)) {
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
      if (alloc)
        it.second = getAccessType(alloc);
      else {
        auto memrefGlobal = mlir::cast<mlir::memref::GetGlobalOp>(it.first);
        it.second = getAccessType(memrefGlobal);
      }
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
      // if (access.hostRead || access.hostWrite)
      //  gpuAlloc->setAttr(gpu_runtime::getAllocSharedAttrName(),
      //                    builder.getUnitAttr());
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

      // if (access.hostRead || access.hostWrite)
      // gpuAlloc->setAttr(gpu_runtime::getAllocSharedAttrName(),
      //                   builder.getUnitAttr());

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