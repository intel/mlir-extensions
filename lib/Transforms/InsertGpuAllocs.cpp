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
/// The pass traverses all the memref (load/store) operations inside the gpu
/// launch op in the IR and checks for its aliases and its defining op. If the
/// defining op is a memref.alloc op it replaces that op in the IR with
/// gpu.alloc op, because all the operations under the gpu.launch op are device
/// side computations and will execute on the device.
///
//===----------------------------------------------------------------------===//

#include <imex/Transforms/Passes.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>

namespace imex {

mlir::StringRef getAllocSharedAttrName() { return "gpu.alloc_shared"; }

struct InsertGPUAllocs
    : public mlir::PassWrapper<InsertGPUAllocs,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
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

    // This lamda function checks the type of memref operation and
    // returns the reference to it.

    auto getMemReadWriteOp = [](mlir::Operation *op)
        -> llvm::Optional<mlir::SmallVector<mlir::Value, 4>> {
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
        return {{load.memref()}};
      } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
        return {{store.memref()}};
      }
      // This case checks if a mlir func call within the gpu.launch has
      // operands which have memref as operands.It just collects them and checks
      // for its origin later in the code
      else if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
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

    // This lamda function checks if the op under consideration
    // within the gpu.launch is a memory operation or no.
    // This pass is only interested in memory operations or operands
    // of mlir call op which are memory ops.
    auto isMemReadWriteOp = [](mlir::Operation *op) -> bool {
      if (auto memInterface =
              mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
        // Only load and Copy op have Read MemoryEffects &
        // Store and TensorOp have Write MemoryEffects
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
    // and add device memory allocation appropriately.
    // It is looking for all the memref producers/consumers used in the device
    // kernels but has its buffers prepared outside.
    if (func.walk([&](mlir::Operation *op) {
              // Limitation is that this pass needs to be be run before the
              // kernel outlining since kernel outlinging with convert the
              // gpu.launch OP to gpu.launch_func.
              if (!op->getParentOfType<mlir::gpu::LaunchOp>())
                return mlir::WalkResult::advance();

              if (!isMemReadWriteOp(op))
                return mlir::WalkResult::advance();

              auto memref = getMemReadWriteOp(op);
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
                    // This is for cases where the memref aliases are just
                    // ViewLikeOps for e.g memref.cast
                    if (mlir::isa<mlir::ViewLikeOpInterface>(op))
                      continue;
                    // Currently the pass only supports memref::AllocOp op and
                    // not its other vairants like memref::AllocaOp,
                    // memref::AllocaScopeOp & AllocaScopeReturnOp.
                    // TODO (nbpatel): Support these ops in the future.
                    if (mlir::isa<mlir::memref::AllocOp>(op)) {
                      gpuBufferAllocs.insert({op, {}});
                    } else {
                      op->emitError("Unhandled memref producer");
                      return mlir::WalkResult::interrupt();
                    }

                  } else {
                    // This is the gpu params case. So if the defining op is not
                    // a memref.alloc or memref.get_global or callOp it assumes
                    // that the inputs are passed in as function args.
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

    // Checks the access type of the OP under consideration.
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

    mlir::OpBuilder builder(func);
    auto &block = funcBody.front();
    auto term = block.getTerminator();
    assert(term);

    // This is the case where a memref.alloc op is directly converted to
    // gpu.alloc
    for (auto it : gpuBufferAllocs) {
      auto alloc = mlir::cast<mlir::memref::AllocOp>(it.first);
      auto access = getAccessType(alloc);
      auto loc = alloc.getLoc();
      builder.setInsertionPoint(alloc);
      auto gpuAlloc = builder.create<mlir::gpu::AllocOp>(
          loc, alloc.getType(), /*asyncToken*/ nullptr,
          /*asyncDependencies*/ llvm::None, alloc.dynamicSizes(),
          alloc.symbolOperands());
      auto allocResult = gpuAlloc.getResult(0);
      alloc->replaceAllUsesWith(gpuAlloc);
      alloc.erase();
      if (access.hostRead || access.hostWrite)
        gpuAlloc->setAttr(imex::getAllocSharedAttrName(),
                          builder.getUnitAttr());

      builder.setInsertionPoint(term);

      builder.create<mlir::gpu::DeallocOp>(loc, llvm::None, allocResult);
    }

    auto add_gpu_alloc = [](mlir::OpBuilder builder, mlir::Value op,
                            AccessType access, auto term) {
      llvm::SmallVector<mlir::Value> dims;
      llvm::SmallPtrSet<mlir::Operation *, 8> filter;
      auto memrefType = op.getType().cast<mlir::MemRefType>();
      auto loc = op.getLoc();
      auto rank = static_cast<unsigned>(memrefType.getRank());
      filter.clear();
      dims.clear();

      // This code handles dynamic dims with known rank.
      for (auto i : llvm::seq(0u, rank)) {
        if (memrefType.isDynamicDim(i)) {
          auto dim_op = builder.create<mlir::memref::DimOp>(loc, op, i);
          dims.push_back(dim_op);
          filter.insert(dim_op);
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
        auto copy = builder.create<mlir::memref::CopyOp>(loc, op, allocResult);
        filter.insert(copy);
      }

      if (allocType != memrefType)
        allocResult =
            builder.create<mlir::memref::CastOp>(loc, memrefType, allocResult);

      op.replaceAllUsesExcept(allocResult, filter);
      builder.setInsertionPoint(term);
      if (access.hostRead && access.deviceWrite) {
        builder.create<mlir::memref::CopyOp>(loc, allocResult, op);
      }

      builder.create<mlir::gpu::DeallocOp>(loc, llvm::None, allocResult);
    };

    // GetMemrefGlobal Op Case:
    // This is the case where the inputs are globals contants and accessed using
    // memref.get_global op. This code will add the IR for memory allocation on
    // the device with gpu.alloc and insert a memref.copy from host to device.
    for (auto &it : gpuGetMemrefGlobalParams) {
      auto getGlobalOp = mlir::cast<mlir::memref::GetGlobalOp>(it.first);
      auto access = getAccessType(getGlobalOp);
      access.hostRead = true;
      access.hostWrite = true;
      builder.setInsertionPointAfter(getGlobalOp);
      add_gpu_alloc(builder, getGlobalOp, access, term);
    }

    // This is the case where the inputs are passed as arguments to the
    // function. This code will add the IR for memory allocation on the device
    // with gpu.alloc and insert a memref.copy from host to device
    for (auto it : gpuBufferParams) {
      auto param = block.getArgument(it.first);
      auto access = getAccessType(param);
      access.hostRead = true;
      access.hostWrite = true;
      builder.setInsertionPointToStart(&block);
      add_gpu_alloc(builder, param, access, term);
    }
  }
};

} // namespace imex

std::unique_ptr<mlir::Pass> imex::createInsertGPUAllocsPass() {
  return std::make_unique<InsertGPUAllocs>();
}
