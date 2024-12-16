//===- InsertGPUAllocs.cpp - InsertGPUAllocs Pass  -------*- C++ -*-===//
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

#include "llvm/Support/Threading.h"
#include <imex/Transforms/Passes.h>

#include <imex/Dialect/Region/RegionUtils.h>
#include <imex/Dialect/XeTile/IR/XeTileOps.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/XeGPU/IR/XeGPU.h>
#include <mlir/Pass/Pass.h>

#include <optional>

namespace imex {
#define GEN_PASS_DEF_INSERTGPUALLOCS
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

namespace {
class InsertGPUAllocsPass final
    : public imex::impl::InsertGPUAllocsBase<InsertGPUAllocsPass> {

public:
  explicit InsertGPUAllocsPass() : m_clientAPI("vulkan") {}
  explicit InsertGPUAllocsPass(const mlir::StringRef &clientAPI)
      : m_clientAPI(clientAPI) {}
  explicit InsertGPUAllocsPass(const imex::InsertGPUAllocsOptions &options)
      : InsertGPUAllocsBase<InsertGPUAllocsPass>(options) {
    if (clientAPI == "opencl") {
      m_clientAPI = "opencl";
    }
  }

  mlir::LogicalResult
  initializeOptions(mlir::StringRef options,
                    mlir::function_ref<mlir::LogicalResult(const llvm::Twine &)>
                        errorHandler) override {
    if (mlir::failed(Pass::initializeOptions(options, errorHandler)))
      return mlir::failure();

    if (clientAPI == "opencl") {
      m_clientAPI = "opencl";
    }

    if (clientAPI != "vulkan" && clientAPI != "opencl")
      return errorHandler(llvm::Twine("Invalid clientAPI: ") + clientAPI);

    if (clientAPI.getValue() != "opencl" && inRegions.getValue())
      return mlir::failure();

    return mlir::success();
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

    mlir::OpBuilder builder(func);

    if (inRegions.getValue()) {
      // collecting alloc ops in GPU regions
      ::mlir::SmallVector<::mlir::memref::AllocOp> allocOpsInGpuRegion;
      ::mlir::SmallVector<::mlir::memref::DeallocOp> deallocOpsInGpuRegion;

      // Traverse ops and identify memref.alloc ops which are in GPU region
      (void)func.walk([&](mlir::Operation *op) {
        // identify and store memref.alloc ops which are inside a GPU-region
        if (::imex::region::isInGpuRegion(op)) {
          if (auto tyOp = ::mlir::dyn_cast<::mlir::memref::AllocOp>(op)) {
            allocOpsInGpuRegion.emplace_back(tyOp);
          } else if (auto tyOp =
                         ::mlir::dyn_cast<::mlir::memref::DeallocOp>(op)) {
            deallocOpsInGpuRegion.emplace_back(tyOp);
          }
        }
      });

      // Now rudely replace allocs with gpu allocs
      for (auto alloc : allocOpsInGpuRegion) {
        builder.setInsertionPoint(alloc);
        auto allocResult = builder.create<::mlir::gpu::AllocOp>(
            alloc.getLoc(), alloc.getType(), /*asyncToken*/ nullptr,
            /*asyncDependencies*/ std::nullopt, alloc.getDynamicSizes(),
            alloc.getSymbolOperands(), true);
        alloc.replaceAllUsesWith(allocResult);
        alloc.erase();
      }

      // finally rudely handle deallocs
      for (auto dealloc : deallocOpsInGpuRegion) {
        builder.setInsertionPoint(dealloc);
        (void)builder.create<::mlir::gpu::DeallocOp>(
            dealloc.getLoc(), std::nullopt /*async*/, dealloc.getMemref());
        dealloc.erase();
      }

      // Done, it can be as simple as that!
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
    llvm::SmallMapVector<mlir::Operation *, AccessType, 8> callOpReturnedBuffer;
    auto &aliases = getAnalysis<mlir::BufferViewFlowAnalysis>();

    // This lamda function checks the type of memref operation and
    // returns the reference to it.

    auto getMemReadWriteOp = [](mlir::Operation *op)
        -> std::optional<mlir::SmallVector<mlir::Value, 4>> {
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
        return {{load.getMemref()}};
      } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
        return {{store.getMemref()}};
      }
      // This case checks if a mlir func call within the gpu.launch has
      // operands which have memref as operands.It just collects them and checks
      // for its origin later in the code
      else if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
        mlir::SmallVector<mlir::Value, 4> ret;
        for (mlir::Value arg : call.getOperands()) {
          if (mlir::isa<mlir::MemRefType>(arg.getType()))
            ret.emplace_back(arg);
        }
        return std::move(ret);
      } else if (auto init_tile =
                     mlir::dyn_cast<imex::xetile::InitTileOp>(op)) {
        return {{init_tile.getSource()}};
      } else if (auto init_xedesc =
                     mlir::dyn_cast<mlir::xegpu::CreateNdDescOp>(op)) {
        return {{init_xedesc.getSource()}};
      } else {
        op->emitError("Uhhandled mem op in gpu region");
        return std::nullopt;
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
        for (const auto &arg : call.getOperands()) {
          if (mlir::isa<mlir::MemRefType>(arg.getType()))
            return true;
        }
      }
      if (auto init_tile = mlir::dyn_cast<imex::xetile::InitTileOp>(op)) {
        // Only handle the case where the tile source is a memref
        return init_tile.isSourceMemRef();
      }
      if (auto init_xedesc = mlir::dyn_cast<mlir::xegpu::CreateNdDescOp>(op)) {
        return true;
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

              for (mlir::Value mem : *memref) {
                while (mlir::ViewLikeOpInterface parentView =
                           mem.getDefiningOp<mlir::ViewLikeOpInterface>())
                  mem = parentView.getViewSource();

                for (mlir::Value alias : aliases.resolve(mem)) {
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
                    } else if (mlir::isa<mlir::func::CallOp>(op)) {
                      callOpReturnedBuffer.insert({op, {}});
                      continue;
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

    // Checks if the memref type has the gpu address space. For this case we
    // don't need to do anything since the memref is already in the device
    // address space.
    auto isGpuAddrSpace = [&](mlir::Value memref) {
      if (auto type = mlir::dyn_cast<mlir::MemRefType>(memref.getType())) {
        return mlir::isa_and_nonnull<mlir::gpu::AddressSpaceAttr>(
            type.getMemorySpace());
      }
      return false;
    };

    // walk over the users and find xegpu.load/store ops
    std::function<void(mlir::Operation *, bool, AccessType &)>
        findXeGPULoadStore;
    findXeGPULoadStore = [&](mlir::Operation *use, bool onDevice,
                             AccessType &ret) {
      if (auto tile_update =
              mlir::dyn_cast<mlir::xegpu::UpdateNdOffsetOp>(use)) {
        auto res = tile_update->getResult(0);
        for (auto u : res.getUsers()) {
          findXeGPULoadStore(u, onDevice, ret);
        }
      }
      if (auto tile_for = mlir::dyn_cast<::mlir::scf::ForOp>(use)) {
        for (size_t idx = 0; idx < tile_for.getInits().size(); idx++) {
          auto a = tile_for.getRegionIterArg(idx);
          for (auto u : a.getUsers()) {
            findXeGPULoadStore(u, onDevice, ret);
          }
        }
      }
      if (auto tile_load = mlir::dyn_cast<mlir::xegpu::LoadNdOp>(use)) {
        (onDevice ? ret.deviceRead : ret.hostRead) = true;
      } else if (auto tile_prefetch =
                     mlir::dyn_cast<mlir::xegpu::PrefetchNdOp>(use)) {
        (onDevice ? ret.deviceRead : ret.hostRead) = true;
      } else if (auto tile_store =
                     mlir::dyn_cast<mlir::xegpu::StoreNdOp>(use)) {
        (onDevice ? ret.deviceWrite : ret.hostWrite) = true;
      }
    };

    // Checks the access type of the OP under consideration.
    auto getAccessType = [&](mlir::Value memref) {
      AccessType ret;
      for (const auto &mem : aliases.resolve(memref)) {
        for (auto user : mem.getUsers()) {
          if (auto init_tile = mlir::dyn_cast<imex::xetile::InitTileOp>(user)) {
            bool onDevice = user->getParentOfType<mlir::gpu::LaunchOp>();
            auto res = init_tile->getResult(0);
            for (auto use : res.getUsers()) {
              if (auto tile_for = mlir::dyn_cast<::mlir::scf::ForOp>(use)) {
                unsigned int idx = 0;
                for (auto i : tile_for.getInits()) {
                  if (i.getDefiningOp() == user) {
                    auto a = tile_for.getRegionIterArg(idx);
                    for (auto u : a.getUsers()) {
                      if (auto tile_load =
                              mlir::dyn_cast<imex::xetile::LoadTileOp>(u)) {
                        (onDevice ? ret.deviceRead : ret.hostRead) = true;
                      } else if (auto tile_store =
                                     mlir::dyn_cast<imex::xetile::StoreTileOp>(
                                         u)) {
                        (onDevice ? ret.deviceWrite : ret.hostWrite) = true;
                      }
                    }
                  }
                  idx++;
                }
              }
              if (auto tile_load =
                      mlir::dyn_cast<imex::xetile::LoadTileOp>(use)) {
                (onDevice ? ret.deviceRead : ret.hostRead) = true;
              } else if (auto tile_store =
                             mlir::dyn_cast<imex::xetile::StoreTileOp>(use)) {
                (onDevice ? ret.deviceWrite : ret.hostWrite) = true;
              }
            }
            continue;
          }

          if (auto init_xedesc =
                  mlir::dyn_cast<mlir::xegpu::CreateNdDescOp>(user)) {
            bool onDevice = user->getParentOfType<mlir::gpu::LaunchOp>();
            auto res = init_xedesc->getResult(0);
            for (auto use : res.getUsers()) {
              findXeGPULoadStore(use, onDevice, ret);
            }
            continue;
          }

          if (mlir::isa<mlir::func::ReturnOp>(user)) {
            ret.hostRead = true;
            ret.hostWrite = true;
            continue;
          }

          if (auto copy = mlir::dyn_cast<mlir::memref::CopyOp>(user)) {
            if (copy.getSource() == mem)
              ret.hostRead = true;

            if (copy.getTarget() == mem)
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

    auto &block = funcBody.front();
    auto term = block.getTerminator();
    assert(term);

    // This is the case where a memref.alloc op is directly converted to
    // gpu.alloc
    if (m_clientAPI == "opencl") {
      for (const auto &it : gpuBufferAllocs) {
        auto alloc = mlir::cast<mlir::memref::AllocOp>(it.first);
        auto access = getAccessType(alloc);
        auto loc = alloc.getLoc();
        builder.setInsertionPoint(alloc);
        bool hostShared = access.hostRead || access.hostWrite;
        auto gpuAlloc = builder.create<mlir::gpu::AllocOp>(
            loc, alloc.getType(), /*asyncToken*/ nullptr,
            /*asyncDependencies*/ std::nullopt, alloc.getDynamicSizes(),
            alloc.getSymbolOperands(), hostShared);
        auto allocResult = gpuAlloc.getResult(0);
        builder.setInsertionPoint(term);
        for (mlir::OpOperand &use : alloc.getResult().getUses()) {
          if (use.getOwner() == term) {
            auto newAlloc = builder.create<mlir::memref::AllocOp>(
                loc, alloc.getType(), alloc.getDynamicSizes(),
                alloc.getSymbolOperands());
            builder.create<mlir::memref::CopyOp>(loc, allocResult,
                                                 newAlloc.getResult());
            use.set(newAlloc.getResult());
          }
        }

        // remove 'memref.dealloc' (it's later replaced with gpu.dealloc)
        auto memory = alloc->getResult(0);
        for (auto u : memory.getUsers()) {
          if (auto dealloc = mlir::dyn_cast<mlir::memref::DeallocOp>(u)) {
            dealloc.erase();
          }
        }

        alloc.replaceAllUsesWith(allocResult);
        builder.create<mlir::gpu::DeallocOp>(loc, std::nullopt, allocResult);
        alloc.erase();
      }
    }

    auto add_gpu_alloc = [this](mlir::OpBuilder builder, mlir::Value op,
                                AccessType access, auto term) {
      llvm::SmallVector<mlir::Value> dims;
      llvm::SmallPtrSet<mlir::Operation *, 8> filter;
      auto memrefType = mlir::cast<mlir::MemRefType>(op.getType());
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
      if (m_clientAPI == "opencl") {
        bool hostShared = access.hostRead || access.hostWrite;
        auto gpuAlloc = builder.create<mlir::gpu::AllocOp>(
            loc, allocType, /*asyncToken*/ nullptr,
            /*asyncDependencies*/ std::nullopt, dims,
            /*symbolOperands*/ std::nullopt, hostShared);
        auto allocResult = gpuAlloc.getResult(0);
        if (access.hostWrite && access.deviceRead) {
          auto copy =
              builder.create<mlir::memref::CopyOp>(loc, op, allocResult);
          filter.insert(copy);
        }

        if (allocType != memrefType) {
          mlir::Value castedAllocResult = builder.create<mlir::memref::CastOp>(
              loc, memrefType, allocResult);

          op.replaceAllUsesExcept(castedAllocResult, filter);
          builder.setInsertionPoint(term);
          if (access.hostRead && access.deviceWrite) {
            builder.create<mlir::memref::CopyOp>(loc, castedAllocResult, op);
          }
          builder.create<mlir::gpu::DeallocOp>(loc, std::nullopt,
                                               castedAllocResult);
        } else {
          op.replaceAllUsesExcept(allocResult, filter);
          builder.setInsertionPoint(term);
          if (access.hostRead && access.deviceWrite) {
            builder.create<mlir::memref::CopyOp>(loc, allocResult, op);
          }
          builder.create<mlir::gpu::DeallocOp>(loc, std::nullopt, allocResult);
        }
      } else if (m_clientAPI == "vulkan") {
        auto gpuAlloc =
            builder.create<mlir::memref::AllocOp>(loc, allocType, dims);
        auto allocResult = gpuAlloc.getResult();
        if (access.hostWrite && access.deviceRead) {
          auto copy =
              builder.create<mlir::memref::CopyOp>(loc, op, allocResult);
          filter.insert(copy);
        }

        if (allocType != memrefType) {
          mlir::Value castedAllocResult = builder.create<mlir::memref::CastOp>(
              loc, memrefType, allocResult);

          op.replaceAllUsesExcept(castedAllocResult, filter);
          builder.setInsertionPoint(term);
          if (access.hostRead && access.deviceWrite) {
            builder.create<mlir::memref::CopyOp>(loc, castedAllocResult, op);
          }
        } else {
          op.replaceAllUsesExcept(allocResult, filter);
          builder.setInsertionPoint(term);
          if (access.hostRead && access.deviceWrite) {
            builder.create<mlir::memref::CopyOp>(loc, allocResult, op);
          }
        }
      }
    };

    // GetMemrefGlobal Op Case:
    // This is the case where the inputs are globals contants and accessed using
    // memref.get_global op. This code will add the IR for memory allocation on
    // the device with gpu.alloc and insert a memref.copy from host to device.
    for (auto &it : gpuGetMemrefGlobalParams) {
      auto getGlobalOp = mlir::cast<mlir::memref::GetGlobalOp>(it.first);
      if (isGpuAddrSpace(getGlobalOp))
        continue;
      auto access = getAccessType(getGlobalOp);
      access.hostRead = true;
      access.hostWrite = true;
      builder.setInsertionPointAfter(getGlobalOp);
      add_gpu_alloc(builder, getGlobalOp, access, term);
    }

    // This is the case where the inputs are passed as arguments to the
    // function. This code will add the IR for memory allocation on the device
    // with gpu.alloc and insert a memref.copy from host to device
    if (!isUsmArgs.getValue()) {
      for (const auto &it : gpuBufferParams) {
        auto param = block.getArgument(it.first);
        if (isGpuAddrSpace(param))
          continue;
        auto access = getAccessType(param);
        access.hostRead = true;
        access.hostWrite = true;
        builder.setInsertionPointToStart(&block);
        add_gpu_alloc(builder, param, access, term);
      }
    }

    // CallOp Case: This is the case where the memref producer is coming
    // from a callOp. This code will add the IR for memory allocation on
    // the device with gpu.alloc and insert a memref.copy from the result
    // of that call op to device.
    for (auto &it : callOpReturnedBuffer) {
      auto op = mlir::cast<mlir::func::CallOp>(it.first);
      mlir::Value callOp = op.getResult(0);
      if (isGpuAddrSpace(callOp))
        continue;
      AccessType access;
      access.deviceRead = true;
      access.deviceWrite = false;
      access.hostRead = true;
      access.hostWrite = true;
      builder.setInsertionPointAfter(op);
      add_gpu_alloc(builder, callOp, access, term);
    }
  }

private:
  mlir::StringRef m_clientAPI;
};

} // namespace

namespace imex {
std::unique_ptr<mlir::Pass> createInsertGPUAllocsPass(const char *clientAPI) {
  return std::make_unique<InsertGPUAllocsPass>(clientAPI);
}
std::unique_ptr<mlir::Pass>
createInsertGPUAllocsPass(const InsertGPUAllocsOptions &option) {
  return std::make_unique<InsertGPUAllocsPass>(option);
}
} // namespace imex
