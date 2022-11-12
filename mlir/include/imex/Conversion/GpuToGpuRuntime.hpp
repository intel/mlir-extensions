// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <functional>
#include <memory>

namespace mlir {
class Pass;
namespace spirv {
class TargetEnvAttr;
}
namespace gpu {
class GPUModuleOp;
}
} // namespace mlir

namespace gpu_runtime {

std::unique_ptr<mlir::Pass> createAbiAttrsPass();
std::unique_ptr<mlir::Pass> createSetSPIRVCapabilitiesPass(
    std::function<mlir::spirv::TargetEnvAttr(mlir::gpu::GPUModuleOp)> mapper =
        nullptr);
std::unique_ptr<mlir::Pass> createGPUToSpirvPass();
std::unique_ptr<mlir::Pass> createInsertGPUAllocsPass();
std::unique_ptr<mlir::Pass> createConvertGPUDeallocsPass();
std::unique_ptr<mlir::Pass> createUnstrideMemrefsPass();
std::unique_ptr<mlir::Pass> createSerializeSPIRVPass();
std::unique_ptr<mlir::Pass> createGPUExPass();
std::unique_ptr<mlir::Pass> createParallelLoopGPUMappingPass();

/// Naively tile parallel loops for gpu, using values obtained from
/// `suggest_block_size`.
std::unique_ptr<mlir::Pass> createTileParallelLoopsForGPUPass();

/// For devices without f64 support, truncate all operations to f32.
std::unique_ptr<mlir::Pass> createTruncateF64ForGPUPass();

/// Update scf.parallel loops with reductions to use gpu_runtime.global_reduce.
/// This pass is intended to be run right before scf-to-gpu.
std::unique_ptr<mlir::Pass> createInsertGPUGlobalReducePass();

} // namespace gpu_runtime
