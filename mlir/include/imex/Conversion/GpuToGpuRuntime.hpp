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

} // namespace gpu_runtime
