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

#ifndef GPURUNTIME_OPS
#define GPURUNTIME_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/GPU/GPUBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/EnumAttr.td"

def GpuRuntime_Dialect : Dialect {
  let name = "gpu_runtime";
  let description = [{
      GpuRuntime Dialect for representing Gpu runtime ops for level zero runtime
    }];
  let cppNamespace = "gpu_runtime";
  let hasConstantMaterializer = 1;
}

class GpuRuntime_Op<string mnemonic, list<Trait> traits = []> :
      Op<GpuRuntime_Dialect, mnemonic, traits>;

def GpuRuntime_OpaqueType
    : DialectType<GpuRuntime_Dialect, CPred<"$_self.isa<::gpu_runtime::OpaqueType>()">,
                  "opaque_type">,
      BuildableType<"$_builder.getType<::gpu_runtime::OpaqueType>()"> {}

def GpuRuntime_FenceFlags : I32EnumAttr<"FenceFlags",
    "Kernel barrier and fence flags",
    [
      I32EnumAttrCase<"local", 1>,
      I32EnumAttrCase<"global", 2>
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::gpu_runtime";
}
def GpuRuntime_FenceFlagsAttr : EnumAttr<GpuRuntime_Dialect, GpuRuntime_FenceFlags, "fence_flags">;

def GpuRuntime_StorageClass : I32EnumAttr<"StorageClass",
    "Kernel barrier and fence flags",
    [
      I32EnumAttrCase<"local", 1>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::gpu_runtime";
}
def GpuRuntime_StorageClassAttr : EnumAttr<GpuRuntime_Dialect, GpuRuntime_StorageClass, "storage_class">;

def CreateGpuStreamOp : GpuRuntime_Op<"create_gpu_stream", [NoSideEffect]> {
  let results = (outs GpuRuntime_OpaqueType : $result);

  let hasCanonicalizer = 1;

  let builders = [OpBuilder<(ins)>];
}

def DestroyGpuStreamOp : GpuRuntime_Op<"destroy_gpu_stream"> {
  let arguments = (ins GpuRuntime_OpaqueType : $source);
}

def LoadGpuModuleOp : GpuRuntime_Op<"load_gpu_module", [NoSideEffect]> {
  let arguments = (ins GpuRuntime_OpaqueType : $stream, SymbolRefAttr : $module);
  let results = (outs GpuRuntime_OpaqueType : $result);

  let hasCanonicalizer = 1;

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $stream, "::mlir::gpu::GPUModuleOp"
                             : $module)>];
}

def DestroyGpuModuleOp : GpuRuntime_Op<"destroy_gpu_module"> {
  let arguments = (ins GpuRuntime_OpaqueType : $source);
}

def GetGpuKernelOp : GpuRuntime_Op<"get_gpu_kernel", [NoSideEffect]> {
  let arguments = (ins GpuRuntime_OpaqueType : $module, SymbolRefAttr : $kernel);
  let results = (outs GpuRuntime_OpaqueType : $result);

  let hasCanonicalizer = 1;

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $module, "::mlir::gpu::GPUFuncOp"
                             : $kernel)>];
}

def LaunchGpuKernelOp
    : GpuRuntime_Op<"launch_gpu_kernel",
               [GPU_AsyncOpInterface, AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                     GpuRuntime_OpaqueType:$stream,
                     GpuRuntime_OpaqueType:$kernel,
                     Index:$gridSizeX, Index:$gridSizeY, Index:$gridSizeZ,
                     Index:$blockSizeX, Index:$blockSizeY, Index:$blockSizeZ,
                     Variadic<AnyType>:$operands);
  let results = (outs Optional<GPU_AsyncToken> : $asyncToken);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "::mlir::Value" : $stream,
                                 "::mlir::Value" : $kernel,
                                 "::mlir::gpu::KernelDim3" : $gridSize,
                                 "::mlir::gpu::KernelDim3" : $blockSize,
                                 "::mlir::ValueRange" : $kernelOperands)>];
}

def DestroyGpuKernelOp : GpuRuntime_Op<"destroy_gpu_kernel"> {
  let arguments = (ins GpuRuntime_OpaqueType : $source);
}

def GPUAllocOp
    : GpuRuntime_Op<"gpu_alloc", [GPU_AsyncOpInterface, AttrSizedOperandSegments]> {

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       GpuRuntime_OpaqueType:$stream,
                       Variadic<Index>:$dynamicSizes,
                       Variadic<Index>:$symbolOperands);
  let results = (outs Res<AnyMemRef, "", [MemAlloc]>:$memref,
                      Optional<GPU_AsyncToken>:$asyncToken);

  let extraClassDeclaration = [{
    ::mlir::MemRefType getType() { return memref().getType().cast<::mlir::MemRefType>(); }
  }];
}

def GPUDeallocOp : GpuRuntime_Op<"gpu_dealloc"> {
  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                  Res<AnyMemRef, "", [MemAlloc]>:$memref,
                  GpuRuntime_OpaqueType : $stream);
}

def GPUSuggestBlockSizeOp : GpuRuntime_Op<"suggest_block_size",
                                     [AttrSizedOperandSegments, NoSideEffect]> {
  let arguments = (ins Optional<GpuRuntime_OpaqueType>:$stream,
                       Optional<GpuRuntime_OpaqueType>:$kernel,
                       OptionalAttr<SymbolRefAttr>:$kernelRef,
                       Variadic<Index>:$gridSize);

  let results = (outs Variadic<Index>);

  let builders = [OpBuilder<(ins "::llvm::Optional<::mlir::Value>" : $stream,
                                 "::mlir::OpFoldResult" : $kernel,
                                 "::mlir::ValueRange" : $gridSize)>];

  let extraClassDeclaration = [{
      /// The name of the kernel's containing module.
      ::mlir::StringAttr getKernelModuleName();

      /// The name of the kernel.
      ::mlir::StringAttr getKernelName();
  }];
}

def GPUBarrierOp : GpuRuntime_Op<"barrier"> {
  let summary = "Synchronizes all work items of a workgroup.";

  let arguments = (ins GpuRuntime_FenceFlagsAttr:$flags);

  let assemblyFormat = "$flags attr-dict";
}

def GPUMemFenceOp : GpuRuntime_Op<"mem_fence"> {
  let summary = "Orders loads and stores of a work-item executing a kernel.";

  let arguments = (ins GpuRuntime_FenceFlagsAttr:$flags);

  let assemblyFormat = "$flags attr-dict";
}

#endif // GPURUNTIME_OPS

