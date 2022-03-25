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

#ifndef PLIER_UTIL_OPS
#define PLIER_UTIL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/GPU/GPUBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def PlierUtil_Dialect : Dialect {
  let name = "plier_util";
  let cppNamespace = "plier";

  let hasCanonicalizer = 1;
  let hasConstantMaterializer = 1;
}

def PlierUtil_OpaqueType
    : DialectType<PlierUtil_Dialect, CPred<"$_self.isa<::plier::OpaqueType>()">,
                  "opaque_type">,
      BuildableType<"$_builder.getType<::plier::OpaqueType>()"> {}

class PlierUtil_Op<string mnemonic, list<Trait> traits = []>
    : Op<PlierUtil_Dialect, mnemonic, traits>;

def EnforceShapeOp : PlierUtil_Op<"enforce_shape"> {
  let arguments = (ins AnyRankedTensor : $value, Variadic<Index> : $sizes);

  let results = (outs AnyRankedTensor : $result);

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $value, "::mlir::ValueRange"
                             : $shape)>];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def ParallelOp : PlierUtil_Op<"parallel", [
  AttrSizedOperandSegments, DeclareOpInterfaceMethods<LoopLikeOpInterface>,
  SingleBlockImplicitTerminator<"plier::YieldOp">, RecursiveSideEffects
]> {

  let arguments = (ins Variadic<Index>:$lowerBounds,
                         Variadic<Index>:$upperBounds,
                         Variadic<Index>:$steps);
  let regions = (region SizedRegion<1> : $region);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(
      ins "::mlir::ValueRange"
      : $lowerBounds, "::mlir::ValueRange"
      : $upperBounds, "::mlir::ValueRange"
      : $steps,
        CArg<"::mlir::function_ref<void(::mlir::OpBuilder &, ::mlir::Location, "
             "::mlir::ValueRange, ::mlir::ValueRange, ::mlir::Value)>",
             "nullptr">)>];

    let extraClassDeclaration = [{
        unsigned getNumLoops() { return steps().size(); }
    }];
}

def UndefOp : PlierUtil_Op<"undef", [NoSideEffect]> {
  let results = (outs AnyType);
}

def YieldOp : PlierUtil_Op<"yield", [
  NoSideEffect, ReturnLike, Terminator, ParentOneOf<["ParallelOp"]>
]> {
  let arguments = (ins Variadic<AnyType> : $results);
  let builders = [OpBuilder<(ins), [{/* nothing to do */}]>];
  // Override default verifier (defined in SCF_Op), no custom verification
  // needed.
  let hasVerifier = 0;
}

def RetainOp : PlierUtil_Op<"retain", [ViewLikeOpInterface]> {
  let arguments = (ins AnyMemRef : $source);

  let results = (outs AnyMemRef);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return source(); }
  }];
}

def ChangeLayoutOp : PlierUtil_Op<"change_layout", [ViewLikeOpInterface, NoSideEffect]> {
  let arguments = (ins AnyMemRef : $source);

  let results = (outs AnyMemRef);
  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return source(); }
  }];
}

def SignCastOp : PlierUtil_Op<"sign_cast", [NoSideEffect]> {
  let arguments = (ins AnyType : $value);

  let results = (outs AnyType);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def ExtractMemrefMetadataOp
    : PlierUtil_Op<"extract_memref_metadata", [NoSideEffect]> {
  let arguments = (ins AnyMemRef : $source, IndexAttr : $dimIndex);

  let results = (outs Index : $result);
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "::mlir::Value" : $src,
                 "int64_t" : $dim)>,
    OpBuilder<(ins "::mlir::Value" : $src)>
  ];
}

def CreateGpuStreamOp : PlierUtil_Op<"create_gpu_stream", [NoSideEffect]> {
  let results = (outs PlierUtil_OpaqueType : $result);

  let hasCanonicalizer = 1;

  let builders = [OpBuilder<(ins)>];
}

def DestroyGpuStreamOp : PlierUtil_Op<"destroy_gpu_stream"> {
  let arguments = (ins PlierUtil_OpaqueType : $source);
}

def LoadGpuModuleOp : PlierUtil_Op<"load_gpu_module", [NoSideEffect]> {
  let arguments = (ins PlierUtil_OpaqueType : $stream, SymbolRefAttr : $module);
  let results = (outs PlierUtil_OpaqueType : $result);

  let hasCanonicalizer = 1;

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $stream, "::mlir::gpu::GPUModuleOp"
                             : $module)>];
}

def DestroyGpuModuleOp : PlierUtil_Op<"destroy_gpu_module"> {
  let arguments = (ins PlierUtil_OpaqueType : $source);
}

def GetGpuKernelOp : PlierUtil_Op<"get_gpu_kernel", [NoSideEffect]> {
  let arguments = (ins PlierUtil_OpaqueType : $module, SymbolRefAttr : $kernel);
  let results = (outs PlierUtil_OpaqueType : $result);

  let hasCanonicalizer = 1;

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $module, "::mlir::gpu::GPUFuncOp"
                             : $kernel)>];
}

def LaunchGpuKernelOp
    : PlierUtil_Op<"launch_gpu_kernel",
               [GPU_AsyncOpInterface, AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                     PlierUtil_OpaqueType:$stream,
                     PlierUtil_OpaqueType:$kernel,
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

def DestroyGpuKernelOp : PlierUtil_Op<"destroy_gpu_kernel"> {
  let arguments = (ins PlierUtil_OpaqueType : $source);
}

def GPUAllocOp
    : PlierUtil_Op<"gpu_alloc", [GPU_AsyncOpInterface, AttrSizedOperandSegments]> {

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       PlierUtil_OpaqueType:$stream,
                       Variadic<Index>:$dynamicSizes,
                       Variadic<Index>:$symbolOperands);
  let results = (outs Res<AnyMemRef, "", [MemAlloc]>:$memref,
                      Optional<GPU_AsyncToken>:$asyncToken);

  let extraClassDeclaration = [{
    ::mlir::MemRefType getType() { return memref().getType().cast<::mlir::MemRefType>(); }
  }];
}

def GPUSuggestBlockSizeOp : PlierUtil_Op<"suggest_block_size",
                                     [AttrSizedOperandSegments, NoSideEffect]> {
  let arguments = (ins Optional<PlierUtil_OpaqueType>:$stream,
                       Optional<PlierUtil_OpaqueType>:$kernel,
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

def PseudoCopyOp : PlierUtil_Op<"pseudo_copy", [ViewLikeOpInterface]> {
  let arguments = (ins AnyType : $source);

  let results = (outs AnyType);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return source(); }
  }];
}

def ForceCopyOp : PlierUtil_Op<"force_copy", [ViewLikeOpInterface]> {
  let arguments = (ins AnyRankedTensor : $source);

  let results = (outs AnyRankedTensor);

  let builders = [OpBuilder<(ins "::mlir::Value" : $value)>];

  let extraClassDeclaration = [{
      ::mlir::Value getViewSource() { return source(); }
  }];
}

def TakeContextOp : PlierUtil_Op<"take_context"> {
  let arguments = (ins OptionalAttr<SymbolRefAttr>:$initFunc,
                       OptionalAttr<SymbolRefAttr>:$releaseFunc);

  let results = (outs PlierUtil_OpaqueType:$context,
                      Variadic<AnyType>:$results);

    let builders = [
      OpBuilder<(ins "::mlir::SymbolRefAttr":$initFunc,
                     "::mlir::SymbolRefAttr":$releaseFunc,
                     "::mlir::TypeRange":$resultTypes)>
   ];
}

def ReleaseContextOp : PlierUtil_Op<"release_context"> {
  let arguments = (ins PlierUtil_OpaqueType:$context);
}

#endif // PLIER_UTIL_OPS
