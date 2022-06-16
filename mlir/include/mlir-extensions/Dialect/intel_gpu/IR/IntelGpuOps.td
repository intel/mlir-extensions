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

#ifndef INTELGPU_OPS
#define INTELGPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/GPU/GPUBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/EnumAttr.td"

def IntelGpu_Dialect : Dialect {
  let name = "intel_gpu";
  let description = [{
      IntelGpu Dialect for representing Intel GPU specific ops for using SYCL runtime
    }];
  let cppNamespace = "intel_gpu";
  let hasConstantMaterializer = 1;
}

class IntelGpu_Op<string mnemonic, list<Trait> traits = []> :
      Op<IntelGpu_Dialect, mnemonic, traits>;

def IntelGpu_OpaqueType
    : DialectType<IntelGpu_Dialect, CPred<"$_self.isa<::intel_gpu::OpaqueType>()">,
                  "opaque_type">,
      BuildableType<"$_builder.getType<::intel_gpu::OpaqueType>()"> {}

def IntelGpu_StorageClass : I32EnumAttr<"StorageClass",
    "Kernel barrier and fence flags",
    [
      I32EnumAttrCase<"local", 1>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::intel_gpu";
}
def IntelGpu_StorageClassAttr : EnumAttr<IntelGpu_Dialect, IntelGpu_StorageClass, "storage_class">;


def GetDeviceOp : IntelGpu_Op<"get_device", [NoSideEffect]> {
  let description = [{
    Returns the gpu device at the given index.

    Example:
      %ordinal = constant.i32 0
      %device = device.get INTELGPU, %ordinal
  }];

  // Start with some default ordinal and device type
 // let arguments = (ins IntelGpu_OpaqueType:$platform, I32:$ordinal);
  let results = (outs IntelGpu_OpaqueType : $device);
  let builders = [OpBuilder<(ins)>];
}

def CreateContextOp : IntelGpu_Op<"create_context"> {
  let description = [{
    Returns a gpu context for the given device.

    Example:
      %ctx = context.create %device
  }];

  let arguments = (ins IntelGpu_OpaqueType : $device);
  let results = (outs IntelGpu_OpaqueType : $context);
  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $device)>];
}

def GetStreamOp : IntelGpu_Op<"get_stream"> {
  let results = (outs IntelGpu_OpaqueType : $stream);
  let hasCanonicalizer = 1;
  let builders = [OpBuilder<(ins)>];
}

def DestroyStreamOp : IntelGpu_Op<"stream.destroy"> {
  let arguments = (ins IntelGpu_OpaqueType : $stream);
}


#endif //INTELGPU_OPS
