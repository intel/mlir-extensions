# SetSPIRVCapabilities Pass


SPIR-V aims to support multiple execution environments. These execution environments affect the availability of certain SPIR-V features. SPIR-V compilation should also take into consideration the execution environment, so we generate SPIR-V modules valid for the target environment. This is conveyed by the spv.target_env  attribute. The SetSPIRVCapabilities pass, adds these various capabilties for the SPIR-V execution. The attribute #spv.vce has a few fields:

A #spv.vce (spirv::VerCapExtAttr) attribute:
1. The target SPIR-V version.
2. A list of SPIR-V capabilities for the target. SPIR-V Capabilities: Capabilities are specific features supported by the target architecture. E.g., VectorAnyIntel capabilities means, the target architecture has the ability to handle any vectors of length (2 to 2^64-1). A SPIR-V module needs to specify the features (capabilities) used by the module so that the client API that consumes this module knows what capabilities are used in the module and may decide to accept and reject the module based on whether it supports them or not. It also allows a validator to validate that the module uses only its declared capabilities.
3. A list of SPIR-V extensions for the target. SPIR-V Extensions: SPIR-V specification allows multiple vendors or parties simultaneously extend the SPIR-V specification for their need. This field lists the extensions supported by the target architecture. Extension may indicate the availability of different types of (capabilities) features (e.g., types, ops, enum case). A extension indicates the availability of one or multiple capabilities (features).

# Example

```
// -----// IR Dump Before {anonymous}::SetSPIRVCapabilitiesPass () //----- //
module attributes {gpu.container_module} {
  func.func @main() {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 2.200000e+00 : f32
    %cst_0 = arith.constant 1.100000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %memref = gpu.alloc  () {gpu.alloc_shared} : memref<8xf32>
    %memref_2 = gpu.alloc  () {gpu.alloc_shared} : memref<8xf32>
    %memref_3 = gpu.alloc  () {gpu.alloc_shared} : memref<8xf32>
    %0 = memref.cast %memref : memref<8xf32> to memref<?xf32>
    %1 = memref.cast %memref_2 : memref<8xf32> to memref<?xf32>
    %2 = memref.cast %memref_3 : memref<8xf32> to memref<?xf32>
    call @fillResource1DFloat(%0, %cst_0) : (memref<?xf32>, f32) -> ()
    call @fillResource1DFloat(%1, %cst) : (memref<?xf32>, f32) -> ()
    call @fillResource1DFloat(%2, %cst_1) : (memref<?xf32>, f32) -> ()
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8xf32>, %memref_2 : memref<8xf32>, %memref_3 : memref<8xf32>)
    %3 = memref.cast %memref_3 : memref<8xf32> to memref<*xf32>
    call @printMemrefF32(%3) : (memref<*xf32>) -> ()
    return
  }
```

The Pass will change the IR to:

```
// -----// IR Dump After {anonymous}::SetSPIRVCapabilitiesPass () //----- //
module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, #spv.resource_limits<>>} {
  func.func @main() {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 2.200000e+00 : f32
    %cst_0 = arith.constant 1.100000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %memref = gpu.alloc  () {gpu.alloc_shared} : memref<8xf32>
    %memref_2 = gpu.alloc  () {gpu.alloc_shared} : memref<8xf32>
    %memref_3 = gpu.alloc  () {gpu.alloc_shared} : memref<8xf32>
    %0 = memref.cast %memref : memref<8xf32> to memref<?xf32>
    %1 = memref.cast %memref_2 : memref<8xf32> to memref<?xf32>
    %2 = memref.cast %memref_3 : memref<8xf32> to memref<?xf32>
    call @fillResource1DFloat(%0, %cst_0) : (memref<?xf32>, f32) -> ()
    call @fillResource1DFloat(%1, %cst) : (memref<?xf32>, f32) -> ()
    call @fillResource1DFloat(%2, %cst_1) : (memref<?xf32>, f32) -> ()
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8xf32>, %memref_2 : memref<8xf32>, %memref_3 : memref<8xf32>)
    %3 = memref.cast %memref_3 : memref<8xf32> to memref<*xf32>
    call @printMemrefF32(%3) : (memref<*xf32>) -> ()
    return
  }
```


As shown in the example above, the pass adds the SPIR-V capabilites as an attribute.


## Reason for this Custom Pass:

Upstream does not have a pass which does these conversions. This pass add a lot of things specific to Intel GPU. So, maybe we can have it as a custom pass rather than upstreaming.
