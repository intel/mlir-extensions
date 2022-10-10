# SerializeSPIRV Pass (serialize-spirv)

The SerializeSPIRV pass utilizes the upstream spirv::serialize() function to serialize MLIR SPIR-V module to SPIR-V binary and attaches the binary as a string attribute to the gpuModule(attributes {gpu.binary}).
This pass works like the upstream SerializeToCubin and SerializeToHsaco, only that the other two passes translate llvm dialect to llvm IR and then translate to ISA binary.

## Example

```
// -----// IR Dump Before SerializeSPIRV //----- //
module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, #spv.resource_limits<>>} {
  spv.module @__spv__addt_kernel Physical64 OpenCL requires #spv.vce<v1.0, [Int64, Addresses, Kernel], []> {
    spv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi64>, Input>
    spv.func @addt_kernel(%arg0: !spv.ptr<f32, CrossWorkgroup>, %arg1: !spv.ptr<f32, CrossWorkgroup>, %arg2: !spv.ptr<f32, CrossWorkgroup>) "None" attributes {spv.entry_point_abi = #spv.entry_point_abi<>, workgroup_attributions = 0 : i64} {
      %cst5_i64 = spv.Constant 5 : i64
      %__builtin_var_WorkgroupId___addr = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi64>, Input>
      %0 = spv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi64>, Input>
      .
      .
      .
      %13 = spv.IMul %1, %cst5_i64 : i64
      %14 = spv.IAdd %13, %3 : i64
      %15 = spv.InBoundsPtrAccessChain %arg2[%14] : !spv.ptr<f32, CrossWorkgroup>, i64
      spv.Store "CrossWorkgroup" %15, %12 ["Aligned", 4] : f32
      spv.Return
    }
    spv.EntryPoint "Kernel" @addt_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @addt_kernel {
    gpu.func @addt_kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) kernel attributes {spv.entry_point_abi = #spv.entry_point_abi<>} {
      %c5 = arith.constant 5 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      .
      .
      .
      %9 = arith.muli %0, %c5 : index
      %10 = arith.addi %9, %1 : index
      memref.store %8, %arg2[%10] : memref<?xf32>
      gpu.return
    }
  }
}

```

The Pass will change the IR to:

```
// -----// IR Dump After SerializeSPIRV //----- //
module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Ve    ctor16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, #spv.re    source_limits<>>} {
  gpu.module @addt_kernel attributes {gpu.binary = "\03\02#\07\00\00\01\00 ... \00"} {
    gpu.func @addt_kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) kernel attributes {spv.entry_point_abi = #spv.entry_point_abi<>    } {
      %c5 = arith.constant 5 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      .
      .
      .
      %9 = arith.muli %0, %c5 : index
      %10 = arith.addi %9, %1 : index
      memref.store %8, %arg2[%10] : memref<?xf32>
      gpu.return
    }
  }
}

```

As shown in the example above, the spv module op was serialized to spv binary and attached to the gpu module op as a "gpu.binary" attribute


## Reason for this Custom Pass:

Upstream does not have a standalone pass which wraps this function. We can upstream this if it proves this is a common flow needed.
