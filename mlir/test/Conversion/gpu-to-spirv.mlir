// RUN: imex-opt --gpu-to-spirv %s | FileCheck %s

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
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) kernel attributes {spv.entry_point_abi = #spv.entry_point_abi<>} {
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %0 = gpu.block_id  x
      %1 = memref.load %arg0[%0] : memref<8xf32>
      %2 = memref.load %arg1[%0] : memref<8xf32>
      %3 = arith.addf %1, %2 : f32
      memref.store %3, %arg2[%0] : memref<8xf32>
      gpu.return
    }

    // CHECK: spv.module @__spv__main_kernel Physical64 OpenCL {
    // CHECK: spv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spv.ptr<vector<3xi64>, Input>
    // CHECK: spv.func @main_kernel(%arg0: !spv.ptr<f32, CrossWorkgroup>, %arg1: !spv.ptr<f32, CrossWorkgroup>, %arg2: !spv.ptr<f32, CrossWorkgroup>) "None" attributes {spv.entry_point_abi = #spv.entry_point_abi<>, workgroup_attributions = 0 : i64} {
    // CHECK: spv.Branch ^bb1
    // CHECK: ^bb1:  // pred: ^bb0
    // CHECK: %__builtin_var_WorkgroupId___addr = spv.mlir.addressof @__builtin_var_WorkgroupId__ : !spv.ptr<vector<3xi64>, Input>
    // CHECK: %[[VAR0:.*]] = spv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
    // CHECK: %[[VAR1:.*]] = spv.CompositeExtract %[[VAR0:.*]][0 : i32] : vector<3xi64>
    // CHECK: %[[VAR2:.*]] = spv.InBoundsPtrAccessChain %arg0[%[[VAR1:.*]]] : !spv.ptr<f32, CrossWorkgroup>, i64
    // CHECK: %[[VAR3:.*]] = spv.Load "CrossWorkgroup" %[[VAR2:.*]] ["Aligned", 4] : f32
    // CHECK: %[[VAR4:.*]] = spv.InBoundsPtrAccessChain %arg1[%[[VAR1:.*]]] : !spv.ptr<f32, CrossWorkgroup>, i64
    // CHECK: %[[VAR5:.*]] = spv.Load "CrossWorkgroup" %[[VAR4:.*]] ["Aligned", 4] : f32
    // CHECK:  %[[VAR6:.*]] = spv.FAdd %[[VAR3:.*]], %[[VAR5:.*]] : f32
    // CHECK:  %[[VAR7:.*]] = spv.InBoundsPtrAccessChain %arg2[%[[VAR1:.*]]] : !spv.ptr<f32, CrossWorkgroup>, i64
    // CHECK:   spv.Store "CrossWorkgroup" %[[VAR7:.*]], %[[VAR6:.*]] ["Aligned", 4] : f32
    // CHECK:   spv.Return
    // CHECK:  }
    // CHECK: }

  }
  func.func private @fillResource1DFloat(memref<?xf32>, f32)
  func.func private @printMemrefF32(memref<*xf32>)
}
