// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_8x16xf32 : memref<8x16xf32> = dense<1.0>
  func.func @test(%arg0: memref<8x16xf32>,%arg1:index)attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref_0 = gpu.alloc  host_shared () : memref<8x16xf32>
    memref.copy %arg0, %memref_0 : memref<8x16xf32> to memref<8x16xf32>
    %memref_1 = gpu.alloc  host_shared () : memref<8x16xf32>
    gpu.launch_func  @test_kernel::@test_padding_f32 blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref_0 : memref<8x16xf32>, %memref_1 : memref<8x16xf32>, %arg1:index)
    %cast1 = memref.cast %memref_1 : memref<8x16xf32> to memref<*xf32>
    call @printMemrefF32(%cast1) : (memref<*xf32>) -> ()
    return
  }

  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {

    gpu.func @test_padding_f32(%arg0: memref<8x16xf32>, %arg1: memref<8x16xf32>, %arg3:index) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %arg0[%arg3, %arg3]
      : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      %1 = xegpu.create_nd_tdesc %arg1[0, 0]
      : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      %3 = xegpu.load_nd %0 : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      xegpu.store_nd %3,%1 : vector<8x16xf32>,!xegpu.tensor_desc<8x16xf32>
      gpu.return
    }

  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_8x16xf32 : memref<8x16xf32>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    call @test(%0, %c1) : (memref<8x16xf32>, index)-> ()
    call @test(%0, %c2) : (memref<8x16xf32>, index)-> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}


// CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
// CHECK-SAME: rank = 2 offset = 0 sizes = [8, 16] strides = [16, 1] data =
// CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0]
// CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
// CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
// CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
// CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
// CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
// CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
// CHECK: [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]
// CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
// CHECK-SAME: rank = 2 offset = 0 sizes = [8, 16] strides = [16, 1] data =
// CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0],
// CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0],
// CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0],
// CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0],
// CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0],
// CHECK: [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0],
// CHECK: [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
// CHECK: [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]
