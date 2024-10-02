// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/imex-emulate-non-native-bf16.pp \
// RUN:                                       --no-mlir-runner --filecheck

module @relu attributes {gpu.container_module} {
  memref.global "private" constant @__constant_4x5xbf16 : memref<4x5xbf16> = dense<[[-1.000980e-01, -2.001950e-01, -3.007810e-01, 4.003910e-01, 5.000000e-01], [1.000980e-01, -2.001950e-01, 3.007810e-01, -4.003910e-01, 5.000000e-01], [1.000980e-01, 2.001950e-01, 3.007810e-01, -4.003910e-01, -5.000000e-01], [1.000980e-01, 2.001950e-01, 3.007810e-01, 4.003910e-01, 5.000000e-01]]>
  func.func @main() {
    %0 = memref.get_global @__constant_4x5xbf16 : memref<4x5xbf16>
    %1 = call @test(%0) : (memref<4x5xbf16>) -> memref<4x5xbf16>
    %cast = memref.cast %1 : memref<4x5xbf16> to memref<*xbf16>
    call @printMemrefBF16(%cast) : (memref<*xbf16>) -> ()
    return
  }
  func.func private @printMemrefBF16(memref<*xbf16>)
  func.func @test(%arg0: memref<4x5xbf16>) -> memref<4x5xbf16> {
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<4x5xbf16>
    memref.copy %arg0, %memref : memref<4x5xbf16> to memref<4x5xbf16>
    %memref_0 = gpu.alloc  () : memref<4x5xi1>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c4, %c5, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<4x5xbf16>, %cst : bf16, %memref_0 : memref<4x5xi1>)
    %memref_1 = gpu.alloc  host_shared () : memref<4x5xbf16>
    gpu.launch_func  @test_kernel_0::@test_kernel_0 blocks in (%c4, %c5, %c1) threads in (%c1, %c1, %c1) args(%memref_0 : memref<4x5xi1>, %memref : memref<4x5xbf16>, %cst : bf16, %memref_1 : memref<4x5xbf16>)
    gpu.dealloc  %memref_0 : memref<4x5xi1>
    gpu.dealloc  %memref : memref<4x5xbf16>
    return %memref_1 : memref<4x5xbf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, BFloat16TypeKHR, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_bfloat16, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    // CHECK: gpu.func @test_kernel(%arg0: memref<4x5xbf16>, %arg1: bf16, %arg2: memref<4x5xi1>)
    gpu.func @test_kernel(%arg0: memref<4x5xbf16>, %arg1: bf16, %arg2: memref<4x5xi1>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 5, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      // CHECK: %[[VAR2_1:.*]] = memref.load %arg0[%[[VAR0_1:.*]], %[[VAR1_1:.*]]] : memref<4x5xbf16>
      %2 = memref.load %arg0[%0, %1] : memref<4x5xbf16>
      // CHECK: %[[VAR3_1:.*]] = arith.extf %[[VAR2_1]] : bf16 to f32
      // CHECK: %[[VAR4_1:.*]] = arith.extf %arg1 : bf16 to f32
      // CHECK: %[[VAR5_1:.*]] = arith.cmpf olt, %[[VAR3_1]], %[[VAR4_1]] : f32
      %3 = arith.cmpf olt, %2, %arg1 : bf16
      memref.store %3, %arg2[%0, %1] : memref<4x5xi1>
      gpu.return
    }
  }
  gpu.module @test_kernel_0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    // CHECK: gpu.func @test_kernel_0(%arg0: memref<4x5xi1>, %arg1: memref<4x5xbf16>, %arg2: bf16, %arg3: memref<4x5xbf16>)
    gpu.func @test_kernel_0(%arg0: memref<4x5xi1>, %arg1: memref<4x5xbf16>, %arg2: bf16, %arg3: memref<4x5xbf16>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 5, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<4x5xi1>
      %3 = memref.load %arg1[%0, %1] : memref<4x5xbf16>
      // CHECK: %[[VAR4_2:.*]] = arith.extf %arg2 : bf16 to f32
      // CHECK: %[[VAR5_2:.*]] = arith.extf %[[VAR3_2:.*]] : bf16 to f32
      // CHECK: %[[VAR6_2:.*]] = arith.select %[[VAR2_2:.*]], %[[VAR4_2]], %[[VAR5_2]] : f32
      // CHECK: %[[VAR7_2:.*]] = arith.truncf %[[VAR6_2]] : f32 to bf16
      %4 = arith.select %2, %arg2, %3 : bf16
      // CHECK: memref.store %[[VAR7_2]], %arg3[%[[VAR0_2:.*]], %[[VAR1_2:.*]]] : memref<4x5xbf16>
      memref.store %4, %arg3[%0, %1] : memref<4x5xbf16>
      gpu.return
    }
  }
}
