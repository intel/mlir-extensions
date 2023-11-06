// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/bf16-to-gpu.pp \
// RUN:                                       --no-mlir-runner --filecheck

module @broadcast_non_numpy attributes {gpu.container_module} {
  memref.global "private" constant @__constant_3xbf16 : memref<3xbf16> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]>
  func.func @test(%arg0: memref<3xbf16>) -> memref<3x4xbf16> {
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c1 = arith.constant 1 : index
    // CHECK: %[[MEMREF:.*]] = gpu.alloc  host_shared () : memref<6xi8>
    // CHECK: %[[VIEW:.*]] = memref.view %memref[%[[CONST0]]][] : memref<6xi8> to memref<3xbf16>
    // CHECK: %[[VIEW0:.*]] = memref.view %memref[%[[CONST0]]][] : memref<6xi8> to memref<3xi16>
    // CHECK: memref.copy %arg0, %[[VIEW]] : memref<3xbf16> to memref<3xbf16>
    %memref = gpu.alloc  host_shared () : memref<3xbf16>
    memref.copy %arg0, %memref : memref<3xbf16> to memref<3xbf16>
    // CHECK: %[[MEMREF1:.*]] = gpu.alloc  () : memref<24xi8>
    // CHECK: %[[VIEW2:.*]] = memref.view %[[MEMREF1]][%[[CONST0]]][] : memref<24xi8> to memref<3x4xi16>
    %memref_0 = gpu.alloc  () : memref<3x4xbf16>
    // CHECK: gpu.launch_func  @test_kernel::@test_kernel blocks in (%[[CONST3]], %[[CONST4]], %[[CONST1]]) threads in (%[[CONST1]], %[[CONST1]], %[[CONST1]]) args(%[[CONST0_I16]] : i16, %[[VIEW2]] : memref<3x4xi16>)
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c3, %c4, %c1) threads in (%c1, %c1, %c1) args(%cst : bf16, %memref_0 : memref<3x4xbf16>)
    // CHECK: %[[MEMREF3:.*]] = gpu.alloc  host_shared () : memref<24xi8>
    // CHECK: %[[VIEW4:.*]] = memref.view %[[MEMREF3]][%[[CONST0]]][] : memref<24xi8> to memref<3x4xbf16>
    // CHECK: %[[VIEW5:.*]] = memref.view %[[MEMREF3]][%[[CONST0]]][] : memref<24xi8> to memref<3x4xi16>
    %memref_1 = gpu.alloc  host_shared () : memref<3x4xbf16>
    // CHECK: gpu.launch_func  @test_kernel_0::@test_kernel blocks in (%[[CONST3]], %[[CONST4]], %[[CONST1]]) threads in (%[[CONST1]], %[[CONST1]], %[[CONST1]]) args(%[[VIEW0]] : memref<3xi16>, %[[VIEW5]] : memref<3x4xi16>)
    gpu.launch_func  @test_kernel_0::@test_kernel blocks in (%c3, %c4, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<3xbf16>, %memref_1 : memref<3x4xbf16>)
    // CHECK: gpu.dealloc  %[[MEMREF1]] : memref<24xi8>
    gpu.dealloc  %memref_0 : memref<3x4xbf16>
    // CHECK: gpu.dealloc  %[[MEMREF]] : memref<6xi8>
    gpu.dealloc  %memref : memref<3xbf16>
    // CHECK: return %[[VIEW4]] : memref<3x4xbf16>
    return %memref_1 : memref<3x4xbf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    // CHECK:   gpu.func @test_kernel(%arg0: i16, %arg1: memref<3x4xi16>)
    gpu.func @test_kernel(%arg0: bf16, %arg1: memref<3x4xbf16>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 3, 4, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      // CHECK:   memref.store %arg0, %arg1[%[[VAR0_1:.*]], %[[VAR1_1:.*]]] : memref<3x4xi16>
      memref.store %arg0, %arg1[%0, %1] : memref<3x4xbf16>
      gpu.return
    }
  }
  gpu.module @test_kernel_0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    // CHECK: gpu.func @test_kernel(%arg0: memref<3xi16>, %arg1: memref<3x4xi16>)
    gpu.func @test_kernel(%arg0: memref<3xbf16>, %arg1: memref<3x4xbf16>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 3, 4, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      // CHECK:  %[[VAR2_2:.*]] = memref.load %arg0[%[[VAR0_2]]] : memref<3xi16>
      // CHECK:  memref.store %[[VAR2_2]], %arg1[%[[VAR0_2]], %[[VAR1_2]]] : memref<3x4xi16>
      %2 = memref.load %arg0[%0] : memref<3xbf16>
      memref.store %2, %arg1[%0, %1] : memref<3x4xbf16>
      gpu.return
    }
  }
  func.func @main() {
    %0 = memref.get_global @__constant_3xbf16 : memref<3xbf16>
    %1 = call @test(%0) : (memref<3xbf16>) -> memref<3x4xbf16>
    %cast = memref.cast %1 : memref<3x4xbf16> to memref<*xbf16>
    call @printMemrefBF16(%cast) : (memref<*xbf16>) -> ()
    return
  }
  func.func private @printMemrefBF16(memref<*xbf16>)
}
