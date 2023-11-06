// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/bf16-to-gpu.pp \
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
    // CHECK: %[[MEMREF:.*]] = gpu.alloc  host_shared () : memref<40xi8>
    // CHECK: %[[VIEW:.*]] = memref.view %[[MEMREF]][%[[CONST0:.*]]][] : memref<40xi8> to memref<4x5xbf16>
    // CHECK: %[[VIEW_0:.*]] = memref.view %[[MEMREF]][%[[CONST0]]][] : memref<40xi8> to memref<4x5xi16>
    // CHECK: memref.copy %arg0, %[[VIEW]] : memref<4x5xbf16> to memref<4x5xbf16>
    %memref = gpu.alloc  host_shared () : memref<4x5xbf16>
    memref.copy %arg0, %memref : memref<4x5xbf16> to memref<4x5xbf16>
    %memref_0 = gpu.alloc  () : memref<4x5xi1>
    // CHECK: args(%[[VIEW_0]] : memref<4x5xi16>, %[[CONST0_I16:.*]] : i16
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c4, %c5, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<4x5xbf16>, %cst : bf16, %memref_0 : memref<4x5xi1>)
    // CHECK: %[[MEMREF_2:.*]] = gpu.alloc  host_shared () : memref<40xi8>
    // CHECK: %[[VIEW_3:.*]] = memref.view %[[MEMREF_2]][%c0][] : memref<40xi8> to memref<4x5xbf16>
    // CHECK: %[[VIEW_4:.*]] = memref.view %[[MEMREF_2]][%c0][] : memref<40xi8> to memref<4x5xi16>
    %memref_1 = gpu.alloc  host_shared () : memref<4x5xbf16>
    // CHECK: args(%[[MEMREF_1]] : memref<4x5xi1>, %[[VIEW_0]] : memref<4x5xi16>, %[[CONST0_I16]] : i16, %[[VIEW_4]] : memref<4x5xi16>)
    gpu.launch_func  @test_kernel_0::@test_kernel blocks in (%c4, %c5, %c1) threads in (%c1, %c1, %c1) args(%memref_0 : memref<4x5xi1>, %memref : memref<4x5xbf16>, %cst : bf16, %memref_1 : memref<4x5xbf16>)
    gpu.dealloc  %memref_0 : memref<4x5xi1>
    // CHECK: gpu.dealloc  %[[MEMREF]] : memref<40xi8>
    // CHECK: return %[[VIEW_3]] : memref<4x5xbf16>
    gpu.dealloc  %memref : memref<4x5xbf16>
    return %memref_1 : memref<4x5xbf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    // CHECK: gpu.func @test_kernel(%arg0: memref<4x5xi16>, %arg1: i16
    gpu.func @test_kernel(%arg0: memref<4x5xbf16>, %arg1: bf16, %arg2: memref<4x5xi1>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 5, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      // CHECK: %[[VAR2_1:.*]] = memref.load %arg0[%[[VAR0_1]], %[[VAR1_1]]] : memref<4x5xi16>
      %2 = memref.load %arg0[%0, %1] : memref<4x5xbf16>
      // CHECK: %[[VAR3_1:.*]] = arith.bitcast %[[VAR2_1]] : i16 to bf16
      // CHECK: %[[VAR4_1:.*]] = arith.extf %[[VAR3_1]] : bf16 to f32
      // CHECK: %[[VAR5_1:.*]] = arith.bitcast %arg1 : i16 to bf16
      // CHECK: %[[VAR6_1:.*]] = arith.extf %[[VAR5_1]] : bf16 to f32
      // CHECK: %[[VAR7_1:.*]] = arith.cmpf olt, %[[VAR4_1]], %[[VAR6_1]] : f32
      %3 = arith.cmpf olt, %2, %arg1 : bf16
      memref.store %3, %arg2[%0, %1] : memref<4x5xi1>
      gpu.return
    }
  }
  gpu.module @test_kernel_0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    // CHECK: gpu.func @test_kernel(%arg0: memref<4x5xi1>, %arg1: memref<4x5xi16>, %arg2: i16, %arg3: memref<4x5xi16>)
    gpu.func @test_kernel(%arg0: memref<4x5xi1>, %arg1: memref<4x5xbf16>, %arg2: bf16, %arg3: memref<4x5xbf16>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 5, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<4x5xi1>
      // CHECK: %[[VAR3_2:.*]] = memref.load %arg1[%[[VAR0_2]], %[[VAR1_2]]] : memref<4x5xi16>
      %3 = memref.load %arg1[%0, %1] : memref<4x5xbf16>
      // CHECK: %[[VAR4_2:.*]] = arith.bitcast %arg2 : i16 to bf16
      // CHECK: %[[VAR5_2:.*]] = arith.extf %[[VAR4_2]] : bf16 to f32
      // CHECK: %[[VAR6_2:.*]] = arith.bitcast %[[VAR3_2]] : i16 to bf16
      // CHECK: %[[VAR7_2:.*]] = arith.extf %[[VAR6_2]] : bf16 to f32
      // CHECK: %[[VAR8_2:.*]] = arith.select %[[VAR2_2]], %[[VAR5_2]], %[[VAR7_2]] : f32
      // CHECK: %[[VAR9_2:.*]] = arith.truncf %[[VAR8_2]] : f32 to bf16
      // CHECK: %[[VAR10_2:.*]] = arith.bitcast %[[VAR9_2]] : bf16 to i16
      %4 = arith.select %2, %arg2, %3 : bf16
      // CHECK: memref.store %[[VAR10_2]], %arg3[%[[VAR0_2]], %[[VAR1_2]]] : memref<4x5xi16>
      memref.store %4, %arg3[%0, %1] : memref<4x5xbf16>
      gpu.return
    }
  }
}
