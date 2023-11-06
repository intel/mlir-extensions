// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/bf16-to-gpu.pp \
// RUN:                                       --no-mlir-runner --filecheck

module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_3x3xbf16_1 : memref<3x3xbf16> = dense<1.000000e+00>
  memref.global "private" constant @__constant_3x3xbf16_0 : memref<3x3xbf16> = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [3.000000e+00, 4.000000e+00, 5.000000e-01], [3.000000e+00, 3.000000e+00, 3.000000e+00]]>
  memref.global "private" constant @__constant_3x3xbf16 : memref<3x3xbf16> = dense<[[5.000000e-01, 2.001950e-01, 4.000000e+00], [1.000000e+00, 1.000000e+00, 2.000000e+00], [3.000000e+00, 3.000000e+00, 3.007810e-01]]>
  func.func @main() {
    %0 = memref.get_global @__constant_3x3xbf16 : memref<3x3xbf16>
    %1 = memref.get_global @__constant_3x3xbf16_0 : memref<3x3xbf16>
    %2 = memref.get_global @__constant_3x3xbf16_1 : memref<3x3xbf16>
    %3 = call @test(%0, %1, %2) : (memref<3x3xbf16>, memref<3x3xbf16>, memref<3x3xbf16>) -> memref<3x3xbf16>
    %cast = memref.cast %3 : memref<3x3xbf16> to memref<*xbf16>
    call @printMemrefBF16(%cast) : (memref<*xbf16>) -> ()
    return
  }
  func.func private @printMemrefBF16(memref<*xbf16>)
  func.func @test(%arg0: memref<3x3xbf16>, %arg1: memref<3x3xbf16>, %arg2: memref<3x3xbf16>) -> memref<3x3xbf16> {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    // CHECK: %[[MEMREF:.*]] = gpu.alloc  host_shared () : memref<18xi8>
    // CHECK: %[[VIEW:.*]] = memref.view %[[MEMREF]][%[[CONST0:.*]]][] : memref<18xi8> to memref<3x3xbf16>
    // CHECK: %[[VIEW_0:.*]] = memref.view %[[MEMREF]][%[[CONST0]]][] : memref<18xi8> to memref<3x3xi16>
    // CHECK: memref.copy %arg1, %[[VIEW]] : memref<3x3xbf16> to memref<3x3xbf16>
    %memref = gpu.alloc  host_shared () : memref<3x3xbf16>
    memref.copy %arg1, %memref : memref<3x3xbf16> to memref<3x3xbf16>
    // CHECK: %[[MEMREF_1:.*]] = gpu.alloc  host_shared () : memref<18xi8>
    // CHECK: %[[VIEW_2:.*]] = memref.view %[[MEMREF_1]][%[[CONST0]]][] : memref<18xi8> to memref<3x3xbf16>
    // CHECK: %[[VIEW_3:.*]] = memref.view %[[MEMREF_1]][%[[CONST0]]][] : memref<18xi8> to memref<3x3xi16>
    // CHECK: memref.copy %arg0, %[[VIEW_2]] : memref<3x3xbf16> to memref<3x3xbf16>
    %memref_0 = gpu.alloc  host_shared () : memref<3x3xbf16>
    memref.copy %arg0, %memref_0 : memref<3x3xbf16> to memref<3x3xbf16>
    // CHECK: %[[MEMREF_4:.*]] = gpu.alloc  host_shared () : memref<18xi8>
    // CHECK: %[[VIEW_5:.*]] = memref.view %[[MEMREF_4]][%[[CONST0]]][] : memref<18xi8> to memref<3x3xbf16>
    // CHECK: %[[VIEW_6:.*]] = memref.view %[[MEMREF_4]][%[[CONST0]]][] : memref<18xi8> to memref<3x3xi16>
    // CHECK: memref.copy %arg2, %[[VIEW_5]] : memref<3x3xbf16> to memref<3x3xbf16>
    %memref_1 = gpu.alloc  host_shared () : memref<3x3xbf16>
    memref.copy %arg2, %memref_1 : memref<3x3xbf16> to memref<3x3xbf16>
    // CHECK: args(%[[VIEW_3]] : memref<3x3xi16>, %[[VIEW_0]] : memref<3x3xi16>, %[[VIEW_6]] : memref<3x3xi16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c3, %c3, %c1) threads in (%c1, %c1, %c1) args(%memref_0 : memref<3x3xbf16>, %memref : memref<3x3xbf16>, %memref_1 : memref<3x3xbf16>, %c0 : index, %c3 : index, %c1 : index)
    // CHECK: gpu.dealloc  %[[MEMREF_1]] : memref<18xi8>
    // CHECK: gpu.dealloc  %[[MEMREF]] : memref<18xi8>
    // CHECK: return %[[VIEW_5]] : memref<3x3xbf16>
    gpu.dealloc  %memref_0 : memref<3x3xbf16>
    gpu.dealloc  %memref : memref<3x3xbf16>
    return %memref_1 : memref<3x3xbf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    // CHECK: gpu.func @test_kernel(%arg0: memref<3x3xi16>, %arg1: memref<3x3xi16>, %arg2: memref<3x3xi16>, %arg3: index, %arg4: index, %arg5: index)
    gpu.func @test_kernel(%arg0: memref<3x3xbf16>, %arg1: memref<3x3xbf16>, %arg2: memref<3x3xbf16>, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 3, 3, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      scf.for %arg6 = %arg3 to %arg4 step %arg5 {
        // CHECK: %[[VAR2:.*]] = memref.load %arg0[%[[VAR0:.*]], %arg6] : memref<3x3xi16>
        // CHECK: %[[VAR3:.*]] = memref.load %arg1[%arg6, %[[VAR1:.*]]] : memref<3x3xi16>
        // CHECK: %[[VAR4:.*]] = memref.load %arg2[%[[VAR0]], %[[VAR1]]] : memref<3x3xi16>
        %2 = memref.load %arg0[%0, %arg6] : memref<3x3xbf16>
        %3 = memref.load %arg1[%arg6, %1] : memref<3x3xbf16>
        %4 = memref.load %arg2[%0, %1] : memref<3x3xbf16>
        // CHECK: %[[VAR5:.*]] = arith.bitcast %[[VAR2]] : i16 to bf16
        // CHECK: %[[VAR6:.*]] = arith.extf %[[VAR5]] : bf16 to f32
        // CHECK: %[[VAR7:.*]] = arith.bitcast %[[VAR3]] : i16 to bf16
        // CHECK: %[[VAR8:.*]] = arith.extf %[[VAR7]] : bf16 to f32
        // CHECK: %[[VAR9:.*]] = arith.mulf %[[VAR6]], %[[VAR8]] : f32
        // CHECK: %[[VAR10:.*]] = arith.truncf %[[VAR9]] : f32 to bf16
        %5 = arith.mulf %2, %3 : bf16
        // CHECK: %[[VAR11:.*]] = arith.bitcast %[[VAR4]] : i16 to bf16
        // CHECK: %[[VAR12:.*]] = arith.extf %[[VAR11]] : bf16 to f32
        // CHECK: %[[VAR13:.*]] = arith.extf %[[VAR10]] : bf16 to f32
        // CHECK: %[[VAR14:.*]] = arith.addf %[[VAR12]], %[[VAR13]] : f32
        // CHECK: %[[VAR15:.*]] = arith.truncf %[[VAR14]] : f32 to bf16
        // CHECK: %[[VAR16:.*]] = arith.bitcast %[[VAR15]] : bf16 to i16
        %6 = arith.addf %4, %5 : bf16
        // CHECK: memref.store %[[VAR16]], %arg2[%[[VAR0]], %[[VAR1]]] : memref<3x3xi16>
        memref.store %6, %arg2[%0, %1] : memref<3x3xbf16>
      }
      gpu.return
    }
  }
}
