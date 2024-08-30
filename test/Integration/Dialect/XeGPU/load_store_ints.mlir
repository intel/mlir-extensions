// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc-rawsend-false.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc-rawsend-false.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<8x16xi8>, %arg1: memref<8x16xi8>, %arg2: memref<8x16xi16>, %arg3: memref<8x16xi16>) -> (memref<8x16xi32>, memref<8x16xi32>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index

    %memref = gpu.alloc  host_shared () : memref<8x16xi8>
    memref.copy %arg0, %memref : memref<8x16xi8> to memref<8x16xi8>
    %memref_1 = gpu.alloc  host_shared () : memref<8x16xi8>
    memref.copy %arg1, %memref_1 : memref<8x16xi8> to memref<8x16xi8>
    %memref_2 = gpu.alloc  host_shared () : memref<8x16xi32>

    %memref_3 = gpu.alloc  host_shared () : memref<8x16xi16>
    memref.copy %arg2, %memref_3 : memref<8x16xi16> to memref<8x16xi16>
    %memref_4 = gpu.alloc  host_shared () : memref<8x16xi16>
    memref.copy %arg3, %memref_4 : memref<8x16xi16> to memref<8x16xi16>
    %memref_5 = gpu.alloc  host_shared () : memref<8x16xi32>

    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c8, %c1, %c1) args(%memref : memref<8x16xi8>, %memref_1 : memref<8x16xi8>, %memref_3 : memref<8x16xi16>, %memref_4 : memref<8x16xi16>, %memref_2 : memref<8x16xi32>, %memref_5 : memref<8x16xi32>)
    gpu.dealloc  %memref : memref<8x16xi8>
    gpu.dealloc  %memref_1 : memref<8x16xi8>
    gpu.dealloc  %memref_3 : memref<8x16xi16>
    gpu.dealloc  %memref_4 : memref<8x16xi16>
    return %memref_2, %memref_5 : memref<8x16xi32>, memref<8x16xi32>
  }

  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<8x16xi8>, %arg1: memref<8x16xi8>, %arg2: memref<8x16xi16>, %arg3: memref<8x16xi16>, %arg4: memref<8x16xi32>, %arg5: memref<8x16xi32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %thread_id_x = gpu.thread_id x
      cf.br ^bb1
    ^bb1:
      %0 = xegpu.create_nd_tdesc %arg1[%thread_id_x, 0] : memref<8x16xi8> -> !xegpu.tensor_desc<16xi8>
      %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<16xi8> -> vector<16xi8>
      %2 = xegpu.create_nd_tdesc %arg0[%thread_id_x, 0] : memref<8x16xi8> -> !xegpu.tensor_desc<16xi8>
      %3 = xegpu.load_nd %2  : !xegpu.tensor_desc<16xi8> -> vector<16xi8>
      %4 = arith.addi %3, %1 : vector<16xi8>
      %5 = arith.extui %4 :vector<16xi8> to vector<16xi32>
      %6 = xegpu.create_nd_tdesc %arg4[%thread_id_x, 0] : memref<8x16xi32> -> !xegpu.tensor_desc<16xi32>
      xegpu.store_nd %5, %6  : vector<16xi32>, !xegpu.tensor_desc<16xi32>

      %7 = xegpu.create_nd_tdesc %arg2[%thread_id_x, 0] : memref<8x16xi16> -> !xegpu.tensor_desc<16xi16>
      %8 = xegpu.load_nd %7  : !xegpu.tensor_desc<16xi16> -> vector<16xi16>
      %9 = xegpu.create_nd_tdesc %arg3[%thread_id_x, 0] : memref<8x16xi16> -> !xegpu.tensor_desc<16xi16>
      %10 = xegpu.load_nd %9  : !xegpu.tensor_desc<16xi16> -> vector<16xi16>
      %11 = arith.addi %8, %10 : vector<16xi16>
      %12 = arith.extui %11 :vector<16xi16> to vector<16xi32>
      %13 = xegpu.create_nd_tdesc %arg5[%thread_id_x, 0] : memref<8x16xi32> -> !xegpu.tensor_desc<16xi32>
      xegpu.store_nd %12, %13  : vector<16xi32>, !xegpu.tensor_desc<16xi32>

      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index

    %A = memref.alloc() : memref<8x16xi8>
    %B = memref.alloc() : memref<8x16xi8>
    %C = memref.alloc() : memref<8x16xi16>
    %D = memref.alloc() : memref<8x16xi16>

    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %val = index.castu %j : index to i8
        %val2 = index.castu %j : index to i16
        memref.store %val, %A[%i, %j] : memref<8x16xi8>
        memref.store %val, %B[%i, %j] : memref<8x16xi8>
        memref.store %val2, %C[%i, %j] : memref<8x16xi16>
        memref.store %val2, %D[%i, %j] : memref<8x16xi16>
      }
    }

    %res0, %res1 = call @test(%A, %B, %C, %D) : (memref<8x16xi8>, memref<8x16xi8>, memref<8x16xi16>, memref<8x16xi16>) -> (memref<8x16xi32>, memref<8x16xi32>)

    %res0_cast = memref.cast %res0 : memref<8x16xi32> to memref<*xi32>
    %res1_cast = memref.cast %res1 : memref<8x16xi32> to memref<*xi32>

    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [8, 16] strides = [16, 1] data =
    // CHECK-NEXT{LITERAL}: [[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    // CHECK-NEXT{LITERAL}:  [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    // CHECK-NEXT{LITERAL}:  [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    // CHECK-NEXT{LITERAL}:  [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    // CHECK-NEXT{LITERAL}:  [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],

    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [8, 16] strides = [16, 1] data =
    // CHECK-NEXT{LITERAL}: [[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    // CHECK-NEXT{LITERAL}:  [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    // CHECK-NEXT{LITERAL}:  [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    // CHECK-NEXT{LITERAL}:  [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    // CHECK-NEXT{LITERAL}:  [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],

    call @printMemrefI32(%res0_cast) : (memref<*xi32>) -> ()
    call @printMemrefI32(%res1_cast) : (memref<*xi32>) -> ()
    return
  }
  func.func private @printMemrefI32(memref<*xi32>) attributes {llvm.emit_c_interface}
}
