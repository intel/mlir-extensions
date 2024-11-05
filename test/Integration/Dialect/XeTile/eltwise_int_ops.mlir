// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @eltwise_int attributes {gpu.container_module} {
  memref.global "private" constant @__constant_5_1024x1024xi32 : memref<1024x1024xi32> = dense<5>
  memref.global "private" constant @__constant_2_1024x1024xi32 : memref<1024x1024xi32> = dense<2>

  func.func @eltwise_int_test(%arg0: memref<1024x1024xi32>, %arg1: memref<1024x1024xi32>) -> memref<1024x1024xi32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index

    %arg0_gpu = gpu.alloc host_shared () : memref<1024x1024xi32>
    memref.copy %arg0, %arg0_gpu : memref<1024x1024xi32> to memref<1024x1024xi32>

    %arg1_gpu = gpu.alloc host_shared () : memref<1024x1024xi32>
    memref.copy %arg1, %arg1_gpu : memref<1024x1024xi32> to memref<1024x1024xi32>

    %result = gpu.alloc host_shared () : memref<1024x1024xi32>

    gpu.launch_func  @eltwise_int::@eltwise_int blocks in (%c64, %c32, %c1) threads in (%c1, %c1, %c1)  args(%arg0_gpu : memref<1024x1024xi32>, %arg1_gpu : memref<1024x1024xi32>, %result : memref<1024x1024xi32>)

    gpu.dealloc %arg0_gpu : memref<1024x1024xi32>
    gpu.dealloc %arg1_gpu : memref<1024x1024xi32>
    return %result : memref<1024x1024xi32>

  }

  gpu.module @eltwise_int attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Bfloat16ConversionINTEL, BFloat16TypeKHR, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL, VectorComputeINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_bfloat16, SPV_KHR_expect_assume, SPV_INTEL_bfloat16_conversion, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @eltwise_int(%arg0: memref<1024x1024xi32>, %arg1: memref<1024x1024xi32>, %arg2: memref<1024x1024xi32>) kernel attributes {VectorComputeFunctionINTEL, known_block_size = array<i32: 1, 32, 1>, known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index

      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y

      %m = arith.muli %block_id_x, %c16 : index
      %n = arith.muli %block_id_y, %c32 : index

      %1 = xetile.init_tile %arg0[%m, %n] : memref<1024x1024xi32> -> !xetile.tile<16x32xi32>
      %2 = xetile.load_tile %1: !xetile.tile<16x32xi32> -> vector<16x32xi32>
      %3 = xetile.init_tile %arg1[%m, %n] : memref<1024x1024xi32> -> !xetile.tile<16x32xi32>
      %4 = xetile.load_tile %3: !xetile.tile<16x32xi32> -> vector<16x32xi32>
      %result_add = arith.addi %2, %4: vector<16x32xi32> //=7
      %result_sub = arith.subi %2, %4: vector<16x32xi32> //=3
      %result_mul = arith.muli %result_add, %result_sub: vector<16x32xi32> //=21
      %result_sdiv = arith.divsi %result_mul, %result_add: vector<16x32xi32> //=3
      %result_udiv = arith.divui %result_mul, %result_add: vector<16x32xi32> //=3
      %result_srem = arith.remsi %result_sdiv, %result_mul: vector<16x32xi32> //=3
      %result_urem = arith.remui %result_udiv, %result_srem: vector<16x32xi32> //=0
      %result = arith.addi %result_srem, %result_urem: vector<16x32xi32> //=3
      %store_tile = xetile.init_tile %arg2[%m, %n] : memref<1024x1024xi32> -> !xetile.tile<16x32xi32>
      xetile.store_tile %result, %store_tile: vector<16x32xi32>, !xetile.tile<16x32xi32>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.get_global @__constant_5_1024x1024xi32 : memref<1024x1024xi32>
    %B = memref.get_global @__constant_2_1024x1024xi32 : memref<1024x1024xi32>

    %c0_i32 = arith.constant 0 : i32

    %result = call @eltwise_int_test(%A, %B) : (memref<1024x1024xi32>, memref<1024x1024xi32>) -> memref<1024x1024xi32>
    %result_cast = memref.cast %result : memref<1024x1024xi32> to memref<*xi32>
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-COUNT-1048576: 3
    call @printMemrefI32(%result_cast) : (memref<*xi32>) -> ()

    return
  }
  func.func private @printMemrefI32(memref<*xi32>) attributes {llvm.emit_c_interface}
}
