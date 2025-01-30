// RUN: IMEX_USE_IGC_VECTOR_BACK_END=1 %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: IMEX_USE_IGC_VECTOR_BACK_END=1 %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xetile-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

// NOTES :
// This example assumes one subgroup per one workgroup and the kernel specifies the computation
// done by a single subgroup.

module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<1024xf32>, %B: memref<1024xf32>) -> memref<1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %A_gpu = gpu.alloc  host_shared () : memref<1024xf32>
    memref.copy %A, %A_gpu : memref<1024xf32> to memref<1024xf32>
    %B_gpu = gpu.alloc  host_shared () : memref<1024xf32>
    memref.copy %B, %B_gpu : memref<1024xf32> to memref<1024xf32>
    %C_gpu = gpu.alloc  host_shared () : memref<1024xf32>
    gpu.launch_func  @test_kernel::@add_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%A_gpu : memref<1024xf32>, %B_gpu : memref<1024xf32>, %C_gpu : memref<1024xf32>)
    gpu.dealloc  %B_gpu : memref<1024xf32>
    return %A_gpu : memref<1024xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL, Bfloat16ConversionINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute, SPV_INTEL_bfloat16_conversion]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @add_kernel(%A: memref<1024xf32>, %B: memref<1024xf32>, %C: memref<1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %indices = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]>: vector<1x32xindex>
      %offsets = arith.constant dense<32>: vector<1x32xindex>
      %mask = arith.constant dense<true>: vector<1x32xi1>

      %a_init_tile = xetile.init_tile %A, %indices : memref<1024xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
      %b_init_tile = xetile.init_tile %B, %indices : memref<1024xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
      %c_init_tile = xetile.init_tile %C, %indices : memref<1024xf32>, vector<1x32xindex> -> !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>

      %out:3 = scf.for %k = %c0 to %c1024 step %c32
        iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_tile = %c_init_tile)
        -> (!xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>) {

        // load A and B tiles
        %b_value = xetile.load %b_tile, %mask : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1> -> vector<1x32xf32>
        %c_value = xetile.atomic_rmw addf %b_value, %a_tile : vector<1x32xf32>, !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>> -> vector<1x32xf32>

        xetile.store %c_value, %c_tile, %mask : vector<1x32xf32>, !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xi1>

        %a_next_tile = xetile.update_tile_offset %a_tile, %offsets : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xindex>
        %b_next_tile = xetile.update_tile_offset %b_tile, %offsets : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xindex>
        %c_next_tile = xetile.update_tile_offset %c_tile, %offsets : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, vector<1x32xindex>

        scf.yield %a_next_tile, %b_next_tile, %c_next_tile
          : !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>, !xetile.tile<1x32xf32, #xetile.tile_attr<scattered = true>>
      }
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cf_0 = arith.constant 0.0 : bf16
    %cf_1 = arith.constant 1.0 : bf16
    %A = memref.alloc() : memref<1024xf32>
    %B = memref.alloc() : memref<1024xf32>
    %C_ref = memref.alloc() : memref<1024xf32>
    // intialize matrix A ;
    scf.for %i = %c0 to %c1024 step %c1 {
      %t = index.castu %i : index to i32
      %val = arith.uitofp %t : i32 to f32
      memref.store %val, %A[%i] : memref<1024xf32>
      memref.store %val, %B[%i] : memref<1024xf32>
    }

    // compute C for reference
    scf.for %i = %c0 to %c1024 step %c1 {
      %a_val = memref.load %A[%i] : memref<1024xf32>
      %b_val = memref.load %B[%i] : memref<1024xf32>
      %c_val = arith.addf %a_val, %b_val : f32
      memref.store %c_val, %C_ref[%i] : memref<1024xf32>
    }
    %2 = call @test(%A, %B) : (memref<1024xf32>, memref<1024xf32>) -> memref<1024xf32>
    %cast_C = memref.cast %2 : memref<1024xf32> to memref<*xf32>
    %cast_C_ref = memref.cast %C_ref : memref<1024xf32> to memref<*xf32>
    //call @printMemrefF32(%cast_C) : (memref<*xf32>) -> ()
    //call @printMemrefF32(%cast_C_ref) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_C, %cast_C_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<1024xf32>
    memref.dealloc %B : memref<1024xf32>
    memref.dealloc %C_ref : memref<1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
