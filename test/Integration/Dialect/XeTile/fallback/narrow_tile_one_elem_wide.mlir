// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xetile-fallback-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xetile-fallback-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

module @narrow_tile attributes {gpu.container_module} {
  func.func @test(%A: memref<64x1xf32>) -> memref<64x1xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %A_gpu = gpu.alloc  host_shared() : memref<64x1xf32>
    memref.copy %A, %A_gpu : memref<64x1xf32> to memref<64x1xf32>
    %B_gpu = gpu.alloc  host_shared() : memref<64x1xf32>
    gpu.launch_func @test_module::@test_scf_for blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%A_gpu : memref<64x1xf32>, %B_gpu : memref<64x1xf32>)
    %B = memref.alloc() : memref<64x1xf32>
    memref.copy %B_gpu, %B : memref<64x1xf32> to memref<64x1xf32>
    gpu.dealloc %A_gpu : memref<64x1xf32>
    gpu.dealloc %B_gpu : memref<64x1xf32>
    return %B : memref<64x1xf32>
  }
  gpu.module @test_module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Bfloat16ConversionINTEL, BFloat16TypeKHR, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, VectorAnyINTEL, VectorComputeINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_bfloat16, SPV_KHR_expect_assume, SPV_INTEL_bfloat16_conversion, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_scf_for(%arg0: memref<64x1xf32>, %arg1: memref<64x1xf32>) kernel attributes {VectorComputeFunctionINTEL, known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst0 = arith.constant 0 : index
      %cst16 = arith.constant 16 : index
      %cst64 = arith.constant 64 : index
      %0 = xetile.init_tile %arg0 [0, 0] : memref<64x1xf32> -> !xetile.tile<16x1xf32, #xetile.tile_attr<order = [1, 0]>>
      %1 = xetile.init_tile %arg1 [0, 0] : memref<64x1xf32> -> !xetile.tile<16x1xf32, #xetile.tile_attr<order = [1, 0]>>
      %out:2 = scf.for %k = %cst0 to %cst64 step %cst16
        iter_args(%a_tile = %0, %b_tile = %1)
        -> (!xetile.tile<16x1xf32, #xetile.tile_attr<order = [1, 0]>>, !xetile.tile<16x1xf32, #xetile.tile_attr<order = [1, 0]>>) {
        %a_value = xetile.load_tile %a_tile : !xetile.tile<16x1xf32, #xetile.tile_attr<order = [1, 0]>> -> vector<16x1xf32>
        xetile.store_tile %a_value, %b_tile : vector<16x1xf32>, !xetile.tile<16x1xf32, #xetile.tile_attr<order = [1, 0]>>
        %a_next_tile = xetile.update_tile_offset %a_tile, [%cst16, %cst0] : !xetile.tile<16x1xf32, #xetile.tile_attr<order = [1, 0]>>
        %b_next_tile = xetile.update_tile_offset %b_tile, [%cst16, %cst0] : !xetile.tile<16x1xf32, #xetile.tile_attr<order = [1, 0]>>
        scf.yield %a_next_tile, %b_next_tile : !xetile.tile<16x1xf32, #xetile.tile_attr<order = [1, 0]>>, !xetile.tile<16x1xf32, #xetile.tile_attr<order = [1, 0]>>
      }
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    %A = memref.alloc() : memref<64x1xf32>
    scf.for %arg0 = %c0 to %c64 step %c1 {
      %0 = index.castu %arg0 : index to i32
      %val = arith.uitofp %0 : i32 to f32
      memref.store %val, %A[%arg0, %c0] : memref<64x1xf32>
    }
    %C = call @test(%A) : (memref<64x1xf32>) -> memref<64x1xf32>
    %cast_A = memref.cast %A : memref<64x1xf32> to memref<*xf32>
    %cast_C = memref.cast %C : memref<64x1xf32> to memref<*xf32>
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_C, %cast_A) : (memref<*xf32>, memref<*xf32>) -> ()
    //call @printMemrefF32(%cast_A) : (memref<*xf32>) -> ()
    //call @printMemrefF32(%cast_C) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  //func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
