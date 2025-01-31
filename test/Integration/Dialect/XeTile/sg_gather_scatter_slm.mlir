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
  // a test case case return the transpose of A, which is viewed as memref<32x32xf16>.
  // it uses one workgroup containing 32 subgroups, organized as (8x4), so each subgroup
  // works on a 4x8 tile of A. It used SLM to do the transpose, to evaluate the functionality
  // of the SLM operations.
  func.func @test(%A: memref<32x32xf16>) -> memref<32x32xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %A_gpu = gpu.alloc  host_shared () : memref<32x32xf16>
    memref.copy %A, %A_gpu : memref<32x32xf16> to memref<32x32xf16>
    %B_gpu = gpu.alloc  host_shared () : memref<32x32xf16>
    gpu.launch_func  @test_kernel::@trans_kernel blocks in (%c1, %c1, %c1) threads in (%c4, %c8, %c1) args(%A_gpu : memref<32x32xf16>, %B_gpu : memref<32x32xf16>)
    gpu.dealloc  %A_gpu : memref<32x32xf16>
    return %B_gpu : memref<32x32xf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL, Bfloat16ConversionINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute, SPV_INTEL_bfloat16_conversion]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @trans_kernel(%A: memref<32x32xf16>, %B: memref<32x32xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index

      %sgid = gpu.subgroup_id : index
      // %tid_y = arith.divui %sgid, %c4 : index
      // %tid_x = arith.remui %sgid, %c4 : index
      %tid_y = arith.shrui %sgid, %c2 : index
      %tid_x = arith.andi %sgid, %c3 : index

      %off_y = arith.muli %tid_y, %c4 : index
      %off_x = arith.muli %tid_x, %c8 : index

      // load data from global memory using block load
      %a_tile = xetile.init_tile %A[%off_y, %off_x] : memref<32x32xf16> -> !xetile.tile<4x8xf16>
      %data = xetile.load_tile %a_tile : !xetile.tile<4x8xf16> -> vector<4x8xf16>

      // %slm = memref.alloc() : memref<32x32xf16, 3>
      // %cast = memref.reinterpret_cast %slm to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf16, 3> to memref<1024xf16, 3>

      %slm = memref.alloc() : memref<1024xf16, 3>

      %mask = arith.constant dense<true>: vector<4x8xi1>

      // store data to slm using original layout
      %base_indices = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7],
                                            [32, 33, 34, 35, 36, 37, 38, 39],
                                            [64, 65, 66, 67, 68, 69, 70, 71],
                                            [96, 97, 98, 99, 100, 101, 102, 103]]>: vector<4x8xindex>
      %off_y2 = arith.muli %tid_y, %c128 : index
      %offset = arith.addi %off_y2, %off_x : index
      %offsets = vector.splat %offset: vector<4x8xindex>
      %indices = arith.addi %base_indices, %offsets : vector<4x8xindex>
      %st_tile = xetile.init_tile %slm, %indices : memref<1024xf16, 3>, vector<4x8xindex> -> !xetile.tile<4x8xf16, #xetile.tile_attr<scattered = true, memory_space=3>>
      xetile.store %data, %st_tile, %mask : vector<4x8xf16>, !xetile.tile<4x8xf16, #xetile.tile_attr<scattered = true, memory_space=3>>, vector<4x8xi1>

      gpu.barrier

      // load data from slm using indices with transpose effects
      %trans_base_indices = arith.constant dense<[[0, 32, 64, 96, 128, 160, 192, 224],
                                                  [1, 33, 65, 97, 129, 161, 193, 225],
                                                  [2, 34, 66, 98, 130, 162, 194, 226],
                                                  [3, 35, 67, 99, 131, 163, 195, 227]]>: vector<4x8xindex>

      %trans_off_x = arith.muli %tid_x, %c256 : index
      %trans_off_y = arith.muli %tid_y, %c4 : index
      %trans_off = arith.addi %trans_off_x, %trans_off_y : index
      %trans_offsets = vector.splat %trans_off: vector<4x8xindex>
      %trans_indices = arith.addi %trans_base_indices, %trans_offsets : vector<4x8xindex>
      %ld_tile = xetile.init_tile %slm, %trans_indices : memref<1024xf16, 3>, vector<4x8xindex> -> !xetile.tile<4x8xf16, #xetile.tile_attr<scattered = true, memory_space=3>>
      %d = xetile.load %ld_tile, %mask : !xetile.tile<4x8xf16, #xetile.tile_attr<scattered = true, memory_space=3>>, vector<4x8xi1> -> vector<4x8xf16>

      %b_tile = xetile.init_tile %B[%off_y, %off_x] : memref<32x32xf16> -> !xetile.tile<4x8xf16>
      xetile.store_tile %d, %b_tile: vector<4x8xf16>, !xetile.tile<4x8xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %cf_0 = arith.constant 0.0 : bf16
    %cf_1 = arith.constant 1.0 : bf16
    %A = memref.alloc() : memref<32x32xf16>
    %Ref = memref.alloc() : memref<32x32xf32>
    // intialize matrix A ;
    scf.for %i = %c0 to %c32 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        %m = arith.muli %i, %c32 : index
        %a = arith.addi %m, %j : index
        %v = index.castu %a : index to i16
        %val = arith.uitofp %v : i16 to f16
        memref.store %val, %A[%i, %j] : memref<32x32xf16>
        %v32 = index.castu %a : index to i32
        %val32 = arith.uitofp %v32 : i32 to f32
        memref.store %val32, %Ref[%j, %i] : memref<32x32xf32>
      }
    }
    %B = call @test(%A) : (memref<32x32xf16>) -> memref<32x32xf16>
    %cast = memref.cast %B : memref<32x32xf16> to memref<*xf16>
    %Ref_cast = memref.cast %Ref : memref<32x32xf32> to memref<*xf32>
    //CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF16(%cast, %Ref_cast) : (memref<*xf16>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<32x32xf16>
    memref.dealloc %Ref : memref<32x32xf32>
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
