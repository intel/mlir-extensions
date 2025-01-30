// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

#wg_map_a_coop = #xetile.wg_map<sg_layout = [1, 2], sg_data = [8, 16]>
#wg_map_a = #xetile.wg_map<sg_layout = [1, 2], sg_data = [8, 32]>
#wg_map_b = #xetile.wg_map<sg_layout = [1, 2], sg_data = [32, 16]>
#wg_map_c = #xetile.wg_map<sg_layout = [1, 2], sg_data = [8, 16]>

module @conv_layout attributes {gpu.container_module} {
  func.func @test_convert_layout_gemm(%a: memref<64x32xf16>, %b: memref<64x64xf16>) -> memref<32x64xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %a_gpu = gpu.alloc host_shared () : memref<64x32xf16>
    memref.copy %a, %a_gpu : memref<64x32xf16> to memref<64x32xf16>
    %b_gpu = gpu.alloc  host_shared () : memref<64x64xf16>
    memref.copy %b, %b_gpu : memref<64x64xf16> to memref<64x64xf16>
    %c_gpu = gpu.alloc  host_shared () : memref<32x64xf32>

    gpu.launch_func @kernel::@test_convert_layout_gemm blocks in (%c1, %c1, %c1) threads in (%c2, %c1, %c1) args(%a_gpu : memref<64x32xf16>, %b_gpu : memref<64x64xf16>, %c_gpu : memref<32x64xf32>)

    gpu.dealloc %a_gpu : memref<64x32xf16>
    gpu.dealloc %b_gpu : memref<64x64xf16>
    return %c_gpu : memref<32x64xf32>
  }

  gpu.module @kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    // this test performs a simple matrix multiplication on 64x32xf16 and 64x64xf16 with a workgroup of 2 threads, which resulting a 32x64xf32 matrix.
    // Each thread will compute 8x16xf32 matrix, which 64x32xf16 * 32x16xf16. a is shared, each thread will load 8x16xf16 from memory, and using convert
    // layout to share the data.
    gpu.func @test_convert_layout_gemm(%arg0 : memref<64x32xf16>, %arg1 : memref<64x64xf16>, %arg2 : memref<32x64xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c1 = arith.constant 1 : index
      %id_x = gpu.block_id x
      %id_y = gpu.block_id y
      %m = arith.muli %id_x, %c1 : index
      %n = arith.muli %id_y, %c1 : index

      %a_tile = xetile.init_tile %arg0[%m, %n] : memref<64x32xf16> -> !xetile.tile<64x32xf16, #xetile.tile_attr<wg_map = #wg_map_a_coop>>
      %a_coop = xetile.load_tile %a_tile: !xetile.tile<64x32xf16, #xetile.tile_attr<wg_map = #wg_map_a_coop>> -> vector<64x32xf16>
      %a = xetile.convert_layout %a_coop {wg_map_result = #wg_map_a, wg_map_source = #wg_map_a_coop} : vector<64x32xf16>

      %b_tile = xetile.init_tile %arg1[%m, %n] : memref<64x64xf16> -> !xetile.tile<64x64xf16, #xetile.tile_attr<wg_map = #wg_map_b>>
      %b = xetile.load_tile %b_tile: !xetile.tile<64x64xf16, #xetile.tile_attr<wg_map = #wg_map_b>> -> vector<64x64xf16>

      %c = xetile.tile_mma %a, %b {wg_map_a = #wg_map_a, wg_map_b = #wg_map_b, wg_map_c = #wg_map_c} : vector<64x32xf16>, vector<64x64xf16> -> vector<32x64xf32>

      %init_store_tile = xetile.init_tile %arg2[%m, %n] :  memref<32x64xf32> -> !xetile.tile<32x64xf32, #xetile.tile_attr<wg_map = #wg_map_c>>
      xetile.store_tile %c, %init_store_tile : vector<32x64xf32>, !xetile.tile<32x64xf32, #xetile.tile_attr<wg_map = #wg_map_c>>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index

    %c1_f16 = arith.constant 1.0 : f16
    %c2_f32 = arith.constant 2.0 : f32
    %c0_f32 = arith.constant 0.0 : f32
    %c100_f16 = arith.constant 100.0 : f16
    %a = memref.alloc() : memref<64x32xf16>
    %b = memref.alloc() : memref<64x64xf16>
    %c_ref = memref.alloc() : memref<32x64xf32>


    // intialize matrix A;
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        %m = arith.muli %i, %c32 : index
        %add = arith.addi %m, %j : index
        %t = index.castu %add : index to i16
        %v = arith.uitofp %t : i16 to f16
        %d = arith.divf %v, %c100_f16 : f16
        memref.store %d, %a[%i, %j] : memref<64x32xf16>
      }
    }

    // intialize matrix B;
    scf.for %i = %c0 to %c32 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        // %m = arith.muli %i, %c32 : index
        // %add = arith.addi %m, %j : index
        // %t = index.castu %add : index to i16
        // %v = arith.uitofp %t : i16 to f16
        // memref.store %v, %b[%i, %j] : memref<64x64xf16>
        memref.store %c1_f16, %b[%i, %j] : memref<64x64xf16>
      }
    }

    // intialize matrix c_ref;
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        memref.store %c0_f32, %c_ref[%i, %j] : memref<32x64xf32>
        scf.for %k = %c0 to %c32 step %c1 {
          %cv = memref.load %c_ref[%i, %j] : memref<32x64xf32>
          %av = memref.load %a[%i, %k] : memref<64x32xf16>
          %bv = memref.load %b[%k, %j] : memref<64x64xf16>

          %a_f32 = arith.extf %av : f16 to f32
          %b_f32 = arith.extf %bv : f16 to f32
          %m = arith.mulf %a_f32, %b_f32 : f32

          %acc = arith.addf %cv, %m : f32
          memref.store %acc, %c_ref[%i, %j] : memref<32x64xf32>
        }
      }
    }

    %c = call @test_convert_layout_gemm(%a, %b) : (memref<64x32xf16>, memref<64x64xf16>) -> memref<32x64xf32>
    %cast_c = memref.cast %c : memref<32x64xf32> to memref<*xf32>
    %cast_c_ref = memref.cast %c_ref :memref<32x64xf32> to memref<*xf32>
    call @printMemrefF32(%cast_c): (memref<*xf32>) -> ()
    call @printMemrefF32(%cast_c_ref): (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_c, %cast_c_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %a : memref<64x32xf16>
    memref.dealloc %b : memref<64x64xf16>
    memref.dealloc %c_ref : memref<32x64xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
