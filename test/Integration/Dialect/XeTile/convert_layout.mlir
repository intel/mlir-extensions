// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

module @conv_layout attributes {gpu.container_module} {
  func.func @convert_layout(%a: memref<64x64xf32>, %b: memref<64x64xf32>) -> memref<64x64xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index

    %a_gpu = gpu.alloc host_shared () : memref<64x64xf32>
    memref.copy %a, %a_gpu : memref<64x64xf32> to memref<64x64xf32>
    %b_gpu = gpu.alloc  host_shared () : memref<64x64xf32>
    memref.copy %b, %b_gpu : memref<64x64xf32> to memref<64x64xf32>
    %c_gpu = gpu.alloc  host_shared () : memref<64x64xf32>

    gpu.launch_func @kernel::@test_convert_layout blocks in (%c1, %c1, %c1) threads in (%c8, %c4, %c1) args(%a_gpu : memref<64x64xf32>, %b_gpu : memref<64x64xf32>, %c_gpu : memref<64x64xf32>)

    gpu.dealloc %a_gpu : memref<64x64xf32>
    gpu.dealloc %b_gpu : memref<64x64xf32>
    return %c_gpu : memref<64x64xf32>
  }

gpu.module @kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @test_convert_layout(%arg0 : memref<64x64xf32>, %arg1 : memref<64x64xf32>, %arg2 : memref<64x64xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c1 = arith.constant 1 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c1 : index
    %n = arith.muli %block_id_y, %c1 : index
    %init_tile_1 = xetile.init_tile %arg0[%m, %n] : memref<64x64xf32> -> !xetile.tile<64x64xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [8, 16]>>>
    %load_tile_1 = xetile.load_tile %init_tile_1: !xetile.tile<64x64xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [8, 16]>>> -> vector<64x64xf32>
    %init_tile_2 = xetile.init_tile %arg1[%m, %n] : memref<64x64xf32> -> !xetile.tile<64x64xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [16, 8]>>>
    %load_tile_2 = xetile.load_tile %init_tile_2: !xetile.tile<64x64xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [16, 8]>>> -> vector<64x64xf32>
    %convert_layout = xetile.convert_layout %load_tile_1 {wg_map_result = #xetile.wg_map<sg_layout = [4, 8], sg_data = [16, 8]>, wg_map_source = #xetile.wg_map<sg_layout = [8, 4], sg_data = [8, 16]>} : vector<64x64xf32>
    %add = arith.addf %load_tile_2, %convert_layout {map = #xetile.wg_map<sg_layout = [4, 8], sg_data = [16, 8]>} : vector<64x64xf32>
    %init_store_tile = xetile.init_tile %arg2[%m, %n] :  memref<64x64xf32> -> !xetile.tile<64x64xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [16, 8]>>>
    xetile.store_tile  %add, %init_store_tile : vector<64x64xf32>, !xetile.tile<64x64xf32, #xetile.tile_attr<wg_map = <sg_layout = [4, 8], sg_data = [16, 8]>>>
    gpu.return
  }
}

func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c1_f32 = arith.constant 1.0 : f32
    %c2_f32 = arith.constant 2.0 : f32
    %c100_f32 = arith.constant 100.0 : f32
    %a = memref.alloc() : memref<64x64xf32>
    %b = memref.alloc() : memref<64x64xf32>
    %c_ref = memref.alloc() : memref<64x64xf32>


    // intialize matrix A, B ; A[i, j] = 1
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        memref.store %c1_f32, %a[%i, %j] : memref<64x64xf32>
        memref.store %c1_f32, %b[%i, %j] : memref<64x64xf32>
        memref.store %c2_f32, %c_ref[%i, %j] : memref<64x64xf32>
      }
    }

    %c = call @convert_layout(%a, %b) : (memref<64x64xf32>, memref<64x64xf32>) -> memref<64x64xf32>
    %cast_c = memref.cast %c : memref<64x64xf32> to memref<*xf32>
    %cast_c_ref = memref.cast %c_ref :memref<64x64xf32> to memref<*xf32>
    //call @printMemrefF32(%cast_c): (memref<*xf32>) -> ()
    //call @printMemrefF32(%cast_c_ref): (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_c, %cast_c_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %a : memref<64x64xf32>
    memref.dealloc %b : memref<64x64xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
