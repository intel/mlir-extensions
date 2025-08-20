// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

module @reduction attributes {gpu.container_module} {
  func.func @reduce_test(%a: memref<256x1024xf32>) -> memref<1x1024xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index

    %a_gpu = gpu.alloc host_shared () : memref<256x1024xf32>
    memref.copy %a, %a_gpu : memref<256x1024xf32> to memref<256x1024xf32>
    %b_gpu = gpu.alloc  host_shared () : memref<1x1024xf32>

    gpu.launch_func @kernel::@test_reduction blocks in (%c1, %c8, %c1) threads in (%c8, %c4, %c1) args(%a_gpu : memref<256x1024xf32>, %b_gpu : memref<1x1024xf32>)

    gpu.dealloc %a_gpu : memref<256x1024xf32>
    return %b_gpu : memref<1x1024xf32>
  }

gpu.module @kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  gpu.func @test_reduction(%arg0 : memref<256x1024xf32>, %arg1 : memref<1x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %m = arith.muli %block_id_x, %c256 : index
    %n = arith.muli %block_id_y, %c128 : index
    %init_tile = xetile.init_tile %arg0[%m, %n] : memref<256x1024xf32> -> !xetile.tile<256x128xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>>
    %load_tile = xetile.load_tile %init_tile: !xetile.tile<256x128xf32, #xetile.tile_attr<wg_map = <sg_layout = [8, 4], sg_data = [32, 32]>>> -> vector<256x128xf32>
    %cst_0 = arith.constant {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [1, 32]>} dense<0.0> : vector<8x128xf32>
    %reshape = vector.shape_cast %load_tile {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [32, 32]>} : vector<256x128xf32> to vector<8x32x128xf32>
    %reduction = vector.multi_reduction <add>, %reshape, %cst_0 {map = #xetile.wg_map<sg_layout = [8, 4], sg_data = [1, 32]>} [1] : vector<8x32x128xf32> to vector<8x128xf32>
    %conv_layout = xetile.convert_layout %reduction {wg_map_result = #xetile.wg_map<sg_layout = [1, 32], sg_data = [8, 4]>} : vector<8x128xf32>
    %cst_1 = arith.constant {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} dense<0.0> : vector<128xf32>
    %reduce = vector.multi_reduction <add>, %conv_layout, %cst_1 {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} [0] : vector<8x128xf32> to vector<128xf32>
    %shape_cast = vector.shape_cast %reduce {map = #xetile.wg_map<sg_layout = [1, 32], sg_data = [1, 4]>} : vector<128xf32> to vector<1x128xf32>
    %init_store_tile = xetile.init_tile %arg1[%c0, %n] :  memref<1x1024xf32> -> !xetile.tile<1x128xf32, #xetile.tile_attr<wg_map = <sg_layout = [1, 32], sg_data = [1, 4]>>>
    xetile.store_tile  %shape_cast, %init_store_tile : vector<1x128xf32>, !xetile.tile<1x128xf32, #xetile.tile_attr<wg_map = <sg_layout = [1, 32], sg_data = [1, 4]>>>
    gpu.return
  }
}

func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %c0_f32 = arith.constant 0.0 : f32
    %c32_f32 = arith.constant 32.0 : f32
    %c1_f32 = arith.constant 1.0 : f32
    %c100_f32 = arith.constant 100.0 : f32
    %a = memref.alloc() : memref<256x1024xf32>
    %b_ref = memref.alloc() : memref<1024xf32>


    // intialize matrix A ; A[i, j] = 1
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c1024 step %c1 {
        memref.store %c1_f32, %a[%i, %j] : memref<256x1024xf32>
      }
    }

    scf.for %j = %c0 to %c1024 step %c1 {
      %sum = scf.for %i = %c0 to %c256 step %c1 iter_args(%arg = %c0_f32) -> (f32) {
        %val = memref.load %a[%i, %j] : memref<256x1024xf32>
        %2 = arith.addf %arg, %val : f32
        scf.yield %2 : f32
      }
      memref.store %sum, %b_ref[%j] : memref<1024xf32>
    }

    %b = call @reduce_test(%a) : (memref<256x1024xf32>) -> memref<1x1024xf32>
    %cast_b = memref.cast %b : memref<1x1024xf32> to memref<*xf32>
    %cast_b_ref = memref.cast %b_ref : memref<1024xf32> to memref<*xf32>
    //call @printMemrefF32(%cast_b): (memref<*xf32>) -> ()
    //call @printMemrefF32(%cast_b_ref): (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%cast_b, %cast_b_ref) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %a : memref<256x1024xf32>
    memref.dealloc %b_ref : memref<1024xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
