// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xetile-wg-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

// NOTES : This test load a tile from A, and then do a transpose on it,
// and store it back to B, using 16 threads in a workgroup. Each thread
// loads a 16x16 block from A, and transpose it. And then share the result
// with other threads via convert_layout. Finally each thread will store
// a 8x32 block to B.
module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<64x64xf16>) -> memref<64x64xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %A_gpu = gpu.alloc  host_shared () : memref<64x64xf16>
    memref.copy %A, %A_gpu : memref<64x64xf16> to memref<64x64xf16>
    %B_gpu = gpu.alloc  host_shared () : memref<64x64xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c16, %c1, %c1) args(%A_gpu : memref<64x64xf16>, %B_gpu : memref<64x64xf16>)
    gpu.dealloc  %A_gpu : memref<64x64xf16>
    return %B_gpu : memref<64x64xf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<64x64xf16>, %B: memref<64x64xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %a_tile = xetile.init_tile %A[%c0, %c0] : memref<64x64xf16> -> !xetile.tile<64x64xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [16, 16]>>>
      %data = xetile.load_tile %a_tile : !xetile.tile<64x64xf16, #xetile.tile_attr<wg_map = <sg_layout = [4, 4], sg_data = [16, 16]>>> -> vector<64x64xf16>

      %trans = xetile.transpose %data, [1, 0] {map = #xetile.wg_map<sg_layout = [4, 4], sg_data = [16, 16]>} : vector<64x64xf16> -> vector<64x64xf16>
      %cvt = xetile.convert_layout %trans {wg_map_result = #xetile.wg_map<sg_layout = [8, 2], sg_data = [8, 32]>} : vector<64x64xf16>

      %b_tile = xetile.init_tile %B[%c0, %c0] : memref<64x64xf16> -> !xetile.tile<64x64xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 2], sg_data = [8, 32]>>>
      xetile.store_tile %cvt, %b_tile: vector<64x64xf16>, !xetile.tile<64x64xf16, #xetile.tile_attr<wg_map = <sg_layout = [8, 2], sg_data = [8, 32]>>>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %cf_0 = arith.constant 0.0 : f16
    %cf_1 = arith.constant 1.0 : f16
    %A = memref.alloc() : memref<64x64xf16>
    %Ref = memref.alloc() : memref<64x64xf32>

    // intialize matrix A ; A[i, j] = j
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        // %mul = arith.muli %i, %c64 : index
        // %add = arith.addi %mul, %j : index
        // %t = index.castu %add : index to i16
        %t = index.castu %j : index to i16
        %val = arith.uitofp %t : i16 to f16
        memref.store %val, %A[%i, %j] : memref<64x64xf16>
        %val32 = arith.extf %val : f16 to f32
        memref.store %val32, %Ref[%j, %i] : memref<64x64xf32>
      }
    }
    %B = call @test(%A) : (memref<64x64xf16>) -> memref<64x64xf16>
    %cast = memref.cast %B : memref<64x64xf16> to memref<*xf16>
    // call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %cast_ref = memref.cast %Ref : memref<64x64xf32> to memref<*xf32>
    call @printAllcloseF16(%cast, %cast_ref) : (memref<*xf16>, memref<*xf32>) -> ()
    memref.dealloc %A : memref<64x64xf16>
    memref.dealloc %Ref : memref<64x64xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
