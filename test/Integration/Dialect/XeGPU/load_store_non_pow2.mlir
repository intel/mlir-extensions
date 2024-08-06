// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @loadstore attributes {gpu.container_module} {
  func.func @test(%A: memref<8x16xf32>, %B: memref<8x16xf32> ) -> (memref<8x16xf32>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<8x16xf32>
    %memref_1 = gpu.alloc  host_shared () : memref<8x16xf32>
    memref.copy %A, %memref : memref<8x16xf32> to memref<8x16xf32>
    memref.copy %B, %memref_1 : memref<8x16xf32> to memref<8x16xf32>
    gpu.launch_func  @module::@test blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8x16xf32>, %memref_1 : memref<8x16xf32>)
    gpu.dealloc  %memref : memref<8x16xf32>
    return %memref_1 : memref<8x16xf32>
  }

  gpu.module @module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test(%A: memref<8x16xf32>, %B: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      // load A tile
      %a_tile = xegpu.create_nd_tdesc %A [%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      %val_a = xegpu.load_nd %a_tile : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      // store to B tile
      %b_tile = xegpu.create_nd_tdesc %B [%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %val_a, %b_tile  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %cf_0 = arith.constant 0.0 : f16
    %cf_0_f32 = arith.constant 0.0 : f32
    %cf_2_f32 = arith.constant 2.0 : f32
    %cf_1 = arith.constant 1.0 : f16
    %A = memref.alloc() : memref<8x16xf32>
    %B = memref.alloc() : memref<8x16xf32>
    // TRY 8x15. While it can encode vector type to 120f32 for intrinsics, the result is wrong.

    // fill A with 2, B with 0
    %A_nonzero = memref.cast %A : memref<8x16xf32> to memref<*xf32>
    %B_zeros = memref.cast %B : memref<8x16xf32> to memref<*xf32>
    call @fillResource1DF32(%A_nonzero, %cf_2_f32) : (memref<*xf32>, f32) -> ()
    call @fillResource1DF32(%B_zeros, %cf_0_f32) : (memref<*xf32>, f32) -> ()
    // Load from A, store to B
    %2 = call @test(%A, %B) : (memref<8x16xf32>, memref<8x16xf32>) -> memref<8x16xf32>

    %B_filled = memref.cast %2 : memref<8x16xf32> to memref<*xf32>
    // call @printMemrefF32(%A_nonzero) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%B_filled) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%A_nonzero, %B_filled) : (memref<*xf32>, memref<*xf32>) -> ()

    memref.dealloc %A : memref<8x16xf32>
    memref.dealloc %B : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
}
