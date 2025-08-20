// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @transpose attributes {gpu.container_module} {
  func.func @test(%arg0: memref<32x32xf16>) -> memref<32x32xf16> attributes {llvm.emit_c_interface} {
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<32x32xf16>
    memref.copy %arg0, %memref : memref<32x32xf16> to memref<32x32xf16>
    %B = gpu.alloc  host_shared () : memref<32x32xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c4, %c2, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<32x32xf16>, %B : memref<32x32xf16>)
    gpu.dealloc  %memref : memref<32x32xf16>
    return %B : memref<32x32xf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<32x32xf16>, %arg1: memref<32x32xf16>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 2, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = arith.muli %0, %c8 : index
      %3 = arith.muli %1, %c16 : index
      %4 = xegpu.create_nd_tdesc %arg0[%2, %3] : memref<32x32xf16> -> !xegpu.tensor_desc<8x16xf16>
      %5 = xegpu.load_nd %4 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %6 = xegpu.create_nd_tdesc %arg1[%3, %2] : memref<32x32xf16> -> !xegpu.tensor_desc<16x8xf16>
      %7 = vector.transpose %5, [1, 0]: vector<8x16xf16> to vector<16x8xf16>
      xegpu.store_nd %7, %6 : vector<16x8xf16>, !xegpu.tensor_desc<16x8xf16>
      gpu.return
    }
  }


  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.alloc() : memref<32x32xf16>
    %ref = memref.alloc() : memref<32x32xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index

    // A matrix: row-major, start from 0.0, increase 0.01 per element
    // B matrix: A matrix + 1.0
    scf.for %i = %c0 to %c32 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        %int = arith.index_cast %j : index to i32
        %fp = arith.uitofp %int : i32 to f16
        memref.store %fp, %0[%i, %j] : memref<32x32xf16>
        %fp_32 = arith.extf %fp : f16 to f32
        memref.store %fp_32, %ref[%j, %i] : memref<32x32xf32>
      }
    }

    %2 = call @test(%0) : (memref<32x32xf16>) -> memref<32x32xf16>
    %res = memref.cast %2 : memref<32x32xf16> to memref<*xf16>
    %cast_ref = memref.cast %ref : memref<32x32xf32> to memref<*xf32>
    // CHECK:   [ALLCLOSE: TRUE]
    call @printAllcloseF16(%res, %cast_ref) : (memref<*xf16>, memref<*xf32>) -> ()
    return
  }
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
