// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  memref.global "private" @__Aconstant_32x64xf16 : memref<32x64xf16> = dense<1.0>
  memref.global "private" @__Bconstant_32x64xf16 : memref<32x64xf16> = dense<2.0>
  func.func @test(%arg0: memref<32x64xf16>, %arg1: memref<32x64xf16>) -> memref<32x64xf32> attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_f32 = arith.constant 0.0 : f32


    %A = gpu.alloc  host_shared () : memref<32x64xf16>
    memref.copy %arg0, %A : memref<32x64xf16> to memref<32x64xf16>
    %B = gpu.alloc  host_shared () : memref<32x64xf16>
    memref.copy %arg1, %B : memref<32x64xf16> to memref<32x64xf16>

    %C = gpu.alloc  host_shared () : memref<32x64xf32>
    %C_unranked = memref.cast %C : memref<32x64xf32> to memref<*xf32>
    call @fillResource1DF32(%C_unranked, %c0_f32) : (memref<*xf32>, f32) -> ()

    %A_strided_dynamic = memref.subview %A[%c0, %c0][%c32, %c32][%c1, %c1] : memref<32x64xf16> to memref<?x?xf16, strided<[?,?], offset: ?>>
    %B_strided_dynamic = memref.subview %B[%c0, %c0][%c32, %c32][%c1, %c1] : memref<32x64xf16> to memref<?x?xf16, strided<[?,?], offset: ?>>
    %C_strided_dynamic = memref.subview %C[%c0, %c0][%c32, %c32][%c1, %c1] : memref<32x64xf32> to memref<?x?xf32, strided<[?,?], offset: ?>>

    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c4, %c2, %c1) threads in (%c1, %c1, %c1) args(%A_strided_dynamic : memref<?x?xf16, strided<[?,?], offset: ?>>, %B_strided_dynamic : memref<?x?xf16, strided<[?,?], offset: ?>>, %C_strided_dynamic : memref<?x?xf32, strided<[?,?], offset: ?>>, %c32 : index, %c32 : index, %c64 : index, %c1 : index)
    gpu.dealloc  %A : memref<32x64xf16>
    gpu.dealloc  %B : memref<32x64xf16>
    return %C : memref<32x64xf32>
  }

gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<?x?xf16, strided<[?,?], offset: ?>>, %B: memref<?x?xf16, strided<[?,?], offset: ?>>, %C: memref<?x?xf32, strided<[?,?], offset: ?>>, %shape_x : index, %shape_y : index, %stride_x : index, %stride_y : index) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 2, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %cst = arith.constant dense<1.0> : vector<8x16xf16>
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y

      %2 = arith.muli %0, %c8 : index
      %3 = arith.muli %1, %c16 : index

      %4 = xegpu.create_nd_tdesc %C[%2, %3], shape: [%shape_x, %shape_y], strides: [%stride_x, %stride_y] : memref<?x?xf32, strided<[?,?], offset: ?>> -> !xegpu.tensor_desc<8x16xf32>
      %5 = xegpu.load_nd %4 : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>

      %6 = scf.for %arg3 = %c0 to %c32 step %c16 iter_args(%arg4 = %5) -> (vector<8x16xf32>) {
        %A0 = xegpu.create_nd_tdesc %A[%2, %arg3], shape: [%shape_x, %shape_y], strides: [%stride_x, %stride_y] : memref<?x?xf16, strided<[?,?], offset: ?>> -> !xegpu.tensor_desc<8x16xf16>
        %A0_val = xegpu.load_nd %A0 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>

        %B0 = xegpu.create_nd_tdesc %B[%arg3, %3], shape: [%shape_x, %shape_y], strides: [%stride_x, %stride_y] : memref<?x?xf16, strided<[?,?], offset: ?>> -> !xegpu.tensor_desc<16x16xf16>
        %B0_val = xegpu.load_nd %B0 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>

        %A0_preop = arith.addf %A0_val, %cst : vector<8x16xf16>

        %dpas0 = xegpu.dpas %A0_preop, %B0_val , %arg4: vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        scf.yield %dpas0 : vector<8x16xf32>
      }
      xegpu.store_nd %6, %4 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>

      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    // Allocate/get regular row major memrefs
    %A = memref.get_global @__Aconstant_32x64xf16 : memref<32x64xf16>
    %B = memref.get_global @__Bconstant_32x64xf16 : memref<32x64xf16>

    %result = call @test(%A, %B) : (memref<32x64xf16>, memref<32x64xf16>) -> memref<32x64xf32>
    %result_cast = memref.cast %result : memref<32x64xf32> to memref<*xf32>
    call @printMemrefF32(%result_cast) : (memref<*xf32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT:[128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    return
  }
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
