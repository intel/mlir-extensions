// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  memref.global "private" @__Aconstant_8x32xf32 : memref<8x32xf32> = dense<1.000000e+00>
  memref.global "private" @__Bconstant_8x32xf32 : memref<8x32xf32> = dense<2.000000e+00>
  func.func @test(%arg0: memref<8x32xf32>, %arg1: memref<8x32xf32>) -> memref<8x32xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    // Create the strided memrefs from A, B, C : first 16 elements of each row
    %cst = arith.constant 0.000000e+00 : f32
    %memref = gpu.alloc  () : memref<8x32xf32>
    gpu.memcpy  %memref, %arg0 : memref<8x32xf32>, memref<8x32xf32>
    %memref_0 = gpu.alloc  () : memref<8x32xf32>
    gpu.memcpy  %memref_0, %arg1 : memref<8x32xf32>, memref<8x32xf32>
    %memref_host = memref.alloc() : memref<8x32xf32>
    %cast_host = memref.cast %memref_host : memref<8x32xf32> to memref<*xf32>
    call @fillResource1DF32(%cast_host, %cst) : (memref<*xf32>, f32) -> ()
    %memref_1 = gpu.alloc  () : memref<8x32xf32>
    gpu.memcpy  %memref_1, %memref_host : memref<8x32xf32>, memref<8x32xf32>
    %subview = memref.subview %memref[0, 0] [8, 16] [1, 1] : memref<8x32xf32> to memref<8x16xf32, strided<[32, 1]>>
    %subview_2 = memref.subview %memref_0[0, 0] [8, 16] [1, 1] : memref<8x32xf32> to memref<8x16xf32, strided<[32, 1]>>
    %subview_3 = memref.subview %memref_1[0, 0] [8, 16] [1, 1] : memref<8x32xf32> to memref<8x16xf32, strided<[32, 1]>>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c8, %c1, %c1)  args(%subview : memref<8x16xf32, strided<[32, 1]>>, %subview_2 : memref<8x16xf32, strided<[32, 1]>>, %subview_3 : memref<8x16xf32, strided<[32, 1]>>)
    memref.dealloc  %memref_host : memref<8x32xf32>
    gpu.dealloc  %memref : memref<8x32xf32>
    gpu.dealloc  %memref_0 : memref<8x32xf32>
    %alloc = memref.alloc() : memref<8x32xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<8x32xf32>, memref<8x32xf32>
    gpu.dealloc  %memref_1 : memref<8x32xf32>
    return %alloc : memref<8x32xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<8x16xf32, strided<[32, 1]>>, %arg1: memref<8x16xf32, strided<[32, 1]>>, %arg2: memref<8x16xf32, strided<[32, 1]>>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %thread_id_x = gpu.thread_id  x
      %0 = xegpu.create_nd_tdesc %arg0[%thread_id_x, 0] : memref<8x16xf32, strided<[32, 1]>> -> !xegpu.tensor_desc<16xf32>
      %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
      %2 = xegpu.create_nd_tdesc %arg1[%thread_id_x, 0] : memref<8x16xf32, strided<[32, 1]>> -> !xegpu.tensor_desc<16xf32>
      %3 = xegpu.load_nd %2  : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
      %4 = arith.addf %3, %1 : vector<16xf32>
      %5 = xegpu.create_nd_tdesc %arg2[%thread_id_x, 0] : memref<8x16xf32, strided<[32, 1]>> -> !xegpu.tensor_desc<16xf32>
      xegpu.store_nd %4, %5  : vector<16xf32>, !xegpu.tensor_desc<16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // Allocate/get regular row major memrefs
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT:[3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    %0 = memref.get_global @__Aconstant_8x32xf32 : memref<8x32xf32>
    %1 = memref.get_global @__Bconstant_8x32xf32 : memref<8x32xf32>
    %2 = call @test(%0, %1) : (memref<8x32xf32>, memref<8x32xf32>) -> memref<8x32xf32>
    %cast = memref.cast %2 : memref<8x32xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}

