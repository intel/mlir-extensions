// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  memref.global "private" @__Aconstant_32x64xf16 : memref<32x64xf16> = dense<1.000000e+00>
  memref.global "private" @__Bconstant_32x64xf16 : memref<32x64xf16> = dense<2.000000e+00>
  func.func @test(%arg0: memref<32x64xf16>, %arg1: memref<32x64xf16>) -> memref<32x64xf32> attributes {llvm.emit_c_interface} {
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    // Create the strided memrefs from A, B, C : first 32 elements of each row
    %memref = gpu.alloc  () : memref<32x64xf16>
    gpu.memcpy  %memref, %arg0 : memref<32x64xf16>, memref<32x64xf16>
    %memref_0 = gpu.alloc  () : memref<32x64xf16>
    gpu.memcpy  %memref_0, %arg1 : memref<32x64xf16>, memref<32x64xf16>
    %memref_host = memref.alloc() : memref<32x64xf32>
    %cast_host = memref.cast %memref_host : memref<32x64xf32> to memref<*xf32>
    call @fillResource1DF32(%cast_host, %cst) : (memref<*xf32>, f32) -> ()
    %memref_1 = gpu.alloc  () : memref<32x64xf32>
    gpu.memcpy  %memref_1, %memref_host : memref<32x64xf32>, memref<32x64xf32>
    %subview = memref.subview %memref[0, 0] [32, 32] [1, 1] : memref<32x64xf16> to memref<32x32xf16, strided<[64, 1]>>
    %subview_2 = memref.subview %memref_0[0, 0] [32, 32] [1, 1] : memref<32x64xf16> to memref<32x32xf16, strided<[64, 1]>>
    %subview_3 = memref.subview %memref_1[0, 0] [32, 32] [1, 1] : memref<32x64xf32> to memref<32x32xf32, strided<[64, 1]>>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c4, %c2, %c1) threads in (%c1, %c1, %c1)  args(%subview : memref<32x32xf16, strided<[64, 1]>>, %subview_2 : memref<32x32xf16, strided<[64, 1]>>, %subview_3 : memref<32x32xf32, strided<[64, 1]>>)
    memref.dealloc  %memref_host : memref<32x64xf32>
    gpu.dealloc  %memref : memref<32x64xf16>
    gpu.dealloc  %memref_0 : memref<32x64xf16>
    %alloc = memref.alloc() : memref<32x64xf32>
    gpu.memcpy  %alloc, %memref_1 : memref<32x64xf32>, memref<32x64xf32>
    gpu.dealloc  %memref_1 : memref<32x64xf32>
    return %alloc : memref<32x64xf32>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<32x32xf16, strided<[64, 1]>>, %arg1: memref<32x32xf16, strided<[64, 1]>>, %arg2: memref<32x32xf32, strided<[64, 1]>>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 2, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %cst = arith.constant dense<1.000000e+00> : vector<8x16xf16>
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c8 : index
      %1 = arith.muli %block_id_y, %c16 : index
      %2 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<32x32xf32, strided<[64, 1]>> -> !xegpu.tensor_desc<8x16xf32>
      %3 = xegpu.load_nd %2  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %4 = scf.for %arg3 = %c0 to %c32 step %c16 iter_args(%arg4 = %3) -> (vector<8x16xf32>) {
        %5 = xegpu.create_nd_tdesc %arg0[%0, %arg3] : memref<32x32xf16, strided<[64, 1]>> -> !xegpu.tensor_desc<8x16xf16>
        %6 = xegpu.load_nd %5  : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
        %7 = xegpu.create_nd_tdesc %arg1[%arg3, %1] : memref<32x32xf16, strided<[64, 1]>> -> !xegpu.tensor_desc<16x16xf16>
        %8 = xegpu.load_nd %7 <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %9 = arith.addf %6, %cst : vector<8x16xf16>
        %10 = xegpu.dpas %9, %8, %arg4 : vector<8x16xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        scf.yield %10 : vector<8x16xf32>
      }
      xegpu.store_nd %4, %2  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // Allocate/get regular row major memrefs
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT:[128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   128,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    %0 = memref.get_global @__Aconstant_32x64xf16 : memref<32x64xf16>
    %1 = memref.get_global @__Bconstant_32x64xf16 : memref<32x64xf16>
    %2 = call @test(%0, %1) : (memref<32x64xf16>, memref<32x64xf16>) -> memref<32x64xf32>
    %cast = memref.cast %2 : memref<32x64xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}

