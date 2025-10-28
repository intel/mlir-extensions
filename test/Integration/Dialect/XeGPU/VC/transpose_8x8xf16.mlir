// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @transpose attributes {gpu.container_module} {
  func.func @test(%arg0: memref<32x32xf16>) -> memref<32x32xf16> attributes {llvm.emit_c_interface} {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<32x32xf16>
    gpu.memcpy  %memref, %arg0 : memref<32x32xf16>, memref<32x32xf16>
    %memref_0 = gpu.alloc  () : memref<32x32xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c4, %c4, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<32x32xf16>, %memref_0 : memref<32x32xf16>)
    gpu.dealloc  %memref : memref<32x32xf16>
    %alloc = memref.alloc() : memref<32x32xf16>
    gpu.memcpy  %alloc, %memref_0 : memref<32x32xf16>, memref<32x32xf16>
    gpu.dealloc  %memref_0 : memref<32x32xf16>
    return %alloc : memref<32x32xf16>
  }
  gpu.module @test_kernel  {
    gpu.func @test_kernel(%arg0: memref<32x32xf16>, %arg1: memref<32x32xf16>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 4, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = arith.muli %block_id_x, %c8 : index
      %1 = arith.muli %block_id_y, %c8 : index
      %2 = xegpu.create_nd_tdesc %arg0[%0, %1] : memref<32x32xf16> -> !xegpu.tensor_desc<8x8xf16>
      %3 = xegpu.load_nd %2  : !xegpu.tensor_desc<8x8xf16> -> vector<8x8xf16>
      %4 = xegpu.create_nd_tdesc %arg1[%1, %0] : memref<32x32xf16> -> !xegpu.tensor_desc<8x8xf16>
      %5 = vector.transpose %3, [1, 0] : vector<8x8xf16> to vector<8x8xf16>
      xegpu.store_nd %5, %4  : vector<8x8xf16>, !xegpu.tensor_desc<8x8xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    // A matrix: row-major, start from 0.0, increase 0.01 per element
    // B matrix: A matrix + 1.0
    %alloc = memref.alloc() : memref<32x32xf16>
    %alloc_0 = memref.alloc() : memref<32x32xf32>
    scf.for %arg0 = %c0 to %c32 step %c1 {
      scf.for %arg1 = %c0 to %c32 step %c1 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.uitofp %1 : i32 to f16
        memref.store %2, %alloc[%arg0, %arg1] : memref<32x32xf16>
        %3 = arith.extf %2 : f16 to f32
        memref.store %3, %alloc_0[%arg1, %arg0] : memref<32x32xf32>
      }
    }
    // call @printMemreff16(%cast_ref) : (memref<*xf16>) -> ()
    // CHECK:   [ALLCLOSE: TRUE]
    %0 = call @test(%alloc) : (memref<32x32xf16>) -> memref<32x32xf16>
    %cast = memref.cast %0 : memref<32x32xf16> to memref<*xf16>
    %cast_1 = memref.cast %alloc_0 : memref<32x32xf32> to memref<*xf32>
    call @printAllcloseF16(%cast, %cast_1) : (memref<*xf16>, memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
