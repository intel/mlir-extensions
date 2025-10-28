// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<32x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<32x16xf32>
    gpu.memcpy  %memref, %arg0 : memref<32x16xf32>, memref<32x16xf32>
    %memref_0 = gpu.alloc  () : memref<8x16xf32>
    gpu.launch_func  @module0::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<32x16xf32>, %memref_0 : memref<8x16xf32>)
    gpu.dealloc  %memref : memref<32x16xf32>
    %alloc = memref.alloc() : memref<8x16xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<8x16xf32>, memref<8x16xf32>
    gpu.dealloc  %memref_0 : memref<8x16xf32>
    return %alloc : memref<8x16xf32>
  }
  gpu.module @module0  {
    gpu.func @test_kernel(%arg0: memref<32x16xf32>, %arg1: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      // load tile
      // extract the bottom 8x8 part of first 32x8 block
      // extract the bottom 8x8 part of second 32x8 block
      // combine these two 8x8 tiles into a single 8x16 tile
      // store the result
      %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<32x16xf32> -> !xegpu.tensor_desc<32x8xf32, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
      %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<32x8xf32, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x32x8xf32>
      %2 = vector.extract_strided_slice %1 {offsets = [0, 24], sizes = [1, 8], strides = [1, 1]} : vector<2x32x8xf32> to vector<1x8x8xf32>
      %3 = vector.extract_strided_slice %1 {offsets = [1, 24], sizes = [1, 8], strides = [1, 1]} : vector<2x32x8xf32> to vector<1x8x8xf32>
      %4 = vector.shape_cast %2 : vector<1x8x8xf32> to vector<8x8xf32>
      %5 = vector.shape_cast %3 : vector<1x8x8xf32> to vector<8x8xf32>
      %6 = vector.shuffle %4, %5 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x8xf32>, vector<8x8xf32>
      %7 = vector.shape_cast %6 : vector<16x8xf32> to vector<8x16xf32>
      %8 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %7, %8  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // init constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c24 = arith.constant 24 : index
    // fill A with values form 0, 1, ...., 511
        // only store the bottom 8x16 into Out_cpu
    %alloc = memref.alloc() : memref<32x16xf32>
    %alloc_0 = memref.alloc() : memref<8x16xf32>
    scf.for %arg0 = %c0 to %c32 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %1 = arith.muli %arg0, %c16 : index
        %2 = arith.addi %1, %arg1 : index
        %3 = arith.index_cast %2 : index to i32
        %4 = arith.sitofp %3 : i32 to f32
        %5 = arith.cmpi sge, %arg0, %c24 : index
        scf.if %5 {
          %6 = arith.subi %arg0, %c24 : index
          memref.store %4, %alloc_0[%6, %arg1] : memref<8x16xf32>
        }
        memref.store %4, %alloc[%arg0, %arg1] : memref<32x16xf32>
      }
    }
    // run GPU version
    // print GPU and CPU outs
    // call @printMemrefF32(%Out_gpu_cast) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%Out_cpu_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    // dealloc
    // gpu dealloc
    %0 = call @test(%alloc) : (memref<32x16xf32>) -> memref<8x16xf32>
    %cast = memref.cast %0 : memref<8x16xf32> to memref<*xf32>
    %cast_1 = memref.cast %alloc_0 : memref<8x16xf32> to memref<*xf32>
    call @printAllcloseF32(%cast, %cast_1) : (memref<*xf32>, memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<32x16xf32>
    memref.dealloc  %0 : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
