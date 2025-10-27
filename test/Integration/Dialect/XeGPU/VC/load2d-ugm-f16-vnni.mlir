// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<8x32xf16>) -> memref<16x32xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  () : memref<8x32xf16>
    gpu.memcpy  %memref, %arg0 : memref<8x32xf16>, memref<8x32xf16>
    %memref_0 = gpu.alloc  () : memref<16x32xf16>
    gpu.launch_func  @test_kernel::@test_copy blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<8x32xf16>, %memref_0 : memref<16x32xf16>)
    gpu.dealloc  %memref : memref<8x32xf16>
    %alloc = memref.alloc() : memref<16x32xf16>
    gpu.memcpy  %alloc, %memref_0 : memref<16x32xf16>, memref<16x32xf16>
    gpu.dealloc  %memref_0 : memref<16x32xf16>
    return %alloc : memref<16x32xf16>
  }
  gpu.module @test_kernel  {
    gpu.func @test_copy(%arg0: memref<8x32xf16>, %arg1: memref<16x32xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16>
      %1 = xegpu.load_nd %0 <{packed}> : !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
      %2 = vector.shape_cast %1 : vector<4x16x2xf16> to vector<4x32xf16>
      %3 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<16x32xf16> -> !xegpu.tensor_desc<4x32xf16>
      xegpu.store_nd %2, %3  : vector<4x32xf16>, !xegpu.tensor_desc<4x32xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<8x32xf16>
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %1 = arith.muli %arg0, %c16 : index
        %2 = arith.addi %1, %arg1 : index
        %3 = index.castu %2 : index to i16
        %4 = arith.uitofp %3 : i16 to f16
        memref.store %4, %alloc[%arg0, %arg1] : memref<8x32xf16>
      }
    }
    //CHECK: [0,   16,   1,   17,   2,   18,   3,   19,   4,   20,   5,   21,   6,   22,   7,   23,   8,   24,   9,   25,   10,   26,   11,   27,   12,   28,   13,   29,   14,   30,   15,   31]
    //CHECK: [32,   48,   33,   49,   34,   50,   35,   51,   36,   52,   37,   53,   38,   54,   39,   55,   40,   56,   41,   57,   42,   58,   43,   59,   44,   60,   45,   61,   46,   62,   47,   63]
    //CHECK: [64,   80,   65,   81,   66,   82,   67,   83,   68,   84,   69,   85,   70,   86,   71,   87,   72,   88,   73,   89,   74,   90,   75,   91,   76,   92,   77,   93,   78,   94,   79,   95]
    //CHECK: [96,   112,   97,   113,   98,   114,   99,   115,   100,   116,   101,   117,   102,   118,   103,   119,   104,   120,   105,   121,   106,   122,   107,   123,   108,   124,   109,   125,   110,   126,   111,   127]
    //CHECK-COUNT-12: [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
    %0 = call @test(%alloc) : (memref<8x32xf16>) -> memref<16x32xf16>
    %cast = memref.cast %0 : memref<16x32xf16> to memref<*xf16>
    call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
}

