// RUN: %python_executable %imex_runner --requires=mlir-levelzero-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%mlir_levelzero_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<8x32xf32>) -> memref<16x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %memref = gpu.alloc  () : memref<8x32xf32>
    gpu.memcpy  %memref, %arg0 : memref<8x32xf32>, memref<8x32xf32>
    %memref_0 = gpu.alloc  () : memref<16x16xf32>
    gpu.launch_func  @test_kernel::@test_transpose blocks in (%c1, %c1, %c1) threads in (%c2, %c1, %c1)  args(%memref : memref<8x32xf32>, %memref_0 : memref<16x16xf32>)
    gpu.dealloc  %memref : memref<8x32xf32>
    %alloc = memref.alloc() : memref<16x16xf32>
    gpu.memcpy  %alloc, %memref_0 : memref<16x16xf32>, memref<16x16xf32>
    gpu.dealloc  %memref_0 : memref<16x16xf32>
    return %alloc : memref<16x16xf32>
  }
  gpu.module @test_kernel  {
    // this example is to illustrate an example of using slm to do the transpose.
    // the high level logic is equivalent to the following code:
    // %data = xegpu.load_nd %in : !xegpu.tensor_desc<8x32xf32> -> vector<8x32xf32>
    // %transpose = vector.transpose %data : vector<8x32xf32> to vector<16x16xf32>
    // %out = xegpu.store_nd %transpose, %out : vector<16x16xf32>, !xegpu.tensor_desc<16x16xf32>
    // But we use store to and load from slm to replace the transpose op here.
    gpu.func @test_transpose(%arg0: memref<8x32xf32>, %arg1: memref<16x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      // the following code uses slm to do the transpose. It contains 3 steps:
      // step1: store the data into slm using store scatter
      // step2: load from slm using 1d block load
      // step3: simply do the shape cast to get the final result
      %0 = gpu.subgroup_id : index
      %1 = arith.muli %0, %c16 : index
      %2 = xegpu.create_nd_tdesc %arg0[0, %1] : memref<8x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %3 = xegpu.load_nd %2  : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %alloc = memref.alloc() : memref<256xf32, 3>
      %cst = arith.constant dense<true> : vector<16xi1>
      %4 = arith.muli %0, %c128 : index
      %5 = vector.broadcast %4 : index to vector<16xindex>
      %cst_0 = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]> : vector<16xindex>
      %6 = arith.addi %5, %cst_0 : vector<16xindex>
      %7 = xegpu.create_tdesc %alloc, %6 : memref<256xf32, 3>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>
      %8 = vector.transpose %3, [1, 0] : vector<8x16xf32> to vector<16x8xf32>
      xegpu.store %8, %7, %cst  : vector<16x8xf32>, !xegpu.tensor_desc<16x8xf32, #xegpu.scatter_tdesc_attr<memory_space =  slm, chunk_size = 8 : i64>>, vector<16xi1>
      %9 = arith.addi %4, %c0 : index
      %10 = arith.addi %4, %c64 : index
      %11 = xegpu.create_nd_tdesc %alloc[%9] : memref<256xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm>>
      %12 = xegpu.create_nd_tdesc %alloc[%10] : memref<256xf32, 3> -> !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm>>
      %13 = xegpu.load_nd %11  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm>> -> vector<64xf32>
      %14 = xegpu.load_nd %12  : !xegpu.tensor_desc<64xf32, #xegpu.block_tdesc_attr<memory_space =  slm>> -> vector<64xf32>
      %15 = vector.shape_cast %13 : vector<64xf32> to vector<8x8xf32>
      %16 = vector.shape_cast %14 : vector<64xf32> to vector<8x8xf32>
      %17 = arith.muli %0, %c8 : index
      %18 = xegpu.create_nd_tdesc %arg1[0, %17] : memref<16x16xf32> -> !xegpu.tensor_desc<8x8xf32>
      %19 = xegpu.create_nd_tdesc %arg1[8, %17] : memref<16x16xf32> -> !xegpu.tensor_desc<8x8xf32>
      xegpu.store_nd %15, %18  : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32>
      xegpu.store_nd %16, %19  : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %alloc = memref.alloc() : memref<8x32xf32>
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c32 step %c1 {
        %1 = arith.muli %arg0, %c32 : index
        %2 = arith.addi %1, %arg1 : index
        %3 = arith.index_cast %2 : index to i32
        %4 = arith.uitofp %3 : i32 to f32
        memref.store %4, %alloc[%arg0, %arg1] : memref<8x32xf32>
      }
    }
    //CHECK: [0,   32,   64,   96,   128,   160,   192,   224,   16,   48,   80,   112,   144,   176,   208,   240]
    //CHECK: [1,   33,   65,   97,   129,   161,   193,   225,   17,   49,   81,   113,   145,   177,   209,   241]
    //CHECK: [2,   34,   66,   98,   130,   162,   194,   226,   18,   50,   82,   114,   146,   178,   210,   242]
    //CHECK: [3,   35,   67,   99,   131,   163,   195,   227,   19,   51,   83,   115,   147,   179,   211,   243]
    //CHECK: [4,   36,   68,   100,   132,   164,   196,   228,   20,   52,   84,   116,   148,   180,   212,   244]
    //CHECK: [5,   37,   69,   101,   133,   165,   197,   229,   21,   53,   85,   117,   149,   181,   213,   245]
    //CHECK: [6,   38,   70,   102,   134,   166,   198,   230,   22,   54,   86,   118,   150,   182,   214,   246]
    //CHECK: [7,   39,   71,   103,   135,   167,   199,   231,   23,   55,   87,   119,   151,   183,   215,   247]
    //CHECK: [8,   40,   72,   104,   136,   168,   200,   232,   24,   56,   88,   120,   152,   184,   216,   248]
    //CHECK: [9,   41,   73,   105,   137,   169,   201,   233,   25,   57,   89,   121,   153,   185,   217,   249]
    //CHECK: [10,   42,   74,   106,   138,   170,   202,   234,   26,   58,   90,   122,   154,   186,   218,   250]
    //CHECK: [11,   43,   75,   107,   139,   171,   203,   235,   27,   59,   91,   123,   155,   187,   219,   251]
    //CHECK: [12,   44,   76,   108,   140,   172,   204,   236,   28,   60,   92,   124,   156,   188,   220,   252]
    //CHECK: [13,   45,   77,   109,   141,   173,   205,   237,   29,   61,   93,   125,   157,   189,   221,   253]
    //CHECK: [14,   46,   78,   110,   142,   174,   206,   238,   30,   62,   94,   126,   158,   190,   222,   254]
    //CHECK: [15,   47,   79,   111,   143,   175,   207,   239,   31,   63,   95,   127,   159,   191,   223,   255]
    %0 = call @test(%alloc) : (memref<8x32xf32>) -> memref<16x16xf32>
    %cast = memref.cast %0 : memref<16x16xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
