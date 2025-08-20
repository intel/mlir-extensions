// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

#slm = #xegpu.scatter_tdesc_attr<memory_space=slm, chunk_size = 8>
#blk_slm = #xegpu.block_tdesc_attr<memory_space=slm>
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<8x32xf32>) -> memref<16x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %memref = gpu.alloc  host_shared () : memref<8x32xf32>
    memref.copy %arg0, %memref : memref<8x32xf32> to memref<8x32xf32>
    %memref_1 = gpu.alloc  host_shared () : memref<16x16xf32>
    gpu.launch_func  @test_kernel::@test_transpose blocks in (%c1, %c1, %c1) threads in (%c2, %c1, %c1) args(%memref : memref<8x32xf32>, %memref_1 : memref<16x16xf32>)

    gpu.dealloc  %memref : memref<8x32xf32>
    return %memref_1 : memref<16x16xf32>
  }

  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
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

      %id = gpu.subgroup_id : index
      %y = arith.muli %id, %c16 : index
      %in = xegpu.create_nd_tdesc %arg0[0, %y] : memref<8x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %data = xegpu.load_nd %in : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>

      // the following code uses slm to do the transpose. It contains 3 steps:
      // step1: store the data into slm using store scatter
      %slm = memref.alloc() : memref<256xf32, 3>
      %mask = arith.constant dense<1> : vector<16xi1>

      %base = arith.muli %id, %c128 : index
      %baseVec = vector.broadcast %base : index to vector<16xindex>
      %staticOff = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]> : vector<16xindex>
      %offsets = arith.addi %baseVec, %staticOff : vector<16xindex>

      %slm_desc = xegpu.create_tdesc %slm, %offsets : memref<256xf32, 3>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #slm>
      %trans = vector.transpose %data, [1, 0] : vector<8x16xf32> to vector<16x8xf32>
      xegpu.store %trans, %slm_desc, %mask : vector<16x8xf32>, !xegpu.tensor_desc<16x8xf32, #slm>, vector<16xi1>

      // step2: load from slm using 1d block load
      %base1 = arith.addi %base, %c0 : index
      %base2 = arith.addi %base, %c64 : index
      %slm_1d_desc_0 = xegpu.create_nd_tdesc %slm[%base1] : memref<256xf32, 3> -> !xegpu.tensor_desc<64xf32, #blk_slm>
      %slm_1d_desc_1 = xegpu.create_nd_tdesc %slm[%base2] : memref<256xf32, 3> -> !xegpu.tensor_desc<64xf32, #blk_slm>
      %data_1d_0 = xegpu.load_nd %slm_1d_desc_0 : !xegpu.tensor_desc<64xf32, #blk_slm> -> vector<64xf32>
      %data_1d_1 = xegpu.load_nd %slm_1d_desc_1 : !xegpu.tensor_desc<64xf32, #blk_slm> -> vector<64xf32>

      // step3: simply do the shape cast to get the final result
      %transposed_0 = vector.shape_cast %data_1d_0 : vector<64xf32> to vector<8x8xf32>
      %transposed_1 = vector.shape_cast %data_1d_1 : vector<64xf32> to vector<8x8xf32>

      %y2 = arith.muli %id, %c8 : index
      %out_0 = xegpu.create_nd_tdesc %arg1[0, %y2]: memref<16x16xf32> -> !xegpu.tensor_desc<8x8xf32>
      %out_1 = xegpu.create_nd_tdesc %arg1[8, %y2]: memref<16x16xf32> -> !xegpu.tensor_desc<8x8xf32>
      xegpu.store_nd %transposed_0, %out_0 : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32>
      xegpu.store_nd %transposed_1, %out_1 : vector<8x8xf32>, !xegpu.tensor_desc<8x8xf32>

      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index

    %0 = memref.alloc() : memref<8x32xf32>

    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        %mul = arith.muli %i, %c32 : index
        %add = arith.addi %mul, %j : index
        %int = arith.index_cast %add : index to i32
        %fp = arith.uitofp %int : i32 to f32
        memref.store %fp, %0[%i, %j] : memref<8x32xf32>
      }
    }

    %2 = call @test(%0) : (memref<8x32xf32>) -> memref<16x16xf32>

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
    %cast = memref.cast %2: memref<16x16xf32> to memref<*xf32>
    call @printMemrefF32(%cast): (memref<*xf32>) -> ()

    return
  }

  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
