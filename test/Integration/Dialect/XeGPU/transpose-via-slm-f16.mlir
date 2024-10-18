// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

#slm = #xegpu.scatter_tdesc_attr<memory_space=slm, chunk_size = 8>
#blk_slm = #xegpu.block_tdesc_attr<memory_space=slm>
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<16x32xf16>) -> memref<8x64xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %memref = gpu.alloc  host_shared () : memref<16x32xf16>
    memref.copy %arg0, %memref : memref<16x32xf16> to memref<16x32xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<8x64xf16>
    gpu.launch_func  @test_kernel::@test_transpose blocks in (%c1, %c1, %c1) threads in (%c4, %c1, %c1) args(%memref : memref<16x32xf16>, %memref_1 : memref<8x64xf16>)

    gpu.dealloc  %memref : memref<16x32xf16>
    return %memref_1 : memref<8x64xf16>
  }

  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    // this example is to illustrate an example of using slm to do the transpose.
    // the high level logic is equivalent to the following code:
    // %data = xegpu.load_nd %in : !xegpu.tensor_desc<16x32xf16> -> vector<16x8xf16>
    // %transpose = vector.transpose %data : vector<16x8xf16> to vector<8x16xf16>
    // %out = xegpu.store_nd %transpose, %out : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
    // But we use load_nd with packed, store to and load from slm to replace the transpose op here.
    gpu.func @test_transpose(%arg0: memref<16x32xf16>, %arg1: memref<8x64xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index

      %id = gpu.subgroup_id : index
      %y_in = arith.muli %id, %c8 : index

      %in = xegpu.create_nd_tdesc %arg0[0, %y_in] : memref<16x32xf16> -> !xegpu.tensor_desc<16x8xf16>
      // original load is %data = xegpu.load_nd %in : !xegpu.tensor_desc<16x8xf16> -> vector<16x8xf16>
      // now it is transformed to use packed attribute
      %data = xegpu.load_nd %in {packed} : !xegpu.tensor_desc<16x8xf16> -> vector<8x8x2xf16>
      %shapecast = vector.shape_cast %data : vector<8x8x2xf16> to vector<128xf16>
      %data32b = vector.bitcast %shapecast : vector<128xf16> to vector<64xf32>
      %tmp = vector.shape_cast %data32b : vector<64xf32> to vector<8x8xf32>
      %pad = arith.constant dense<0.0> : vector<8x8xf32>
      %comb = vector.shuffle %tmp, %pad[0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x8xf32>, vector<8x8xf32>
      %cast = vector.shape_cast %comb : vector<16x8xf32> to vector<8x16xf32>

      // the following code uses slm to do the transpose. It contains 3 steps:
      // step1: store the data into slm using store scatter
      %slm = memref.alloc() : memref<256xf32, 3>
      %mask = arith.constant dense<[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]> : vector<16xi1>

      %base = arith.muli %id, %c64 : index
      %baseVec = vector.broadcast %base : index to vector<16xindex>
      %staticOff = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]> : vector<16xindex>
      %offsets = arith.addi %baseVec, %staticOff : vector<16xindex>

      %slm_desc = xegpu.create_tdesc %slm, %offsets : memref<256xf32, 3>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #slm>
      xegpu.store %cast, %slm_desc, %mask {transpose} : vector<8x16xf32>, !xegpu.tensor_desc<16x8xf32, #slm>, vector<16xi1>

      // step2: load from slm using 1d block load
      %off = arith.muli %id, %c64 : index
      %slm_1d_desc = xegpu.create_nd_tdesc %slm[%off] : memref<256xf32, 3> -> !xegpu.tensor_desc<64xf32, #blk_slm>
      %data_1d = xegpu.load_nd %slm_1d_desc : !xegpu.tensor_desc<64xf32, #blk_slm> -> vector<64xf32>

      // step3: simply do the shape cast to get the final result
      %bitcast = vector.bitcast %data_1d : vector<64xf32> to vector<128xf16>
      %transposed = vector.shape_cast %bitcast : vector<128xf16> to vector<8x16xf16>

      %out_y = arith.muli %id, %c16 : index
      %out = xegpu.create_nd_tdesc %arg1[0, %out_y]: memref<8x64xf16> -> !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %transposed, %out : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index

    %0 = memref.alloc() : memref<16x32xf16>

    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c32 step %c1 {
        %mul = arith.muli %i, %c32 : index
        %add = arith.addi %mul, %j : index
        %int = arith.index_cast %add : index to i16
        %fp = arith.uitofp %int : i16 to f16
        memref.store %fp, %0[%i, %j] : memref<16x32xf16>
      }
    }
    %2 = call @test(%0) : (memref<16x32xf16>) -> memref<8x64xf16>
    %cast = memref.cast %2: memref<8x64xf16> to memref<*xf16>

    //CHECK: [0,   32,   64,   96,   128,   160,   192,   224,   256,   288,   320,   352,   384,   416,   448,   480,   8,   40,   72,   104,   136,   168,   200,   232,   264,   296,   328,   360,   392,   424,   456,   488,   16,   48,   80,   112,   144,   176,   208,   240,   272,   304,   336,   368,   400,   432,   464,   496,   24,   56,   88,   120,   152,   184,   216,   248,   280,   312,   344,   376,   408,   440,   472,   504]
    //CHECK: [1,   33,   65,   97,   129,   161,   193,   225,   257,   289,   321,   353,   385,   417,   449,   481,   9,   41,   73,   105,   137,   169,   201,   233,   265,   297,   329,   361,   393,   425,   457,   489,   17,   49,   81,   113,   145,   177,   209,   241,   273,   305,   337,   369,   401,   433,   465,   497,   25,   57,   89,   121,   153,   185,   217,   249,   281,   313,   345,   377,   409,   441,   473,   505]
    //CHECK: [2,   34,   66,   98,   130,   162,   194,   226,   258,   290,   322,   354,   386,   418,   450,   482,   10,   42,   74,   106,   138,   170,   202,   234,   266,   298,   330,   362,   394,   426,   458,   490,   18,   50,   82,   114,   146,   178,   210,   242,   274,   306,   338,   370,   402,   434,   466,   498,   26,   58,   90,   122,   154,   186,   218,   250,   282,   314,   346,   378,   410,   442,   474,   506]
    //CHECK: [3,   35,   67,   99,   131,   163,   195,   227,   259,   291,   323,   355,   387,   419,   451,   483,   11,   43,   75,   107,   139,   171,   203,   235,   267,   299,   331,   363,   395,   427,   459,   491,   19,   51,   83,   115,   147,   179,   211,   243,   275,   307,   339,   371,   403,   435,   467,   499,   27,   59,   91,   123,   155,   187,   219,   251,   283,   315,   347,   379,   411,   443,   475,   507]
    //CHECK: [4,   36,   68,   100,   132,   164,   196,   228,   260,   292,   324,   356,   388,   420,   452,   484,   12,   44,   76,   108,   140,   172,   204,   236,   268,   300,   332,   364,   396,   428,   460,   492,   20,   52,   84,   116,   148,   180,   212,   244,   276,   308,   340,   372,   404,   436,   468,   500,   28,   60,   92,   124,   156,   188,   220,   252,   284,   316,   348,   380,   412,   444,   476,   508]
    //CHECK: [5,   37,   69,   101,   133,   165,   197,   229,   261,   293,   325,   357,   389,   421,   453,   485,   13,   45,   77,   109,   141,   173,   205,   237,   269,   301,   333,   365,   397,   429,   461,   493,   21,   53,   85,   117,   149,   181,   213,   245,   277,   309,   341,   373,   405,   437,   469,   501,   29,   61,   93,   125,   157,   189,   221,   253,   285,   317,   349,   381,   413,   445,   477,   509]
    //CHECK: [6,   38,   70,   102,   134,   166,   198,   230,   262,   294,   326,   358,   390,   422,   454,   486,   14,   46,   78,   110,   142,   174,   206,   238,   270,   302,   334,   366,   398,   430,   462,   494,   22,   54,   86,   118,   150,   182,   214,   246,   278,   310,   342,   374,   406,   438,   470,   502,   30,   62,   94,   126,   158,   190,   222,   254,   286,   318,   350,   382,   414,   446,   478,   510]
    //CHECK: [7,   39,   71,   103,   135,   167,   199,   231,   263,   295,   327,   359,   391,   423,   455,   487,   15,   47,   79,   111,   143,   175,   207,   239,   271,   303,   335,   367,   399,   431,   463,   495,   23,   55,   87,   119,   151,   183,   215,   247,   279,   311,   343,   375,   407,   439,   471,   503,   31,   63,   95,   127,   159,   191,   223,   255,   287,   319,   351,   383,   415,   447,   479,   511]
    call @printMemrefF16(%cast): (memref<*xf16>) -> ()
    return
  }

  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
}
