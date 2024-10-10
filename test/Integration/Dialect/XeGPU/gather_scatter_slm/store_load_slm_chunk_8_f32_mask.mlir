// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/../xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/../xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck


#global = #xegpu.scatter_tdesc_attr<memory_space=global, chunk_size = 8>
#slm = #xegpu.scatter_tdesc_attr<memory_space=slm, chunk_size = 8>

module @gemm attributes {gpu.container_module} {
  func.func @test() -> memref<16x8xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %out = gpu.alloc host_shared () : memref<16x8xf32>
    %slm = memref.alloc() : memref<128xf32, 3>
    gpu.launch_func  @test_kernel::@test_store_scatter blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%out : memref<16x8xf32>, %slm : memref<128xf32, 3>)
    return %out : memref<16x8xf32>
  }

  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_store_scatter(%mem: memref<16x8xf32>, %slm: memref<128xf32, 3>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<[[  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,  12.,  13.,  14.,  15.],
                                   [ 16.,  17.,  18.,  19.,  20.,  21.,  22.,  23.,  24.,  25.,  26.,  27.,  28.,  29.,  30.,  31.],
                                   [ 32.,  33.,  34.,  35.,  36.,  37.,  38.,  39.,  40.,  41.,  42.,  43.,  44.,  45.,  46.,  47.],
                                   [ 48.,  49.,  50.,  51.,  52.,  53.,  54.,  55.,  56.,  57.,  58.,  59.,  60.,  61.,  62.,  63.],
                                   [ 64.,  65.,  66.,  67.,  68.,  69.,  70.,  71.,  72.,  73.,  74.,  75.,  76.,  77.,  78.,  79.],
                                   [ 80.,  81.,  82.,  83.,  84.,  85.,  86.,  87.,  88.,  89.,  90.,  91.,  92.,  93.,  94.,  95.],
                                   [ 96.,  97.,  98.,  99., 100., 101., 102., 103., 104., 105., 106., 107., 108., 109., 110., 111.],
                                   [112., 113., 114., 115., 116., 117., 118., 119., 120., 121., 122., 123., 124., 125., 126., 127.]]> : vector<8x16xf32>

      %mask = arith.constant dense<[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]> : vector<16xi1>
      %offsets = arith.constant dense<[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]> : vector<16xindex>

      // store the cst into slm
      %slm_tdesc = xegpu.create_tdesc %slm, %offsets : memref<128xf32, 3>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #slm>
      xegpu.store %cst, %slm_tdesc, %mask {transpose} : vector<8x16xf32>, !xegpu.tensor_desc<16x8xf32, #slm>, vector<16xi1>

      // load from slm
      %data = xegpu.load %slm_tdesc, %mask {transpose} : !xegpu.tensor_desc<16x8xf32, #slm>, vector<16xi1> -> vector<8x16xf32>

      // store data to global memory
      %cast = memref.reinterpret_cast %mem to offset: [0], sizes: [128], strides: [1] : memref<16x8xf32> to memref<128xf32>
      %5 = xegpu.create_tdesc %cast, %offsets : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #global>
      xegpu.store %data, %5, %mask {transpose} : vector<8x16xf32>, !xegpu.tensor_desc<16x8xf32, #global>, vector<16xi1>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %B = call @test() : () -> memref<16x8xf32>
    %cast = memref.cast %B : memref<16x8xf32> to memref<*xf32>


    //CHECK: 0,   16,   32,   48,   64,   80,   96,   112
    //CHECK: 0,   0,   0,   0,   0,   0,   0,   0
    //CHECK: 2,   18,   34,   50,   66,   82,   98,   114
    //CHECK: 0,   0,   0,   0,   0,   0,   0,   0
    //CHECK: 4,   20,   36,   52,   68,   84,   100,   116
    //CHECK: 0,   0,   0,   0,   0,   0,   0,   0
    //CHECK: 6,   22,   38,   54,   70,   86,   102,   118
    //CHECK: 0,   0,   0,   0,   0,   0,   0,   0
    //CHECK: 8,   24,   40,   56,   72,   88,   104,   120
    //CHECK: 0,   0,   0,   0,   0,   0,   0,   0
    //CHECK: 10,   26,   42,   58,   74,   90,   106,   122
    //CHECK: 0,   0,   0,   0,   0,   0,   0,   0
    //CHECK: 12,   28,   44,   60,   76,   92,   108,   124
    //CHECK: 0,   0,   0,   0,   0,   0,   0,   0
    //CHECK: 14,   30,   46,   62,   78,   94,   110,   126
    //CHECK: 0,   0,   0,   0,   0,   0,   0,   0
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }

  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
