// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/../xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/../xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

#global = #xegpu.scatter_tdesc_attr<memory_space=global, chunk_size = 4>
#slm = #xegpu.scatter_tdesc_attr<memory_space=slm, chunk_size = 4>

module @gemm attributes {gpu.container_module} {
  func.func @test() -> memref<16x4xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %out = gpu.alloc host_shared () : memref<16x4xf32>
    gpu.launch_func  @test_kernel::@test_store_scatter blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%out : memref<16x4xf32>)
    return %out : memref<16x4xf32>
  }

  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_store_scatter(%mem: memref<16x4xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<[[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15.],
                                   [16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31.],
                                   [32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47.],
                                   [48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63.]]> : vector<4x16xf32>

      %mask = arith.constant dense<1> : vector<16xi1>
      %offsets = arith.constant dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xindex>

      // store the cst into slm and load it back;
      %slm = memref.alloc() : memref<64xf32, 3>
      %slm_tdesc = xegpu.create_tdesc %slm, %offsets : memref<64xf32, 3>, vector<16xindex> -> !xegpu.tensor_desc<16x4xf32, #slm>
      xegpu.store %cst, %slm_tdesc, %mask {transpose} : vector<4x16xf32>, !xegpu.tensor_desc<16x4xf32, #slm>, vector<16xi1>
      // load from slm
      %data = xegpu.load %slm_tdesc, %mask {transpose} : !xegpu.tensor_desc<16x4xf32, #slm>, vector<16xi1> -> vector<4x16xf32>

      // store data to global memory
      %cast = memref.reinterpret_cast %mem to offset: [0], sizes: [64], strides: [1] : memref<16x4xf32> to memref<64xf32>
      %5 = xegpu.create_tdesc %cast, %offsets : memref<64xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x4xf32, #global>
      xegpu.store %data, %5, %mask {transpose} : vector<4x16xf32>, !xegpu.tensor_desc<16x4xf32, #global>, vector<16xi1>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %B = call @test() : () -> memref<16x4xf32>
    %cast = memref.cast %B : memref<16x4xf32> to memref<*xf32>

    //CHECK: [0,   16,   32,   48],
    //CHECK: [1,   17,   33,   49],
    //CHECK: [2,   18,   34,   50],
    //CHECK: [3,   19,   35,   51],
    //CHECK: [4,   20,   36,   52],
    //CHECK: [5,   21,   37,   53],
    //CHECK: [6,   22,   38,   54],
    //CHECK: [7,   23,   39,   55],
    //CHECK: [8,   24,   40,   56],
    //CHECK: [9,   25,   41,   57],
    //CHECK: [10,   26,   42,   58],
    //CHECK: [11,   27,   43,   59],
    //CHECK: [12,   28,   44,   60],
    //CHECK: [13,   29,   45,   61],
    //CHECK: [14,   30,   46,   62],
    //CHECK: [15,   31,   47,   63]
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }

  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
