// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<16x16xf32>) -> memref<16x32xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<16x16xf32>
    memref.copy %arg0, %memref : memref<16x16xf32> to memref<16x16xf32>
    %memref_1 = gpu.alloc  host_shared () : memref<16x32xf32>
    gpu.launch_func  @test_kernel::@test_copy blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<16x16xf32>, %memref_1 : memref<16x32xf32>)

    gpu.dealloc  %memref : memref<16x16xf32>
    return %memref_1 : memref<16x32xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_copy(%arg0: memref<16x16xf32>, %arg1: memref<16x32xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<16x16xf32> -> !xegpu.tensor_desc<16x8xf32>
      %1 = xegpu.load_nd %0 {transpose = array<i64: 1, 0>} : !xegpu.tensor_desc<16x8xf32> -> vector<8x16xf32>
      %2 = xegpu.create_nd_tdesc %arg1[2, 2] : memref<16x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %1, %2 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>

      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.alloc() : memref<16x16xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index

    // input matrix is [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], ...]
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c8 step %c1 {
        %m = arith.muli %i, %c8 : index
        %a = arith.addi %m, %j : index
        %t = index.castu %a : index to i32
        %val = arith.uitofp %t : i32 to f32
        memref.store %val, %0[%i, %j] : memref<16x16xf32>
      }
    }


    %2 = call @test(%0) : (memref<16x16xf32>) -> memref<16x32xf32>

    %cast = memref.cast %2: memref<16x32xf32> to memref<*xf32>

    //CHECK-COUNT-2: [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
    //CHECK: [0,   0,   0,   8,   16,   24,   32,   40,   48,   56,   64,   72,   80,   88,   96,   104,   112,   120,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    //CHECK: [0,   0,   1,   9,   17,   25,   33,   41,   49,   57,   65,   73,   81,   89,   97,   105,   113,   121,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    //CHECK: [0,   0,   2,   10,   18,   26,   34,   42,   50,   58,   66,   74,   82,   90,   98,   106,   114,   122,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    //CHECK: [0,   0,   3,   11,   19,   27,   35,   43,   51,   59,   67,   75,   83,   91,   99,   107,   115,   123,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    //CHECK: [0,   0,   4,   12,   20,   28,   36,   44,   52,   60,   68,   76,   84,   92,   100,   108,   116,   124,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    //CHECK: [0,   0,   5,   13,   21,   29,   37,   45,   53,   61,   69,   77,   85,   93,   101,   109,   117,   125,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    //CHECK: [0,   0,   6,   14,   22,   30,   38,   46,   54,   62,   70,   78,   86,   94,   102,   110,   118,   126,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    //CHECK: [0,   0,   7,   15,   23,   31,   39,   47,   55,   63,   71,   79,   87,   95,   103,   111,   119,   127,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    //CHECK-COUNT-6: [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
    call @printMemrefF32(%cast): (memref<*xf32>) -> ()
    return
  }

  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
