// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<8x32xf16>) -> memref<16x32xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<8x32xf16>
    memref.copy %arg0, %memref : memref<8x32xf16> to memref<8x32xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<16x32xf16>
    gpu.launch_func  @test_kernel::@test_copy blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8x32xf16>, %memref_1 : memref<16x32xf16>)

    gpu.dealloc  %memref : memref<8x32xf16>
    return %memref_1 : memref<16x32xf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_copy(%arg0: memref<8x32xf16>, %arg1: memref<16x32xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16>
      %2 = xegpu.create_nd_tdesc %arg1[2, 2] : memref<16x32xf16> -> !xegpu.tensor_desc<8x16xf16>
      %3 = xegpu.load_nd %0 {transpose = array<i64: 1, 0>, transpose_bit_width = 32:i32} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
      %4 = vector.shape_cast %3 : vector<8x8x2xf16> to vector<8x16xf16>
      xegpu.store_nd %4, %2 : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.alloc() : memref<8x32xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index

    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %m = arith.muli %i, %c16 : index
        %a = arith.addi %m, %j : index
        %t = index.castu %a : index to i16
        %val = arith.uitofp %t : i16 to f16
        memref.store %val, %0[%i, %j] : memref<8x32xf16>
      }
    }


    %2 = call @test(%0) : (memref<8x32xf16>) -> memref<16x32xf16>

    %cast = memref.cast %2: memref<16x32xf16> to memref<*xf16>

    //CHECK-COUNT-2: [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
    //CHECK: [0,   0,   0,   1,   16,   17,   32,   33,   48,   49,   64,   65,   80,   81,   96,   97,   112,   113,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    //CHECK: [0,   0,   2,   3,   18,   19,   34,   35,   50,   51,   66,   67,   82,   83,   98,   99,   114,   115,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    //CHECK: [0,   0,   4,   5,   20,   21,   36,   37,   52,   53,   68,   69,   84,   85,   100,   101,   116,   117,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    //CHECK: [0,   0,   6,   7,   22,   23,   38,   39,   54,   55,   70,   71,   86,   87,   102,   103,   118,   119,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    //CHECK: [0,   0,   8,   9,   24,   25,   40,   41,   56,   57,   72,   73,   88,   89,   104,   105,   120,   121,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    //CHECK: [0,   0,   10,   11,   26,   27,   42,   43,   58,   59,   74,   75,   90,   91,   106,   107,   122,   123,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    //CHECK: [0,   0,   12,   13,   28,   29,   44,   45,   60,   61,   76,   77,   92,   93,   108,   109,   124,   125,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    //CHECK: [0,   0,   14,   15,   30,   31,   46,   47,   62,   63,   78,   79,   94,   95,   110,   111,   126,   127,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    //CHECK-COUNT-6: [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
    call @printMemrefF16(%cast): (memref<*xf16>) -> ()
    return
  }

  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
}
