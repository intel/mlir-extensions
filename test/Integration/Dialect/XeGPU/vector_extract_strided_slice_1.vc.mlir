// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<8x16xf16>, %B: memref<16x16xf16> ) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<8x16xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<16x16xf16>
    memref.copy %A, %memref : memref<8x16xf16> to memref<8x16xf16>
    memref.copy %B, %memref_1 : memref<16x16xf16> to memref<16x16xf16>
    %memref_2 = gpu.alloc  host_shared () : memref<8x16xf32>
    gpu.launch_func  @module0::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8x16xf16>, %memref_1 : memref<16x16xf16>, %memref_2 : memref<8x16xf32>)
    gpu.dealloc  %memref : memref<8x16xf16>
    gpu.dealloc  %memref_1 : memref<16x16xf16>
    return %memref_2 : memref<8x16xf32>
  }

    gpu.module @module0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<8x16xf16>, %B: memref<16x16xf16>, %Out: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      // load A tile
      %a_tile0 = xegpu.create_nd_tdesc %A [%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %val0 = xegpu.load_nd %a_tile0 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      // load B tile
      %b_tile0 = xegpu.create_nd_tdesc %B [%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
      %val2 = xegpu.load_nd %b_tile0 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
      // do DPAS
      %val4 = xegpu.dpas %val0, %val2 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
      // extract second 8x8
      %val5 = vector.extract_strided_slice %val4 {sizes = [8, 8], strides = [1, 1], offsets = [0, 8]} : vector<8x16xf32> to vector<8x8xf32>
      %cst_8x8_flat = arith.constant dense<1.0> : vector<64xf32>
      %cst_8x8 = vector.shape_cast %cst_8x8_flat : vector<64xf32> to vector<8x8xf32>
      // shift the first half to left and use %cst_8x8 as the second half
      %val6 = vector.shuffle %val5, %cst_8x8 [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<8x8xf32>, vector<8x8xf32>
      %val7 = vector.shape_cast %val6 : vector<16x8xf32> to vector<8x16xf32>
      // store
      %out_tile = xegpu.create_nd_tdesc %Out [%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %val7, %out_tile  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // init constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1_f32 = arith.constant 1.0 : f32
    // random init
    %lower = arith.constant -1.0 : f32
    %upper = arith.constant 1.0 : f32
    %false = arith.constant 0 : i1
    %A = memref.alloc() : memref<8x16xf16>
    %B = memref.alloc() : memref<16x16xf16>
    %Out_cpu = memref.alloc() : memref<8x16xf32>
    %A_random = memref.cast %A : memref<8x16xf16> to memref<*xf16>
    %B_random = memref.cast %B : memref<16x16xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%A_random, %lower, %upper, %false) : (memref<*xf16>, f32, f32, i1) -> ()
    call @fillResource1DRandomF16(%B_random, %lower, %upper, %false) : (memref<*xf16>, f32, f32, i1) -> ()
    // run GPU version
    %Out_gpu = call @test(%A, %B) : (memref<8x16xf16>, memref<16x16xf16>) -> memref<8x16xf32>
    %Out_gpu_cast = memref.cast %Out_gpu : memref<8x16xf32> to memref<*xf32>
    // run CPU version
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c8 to  %c16 step %c1 {
        %v0_init = arith.constant 0.0 : f32
        %result:1 = scf.for %k = %c0 to %c16 step %c1 iter_args(%v0 = %v0_init) -> f32 {
          %a0 = memref.load %A[%i, %k] : memref<8x16xf16>
          %b0 = memref.load %B[%k, %j] : memref<16x16xf16>
          %a0_f32 = arith.extf %a0 : f16 to f32
          %b0_f32 = arith.extf %b0 : f16 to f32
          %t0 = arith.mulf %a0_f32, %b0_f32 : f32
          %v0_new = arith.addf %v0, %t0 : f32
          scf.yield %v0_new : f32
        }
        // only update the first 8x8 of the result, next 8x8 is value 1
        %shifted_j = arith.subi %j, %c8 : index
        memref.store %result#0, %Out_cpu[%i, %shifted_j] : memref<8x16xf32>
        memref.store %c1_f32, %Out_cpu[%i, %j] : memref<8x16xf32>
      }
    }
    %Out_cpu_cast = memref.cast %Out_cpu : memref<8x16xf32> to memref<*xf32>
    // print GPU and CPU outs
    // call @printMemrefF32(%Out_cpu_cast) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%Out_gpu_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%Out_gpu_cast, %Out_cpu_cast) : (memref<*xf32>, memref<*xf32>) -> ()
    // dealloc
    memref.dealloc %A : memref<8x16xf16>
    memref.dealloc %B : memref<16x16xf16>
    memref.dealloc %Out_cpu : memref<8x16xf32>
    // gpu dealloc
    gpu.dealloc %Out_gpu : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
