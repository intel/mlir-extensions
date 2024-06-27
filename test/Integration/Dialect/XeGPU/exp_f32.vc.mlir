// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<8x16xf16>, %B: memref<16x16xf16> ) -> (memref<8x16xf32>, memref<8x16xf32>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<8x16xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<16x16xf16>
    memref.copy %A, %memref : memref<8x16xf16> to memref<8x16xf16>
    memref.copy %B, %memref_1 : memref<16x16xf16> to memref<16x16xf16>
    %memref_2 = gpu.alloc  host_shared () : memref<8x16xf32>
    %memref_3 = gpu.alloc  host_shared () : memref<8x16xf32>
    gpu.launch_func  @module0::@test_exp_larger_vec blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8x16xf16>, %memref_1 : memref<16x16xf16>, %memref_2 : memref<8x16xf32>)
    gpu.launch_func  @module1::@test_exp_generic_vec blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8x16xf16>, %memref_1 : memref<16x16xf16>, %memref_3 : memref<8x16xf32>)
    gpu.dealloc  %memref : memref<8x16xf16>
    gpu.dealloc  %memref_1 : memref<16x16xf16>
    return %memref_2, %memref_3 : memref<8x16xf32>, memref<8x16xf32>
  }

    gpu.module @module0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_exp_larger_vec(%A: memref<8x16xf16>, %B: memref<16x16xf16>, %Out: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      // load A tile
      %a_tile0 = xegpu.create_nd_tdesc %A [%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %val0 = xegpu.load_nd %a_tile0 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      // load B tile
      %b_tile0 = xegpu.create_nd_tdesc %B [%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
      %val2 = xegpu.load_nd %b_tile0 { packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
      // do DPAS
      %val4 = xegpu.dpas %val0, %val2 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
      // take exp
      %t6 = math.exp %val4 : vector<8x16xf32>
      // store
      %out_tile = xegpu.create_nd_tdesc %Out [%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %t6, %out_tile  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  gpu.module @module1 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_exp_generic_vec(%A: memref<8x16xf16>, %B: memref<16x16xf16>, %Out: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
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
      // extract dpas out into 16xf32 vectors
      %cst1 = arith.constant dense<1.4426950408889634> : vector<128xf32>
      %v0 = vector.extract %val4[0] : vector<16xf32> from vector<8x16xf32>
      %v1 = vector.extract %val4[1] : vector<16xf32> from vector<8x16xf32>
      %v2 = vector.extract %val4[2] : vector<16xf32> from vector<8x16xf32>
      %v3 = vector.extract %val4[3] : vector<16xf32> from vector<8x16xf32>
      %v4 = vector.extract %val4[4] : vector<16xf32> from vector<8x16xf32>
      %v5 = vector.extract %val4[5] : vector<16xf32> from vector<8x16xf32>
      %v6 = vector.extract %val4[6] : vector<16xf32> from vector<8x16xf32>
      %v7 = vector.extract %val4[7] : vector<16xf32> from vector<8x16xf32>
      // do generic size exp
      %v0_exp = math.exp %v0 : vector<16xf32>
      %v1_exp = math.exp %v1 : vector<16xf32>
      %v2_exp = math.exp %v2 : vector<16xf32>
      %v3_exp = math.exp %v3 : vector<16xf32>
      %v4_exp = math.exp %v4 : vector<16xf32>
      %v5_exp = math.exp %v5 : vector<16xf32>
      %v6_exp = math.exp %v6 : vector<16xf32>
      %v7_exp = math.exp %v7 : vector<16xf32>
      %v0_exp_cast = vector.shape_cast %v0_exp : vector<16xf32> to vector<1x16xf32>
      %v1_exp_cast = vector.shape_cast %v1_exp : vector<16xf32> to vector<1x16xf32>
      %v2_exp_cast = vector.shape_cast %v2_exp : vector<16xf32> to vector<1x16xf32>
      %v3_exp_cast = vector.shape_cast %v3_exp : vector<16xf32> to vector<1x16xf32>
      %v4_exp_cast = vector.shape_cast %v4_exp : vector<16xf32> to vector<1x16xf32>
      %v5_exp_cast = vector.shape_cast %v5_exp : vector<16xf32> to vector<1x16xf32>
      %v6_exp_cast = vector.shape_cast %v6_exp : vector<16xf32> to vector<1x16xf32>
      %v7_exp_cast = vector.shape_cast %v7_exp : vector<16xf32> to vector<1x16xf32>
      // construct 4x16xf32 vector from the smaller ones
      %t0 = vector.shuffle %v0_exp_cast, %v1_exp_cast [0, 1] : vector<1x16xf32>, vector<1x16xf32>
      %t1 = vector.shuffle %v2_exp_cast, %v3_exp_cast [0, 1] : vector<1x16xf32>, vector<1x16xf32>
      %t2 = vector.shuffle %v4_exp_cast, %v5_exp_cast [0, 1] : vector<1x16xf32>, vector<1x16xf32>
      %t3 = vector.shuffle %v6_exp_cast, %v7_exp_cast [0, 1] : vector<1x16xf32>, vector<1x16xf32>
      %t4 = vector.shuffle %t0, %t1 [0, 1, 2, 3] : vector<2x16xf32>, vector<2x16xf32>
      %t5 = vector.shuffle %t2, %t3 [0, 1, 2, 3] : vector<2x16xf32>, vector<2x16xf32>
      %t6 = vector.shuffle %t4, %t5 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4x16xf32>, vector<4x16xf32>
      // store
      %out_tile = xegpu.create_nd_tdesc %Out [%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %t6, %out_tile  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // init constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %rand_lower = arith.constant -1.0 : f32
    %rand_upper = arith.constant 1.0 : f32
    %gen_int = arith.constant 0 : i1
    %A = memref.alloc() : memref<8x16xf16>
    %B = memref.alloc() : memref<16x16xf16>
    %Out_cpu = memref.alloc() : memref<8x16xf32>
    %A_random = memref.cast %A : memref<8x16xf16> to memref<*xf16>
    %B_random = memref.cast %B : memref<16x16xf16> to memref<*xf16>
    call @fillResource1DRandomF16(%A_random, %rand_lower, %rand_upper, %gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    call @fillResource1DRandomF16(%B_random, %rand_lower, %rand_upper, %gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    // run GPU version
    %Out_gpu_large, %Out_gpu_generic = call @test(%A, %B) : (memref<8x16xf16>, memref<16x16xf16>) -> (memref<8x16xf32>, memref<8x16xf32>)
    %Out_gpu_generic_cast = memref.cast %Out_gpu_generic : memref<8x16xf32> to memref<*xf32>
    %Out_gpu_large_cast = memref.cast %Out_gpu_large : memref<8x16xf32> to memref<*xf32>
    // run CPU version
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to  %c16 step %c1 {
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
        %vexp = math.exp %result#0: f32
        memref.store %vexp, %Out_cpu[%i, %j] : memref<8x16xf32>
      }
    }
    %Out_cpu_cast = memref.cast %Out_cpu : memref<8x16xf32> to memref<*xf32>
    // print GPU and CPU outs
    // call @printMemrefF32(%Out_cpu_cast) : (memref<*xf32>) -> ()
    // call @printMemrefF32(%Out_gpu_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%Out_gpu_generic_cast, %Out_cpu_cast) : (memref<*xf32>, memref<*xf32>) -> ()
    call @printAllcloseF32(%Out_gpu_large_cast, %Out_cpu_cast) : (memref<*xf32>, memref<*xf32>) -> ()
    // dealloc
    memref.dealloc %A : memref<8x16xf16>
    memref.dealloc %B : memref<16x16xf16>
    memref.dealloc %Out_cpu : memref<8x16xf32>
    // gpu dealloc
    gpu.dealloc %Out_gpu_generic : memref<8x16xf32>
    gpu.dealloc %Out_gpu_large : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
