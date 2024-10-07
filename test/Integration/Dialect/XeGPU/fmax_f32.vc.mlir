// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp  \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp  \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<8x32xf16>, %B: memref<16x32xf16> ) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<8x32xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<16x32xf16>
    memref.copy %A, %memref : memref<8x32xf16> to memref<8x32xf16>
    memref.copy %B, %memref_1 : memref<16x32xf16> to memref<16x32xf16>
    %memref_2 = gpu.alloc  host_shared () : memref<8x16xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8x32xf16>, %memref_1 : memref<16x32xf16>, %memref_2 : memref<8x16xf32>)
    gpu.dealloc  %memref : memref<8x32xf16>
    gpu.dealloc  %memref_1 : memref<16x32xf16>
    return %memref_2 : memref<8x16xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<8x32xf16>, %B: memref<16x32xf16>, %Out: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %a_tile0 = xegpu.create_nd_tdesc %A [%c0, %c0] : memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16>
      %a_tile1 = xegpu.create_nd_tdesc %A [%c0, %c16] : memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16>
      // load A tiles
      %val0 = xegpu.load_nd %a_tile0 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %val1 = xegpu.load_nd %a_tile1 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %b_tile0 = xegpu.create_nd_tdesc %B [%c0, %c0] : memref<16x32xf16> -> !xegpu.tensor_desc<16x16xf16>
      %b_tile1 = xegpu.create_nd_tdesc %B [%c0, %c16] : memref<16x32xf16> -> !xegpu.tensor_desc<16x16xf16>
      // load B tiles
      %val2 = xegpu.load_nd %b_tile0 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
      %val3 = xegpu.load_nd %b_tile1 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
      // do DPAS
      %val4 = xegpu.dpas %val0, %val2 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
      %val5 = xegpu.dpas %val1, %val3 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
      // take fmax
      %val6 = arith.maximumf %val4, %val5 fastmath<nnan> : vector<8x16xf32>
      // store fmax
      %out_tile = xegpu.create_nd_tdesc %Out [%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %val6, %out_tile  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    // init constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index

    %A = memref.alloc() : memref<8x32xf16>
    %B = memref.alloc() : memref<16x32xf16>
    %Out_cpu = memref.alloc() : memref<8x16xf32>
    %A_random = memref.cast %A : memref<8x32xf16> to memref<*xf16>
    %B_random = memref.cast %B : memref<16x32xf16> to memref<*xf16>

    %c_gen_int = arith.constant 0 : i1
    %cf_lower = arith.constant -0.5 : f32
    %cf_upper = arith.constant 0.5 : f32

    call @fillResource1DRandomF16(%A_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()
    call @fillResource1DRandomF16(%B_random, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xf16>, f32, f32, i1) -> ()

    // run GPU version
    %Out_gpu = call @test(%A, %B) : (memref<8x32xf16>, memref<16x32xf16>) -> memref<8x16xf32>
    %Out_gpu_cast = memref.cast %Out_gpu : memref<8x16xf32> to memref<*xf32>
    // run CPU version
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to  %c16 step %c1 {
        %v0_init = arith.constant 0.0 : f32
        %v1_init = arith.constant 0.0 : f32
        %result:2 = scf.for %k = %c0 to %c16 step %c1 iter_args(%v0 = %v0_init, %v1 = %v1_init) -> (f32, f32){
          %1 = arith.addi %k, %c16 : index
          %2 = arith.addi %j, %c16 : index
          %a0 = memref.load %A[%i, %k] : memref<8x32xf16>
          %a1 = memref.load %A[%i, %1] : memref<8x32xf16>
          %b0 = memref.load %B[%k, %j] : memref<16x32xf16>
          %b1 = memref.load %B[%k, %2] : memref<16x32xf16>
          %a0_f32 = arith.extf %a0 : f16 to f32
          %a1_f32 = arith.extf %a1 : f16 to f32
          %b0_f32 = arith.extf %b0 : f16 to f32
          %b1_f32 = arith.extf %b1 : f16 to f32
          %t0 = arith.mulf %a0_f32, %b0_f32 : f32
          %t1 = arith.mulf %a1_f32, %b1_f32 : f32
          %v0_new = arith.addf %v0, %t0 : f32
          %v1_new = arith.addf %v1, %t1 : f32
          scf.yield %v0_new, %v1_new : f32, f32
        }
        %vmax = arith.maximumf %result#0, %result#1 : f32
        memref.store %vmax, %Out_cpu[%i, %j] : memref<8x16xf32>
      }
    }
    %Out_cpu_cast = memref.cast %Out_cpu : memref<8x16xf32> to memref<*xf32>
    // print GPU and CPU outs
     call @printMemrefF32(%Out_cpu_cast) : (memref<*xf32>) -> ()
     call @printMemrefF32(%Out_gpu_cast) : (memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    call @printAllcloseF32(%Out_gpu_cast, %Out_cpu_cast) : (memref<*xf32>, memref<*xf32>) -> ()
    // dealloc
    memref.dealloc %A : memref<8x32xf16>
    memref.dealloc %B : memref<16x32xf16>
    memref.dealloc %Out_cpu : memref<8x16xf32>
    // gpu dealloc
    gpu.dealloc %Out_gpu : memref<8x16xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
