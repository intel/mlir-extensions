// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck


module @eltwise_add attributes {gpu.container_module} {
  memref.global "private" constant @__constant_10x20xf32_0 : memref<10x20xf32> = dense<5.000000e-01>
  memref.global "private" constant @__constant_10x20xf32 : memref<10x20xf32> = dense<[
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
  ]>
  memref.global "private" constant @__constant_10x20xf32_ref_result : memref<10x20xf32> = dense<[
    [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5],
    [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5],
    [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5],
    [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5],
    [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5],
    [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5],
    [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5],
    [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5],
    [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5],
    [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5]
  ]>
  func.func @test(%arg0: memref<10x20xf32>, %arg1: memref<10x20xf32>) -> memref<10x20xf32> attributes {llvm.emit_c_interface} {
    %c20 = arith.constant 20 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index

    // Explicit memory copy to and from host
    %memref_arg0_gpu = gpu.alloc () : memref<10x20xf32>
    gpu.memcpy %memref_arg0_gpu, %arg0 : memref<10x20xf32>, memref<10x20xf32>

    %memref_arg1_gpu = gpu.alloc () : memref<10x20xf32>
    gpu.memcpy %memref_arg1_gpu, %arg1 : memref<10x20xf32>, memref<10x20xf32>

    %memref_result_gpu = gpu.alloc () : memref<10x20xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c10, %c20, %c1) threads in (%c1, %c1, %c1) args(%memref_arg0_gpu : memref<10x20xf32>, %memref_arg1_gpu : memref<10x20xf32>, %memref_result_gpu : memref<10x20xf32>)
    %memref_result_host = memref.alloc () : memref<10x20xf32>
    gpu.memcpy %memref_result_host, %memref_result_gpu : memref<10x20xf32>, memref<10x20xf32>
    gpu.dealloc  %memref_arg0_gpu : memref<10x20xf32>
    gpu.dealloc  %memref_arg1_gpu : memref<10x20xf32>
    gpu.dealloc  %memref_result_gpu : memref<10x20xf32>

    return %memref_result_host : memref<10x20xf32>
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, %arg2: !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 20, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi64>
      %cst0_i64 = spirv.Constant 0 : i64
      %cst20_i64 = spirv.Constant 20 : i64
      %4 = spirv.IMul %cst20_i64, %1 : i64
      %5 = spirv.IAdd %cst0_i64, %4 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %6 = spirv.IMul %cst1_i64, %3 : i64
      %7 = spirv.IAdd %5, %6 : i64
      %8 = spirv.AccessChain %arg0[%7] : !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, i64
      %9 = spirv.Load "CrossWorkgroup" %8 : f32
      %cst0_i64_1 = spirv.Constant 0 : i64
      %cst20_i64_2 = spirv.Constant 20 : i64
      %10 = spirv.IMul %cst20_i64_2, %1 : i64
      %11 = spirv.IAdd %cst0_i64_1, %10 : i64
      %cst1_i64_3 = spirv.Constant 1 : i64
      %12 = spirv.IMul %cst1_i64_3, %3 : i64
      %13 = spirv.IAdd %11, %12 : i64
      %14 = spirv.AccessChain %arg1[%13] : !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, i64
      %15 = spirv.Load "CrossWorkgroup" %14 : f32
      %16 = spirv.FAdd %9, %15 : f32
      %cst0_i64_4 = spirv.Constant 0 : i64
      %cst20_i64_5 = spirv.Constant 20 : i64
      %17 = spirv.IMul %cst20_i64_5, %1 : i64
      %18 = spirv.IAdd %cst0_i64_4, %17 : i64
      %cst1_i64_6 = spirv.Constant 1 : i64
      %19 = spirv.IMul %cst1_i64_6, %3 : i64
      %20 = spirv.IAdd %18, %19 : i64
      %21 = spirv.AccessChain %arg2[%20] : !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, i64
      spirv.Store "CrossWorkgroup" %21, %16 : f32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<10x20xf32>, %arg1: memref<10x20xf32>, %arg2: memref<10x20xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 10, 20, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<10x20xf32>
      %3 = memref.load %arg1[%0, %1] : memref<10x20xf32>
      %4 = arith.addf %2, %3 : f32
      memref.store %4, %arg2[%0, %1] : memref<10x20xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index

    %0 = memref.get_global @__constant_10x20xf32 : memref<10x20xf32>
    %1 = memref.get_global @__constant_10x20xf32_0 : memref<10x20xf32>
    %ref_result = memref.get_global @__constant_10x20xf32_ref_result : memref<10x20xf32>
    %unranked_ref_result = memref.cast %ref_result : memref<10x20xf32> to memref<*xf32>

    scf.for %arg0 = %c0 to %c100 step %c1 {
      %2 = func.call @test(%0, %1) : (memref<10x20xf32>, memref<10x20xf32>) -> memref<10x20xf32>
      %cast = memref.cast %2 : memref<10x20xf32> to memref<*xf32>
      func.call @printAllcloseF32(%cast, %unranked_ref_result) : (memref<*xf32>, memref<*xf32>) -> ()
      func.call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
      // CHECK:   [ALLCLOSE: TRUE]
    }
    return
  }
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
