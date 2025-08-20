// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck


module @transpose attributes {gpu.container_module} {
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
  memref.global "private" constant @__constant_20x10xf32_ref_result : memref<20x10xf32> = dense<[
    [1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0],
    [2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0,   2.0],
    [3.0,   3.0,   3.0,   3.0,   3.0,   3.0,   3.0,   3.0,   3.0,   3.0],
    [4.0,   4.0,   4.0,   4.0,   4.0,   4.0,   4.0,   4.0,   4.0,   4.0],
    [5.0,   5.0,   5.0,   5.0,   5.0,   5.0,   5.0,   5.0,   5.0,   5.0],
    [6.0,   6.0,   6.0,   6.0,   6.0,   6.0,   6.0,   6.0,   6.0,   6.0],
    [7.0,   7.0,   7.0,   7.0,   7.0,   7.0,   7.0,   7.0,   7.0,   7.0],
    [8.0,   8.0,   8.0,   8.0,   8.0,   8.0,   8.0,   8.0,   8.0,   8.0],
    [9.0,   9.0,   9.0,   9.0,   9.0,   9.0,   9.0,   9.0,   9.0,   9.0],
    [10.0,   10.0,   10.0,   10.0,   10.0,   10.0,   10.0,   10.0,   10.0,   10.0],
    [11.0,   11.0,   11.0,   11.0,   11.0,   11.0,   11.0,   11.0,   11.0,   11.0],
    [12.0,   12.0,   12.0,   12.0,   12.0,   12.0,   12.0,   12.0,   12.0,   12.0],
    [13.0,   13.0,   13.0,   13.0,   13.0,   13.0,   13.0,   13.0,   13.0,   13.0],
    [14.0,   14.0,   14.0,   14.0,   14.0,   14.0,   14.0,   14.0,   14.0,   14.0],
    [15.0,   15.0,   15.0,   15.0,   15.0,   15.0,   15.0,   15.0,   15.0,   15.0],
    [16.0,   16.0,   16.0,   16.0,   16.0,   16.0,   16.0,   16.0,   16.0,   16.0],
    [17.0,   17.0,   17.0,   17.0,   17.0,   17.0,   17.0,   17.0,   17.0,   17.0],
    [18.0,   18.0,   18.0,   18.0,   18.0,   18.0,   18.0,   18.0,   18.0,   18.0],
    [19.0,   19.0,   19.0,   19.0,   19.0,   19.0,   19.0,   19.0,   19.0,   19.0],
    [20.0,   20.0,   20.0,   20.0,   20.0,   20.0,   20.0,   20.0,   20.0,   20.0]
  ]>

  func.func @test(%arg0: memref<10x20xf32>) -> memref<20x10xf32> attributes {llvm.emit_c_interface} {
    %c10 = arith.constant 10 : index
    %c20 = arith.constant 20 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<10x20xf32>
    memref.copy %arg0, %memref : memref<10x20xf32> to memref<10x20xf32>
    %memref_0 = gpu.alloc  () : memref<20x10xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c20, %c10, %c1) threads in (%c1, %c1, %c1) args(%cst : f32, %memref_0 : memref<20x10xf32>)
    %memref_1 = gpu.alloc  host_shared () : memref<20x10xf32>
    gpu.launch_func  @test_kernel_0::@test_kernel blocks in (%c20, %c10, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<10x20xf32>, %memref_1 : memref<20x10xf32>)
    gpu.dealloc  %memref_0 : memref<20x10xf32>
    gpu.dealloc  %memref : memref<10x20xf32>
    return %memref_1 : memref<20x10xf32>
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: f32, %arg1: !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 20, 10, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi64>
      %cst0_i64 = spirv.Constant 0 : i64
      %cst10_i64 = spirv.Constant 10 : i64
      %4 = spirv.IMul %cst10_i64, %1 : i64
      %5 = spirv.IAdd %cst0_i64, %4 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %6 = spirv.IMul %cst1_i64, %3 : i64
      %7 = spirv.IAdd %5, %6 : i64
      %8 = spirv.AccessChain %arg1[%7] : !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %8, %arg0 : f32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: f32, %arg1: memref<20x10xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 20, 10, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      memref.store %arg0, %arg1[%0, %1] : memref<20x10xf32>
      gpu.return
    }
  }
  spirv.module @__spv__test_kernel_0 Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 20, 10, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi64>
      %cst0_i64 = spirv.Constant 0 : i64
      %cst20_i64 = spirv.Constant 20 : i64
      %4 = spirv.IMul %cst20_i64, %3 : i64
      %5 = spirv.IAdd %cst0_i64, %4 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %6 = spirv.IMul %cst1_i64, %1 : i64
      %7 = spirv.IAdd %5, %6 : i64
      %8 = spirv.AccessChain %arg0[%7] : !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      %9 = spirv.Load "CrossWorkgroup" %8 : f32
      %cst0_i64_1 = spirv.Constant 0 : i64
      %cst10_i64 = spirv.Constant 10 : i64
      %10 = spirv.IMul %cst10_i64, %1 : i64
      %11 = spirv.IAdd %cst0_i64_1, %10 : i64
      %cst1_i64_2 = spirv.Constant 1 : i64
      %12 = spirv.IMul %cst1_i64_2, %3 : i64
      %13 = spirv.IAdd %11, %12 : i64
      %14 = spirv.AccessChain %arg1[%13] : !spirv.ptr<!spirv.array<200 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %14, %9 : f32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel_0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<10x20xf32>, %arg1: memref<20x10xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 20, 10, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%1, %0] : memref<10x20xf32>
      memref.store %2, %arg1[%0, %1] : memref<20x10xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index

    %0 = memref.get_global @__constant_10x20xf32 : memref<10x20xf32>
    %ref_result = memref.get_global @__constant_20x10xf32_ref_result : memref<20x10xf32>
    %unranked_ref_result = memref.cast %ref_result : memref<20x10xf32> to memref<*xf32>

    scf.for %arg0 = %c0 to %c100 step %c1 {
      %1 = func.call @test(%0) : (memref<10x20xf32>) -> memref<20x10xf32>
      %cast = memref.cast %1 : memref<20x10xf32> to memref<*xf32>
      func.call @printAllcloseF32(%cast, %unranked_ref_result) : (memref<*xf32>, memref<*xf32>) -> ()
      func.call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
      // CHECK:   [ALLCLOSE: TRUE]
    }
    return
  }
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
