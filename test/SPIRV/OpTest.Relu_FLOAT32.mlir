// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck


module @relu attributes {gpu.container_module} {
  memref.global "private" constant @__constant_4x5xf32 : memref<4x5xf32> = dense<[
    [-1.000000e-01, -2.000000e-01, -3.000000e-01, 4.000000e-01, 5.000000e-01],
    [1.000000e-01, -2.000000e-01, 3.000000e-01, -4.000000e-01, 5.000000e-01],
    [1.000000e-01, 2.000000e-01, 3.000000e-01, -4.000000e-01, -5.000000e-01],
    [1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01]
  ]>
  memref.global "private" constant @__constant_4x5xf32_ref_result : memref<4x5xf32> = dense<[
    [0.0, 0.0, 0.0, 4.000000e-01, 5.000000e-01],
    [1.000000e-01, 0.0, 3.000000e-01, 0.0, 5.000000e-01],
    [1.000000e-01, 2.000000e-01, 3.000000e-01, 0.0, 0.0],
    [1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01]
  ]>

  func.func @main() attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_4x5xf32 : memref<4x5xf32>
    %ref_result = memref.get_global @__constant_4x5xf32_ref_result : memref<4x5xf32>
    %unranked_ref_result =  memref.cast %ref_result : memref<4x5xf32> to memref<*xf32>

    scf.for %arg0 = %c0 to %c100 step %c1 {
      %1 = func.call @test(%0) : (memref<4x5xf32>) -> memref<4x5xf32>
      %cast = memref.cast %1 : memref<4x5xf32> to memref<*xf32>
      func.call @printAllcloseF32(%cast, %unranked_ref_result) : (memref<*xf32>, memref<*xf32>) -> ()
      func.call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
      // CHECK:   [ALLCLOSE: TRUE]
    }
    return
  }

  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func @test(%arg0: memref<4x5xf32>) -> memref<4x5xf32> attributes {llvm.emit_c_interface} {
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<4x5xf32>
    memref.copy %arg0, %memref : memref<4x5xf32> to memref<4x5xf32>
    %memref_0 = gpu.alloc  () : memref<4x5xi1>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c4, %c5, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<4x5xf32>, %cst : f32, %memref_0 : memref<4x5xi1>)
    %memref_1 = gpu.alloc  host_shared () : memref<4x5xf32>
    gpu.launch_func  @test_kernel_0::@test_kernel blocks in (%c4, %c5, %c1) threads in (%c1, %c1, %c1) args(%memref_0 : memref<4x5xi1>, %memref : memref<4x5xf32>, %cst : f32, %memref_1 : memref<4x5xf32>)
    gpu.dealloc  %memref_0 : memref<4x5xi1>
    gpu.dealloc  %memref : memref<4x5xf32>
    return %memref_1 : memref<4x5xf32>
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Int8, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<20 x f32>, CrossWorkgroup>, %arg1: f32, %arg2: !spirv.ptr<!spirv.array<20 x i8>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 5, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi64>
      %cst0_i64 = spirv.Constant 0 : i64
      %cst5_i64 = spirv.Constant 5 : i64
      %4 = spirv.IMul %cst5_i64, %1 : i64
      %5 = spirv.IAdd %cst0_i64, %4 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %6 = spirv.IMul %cst1_i64, %3 : i64
      %7 = spirv.IAdd %5, %6 : i64
      %8 = spirv.AccessChain %arg0[%7] : !spirv.ptr<!spirv.array<20 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      %9 = spirv.Load "CrossWorkgroup" %8 : f32
      %10 = spirv.FOrdLessThan %9, %arg1 : f32
      %cst0_i64_1 = spirv.Constant 0 : i64
      %cst5_i64_2 = spirv.Constant 5 : i64
      %11 = spirv.IMul %cst5_i64_2, %1 : i64
      %12 = spirv.IAdd %cst0_i64_1, %11 : i64
      %cst1_i64_3 = spirv.Constant 1 : i64
      %13 = spirv.IMul %cst1_i64_3, %3 : i64
      %14 = spirv.IAdd %12, %13 : i64
      %15 = spirv.AccessChain %arg2[%14] : !spirv.ptr<!spirv.array<20 x i8>, CrossWorkgroup>, i64 -> !spirv.ptr<i8, CrossWorkgroup>
      %cst0_i8 = spirv.Constant 0 : i8
      %cst1_i8 = spirv.Constant 1 : i8
      %16 = spirv.Select %10, %cst1_i8, %cst0_i8 : i1, i8
      spirv.Store "CrossWorkgroup" %15, %16 : i8
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<4x5xf32>, %arg1: f32, %arg2: memref<4x5xi1>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 5, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<4x5xf32>
      %3 = arith.cmpf olt, %2, %arg1 : f32
      memref.store %3, %arg2[%0, %1] : memref<4x5xi1>
      gpu.return
    }
  }
  spirv.module @__spv__test_kernel_0 Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Int8, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<20 x i8>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<20 x f32>, CrossWorkgroup>, %arg2: f32, %arg3: !spirv.ptr<!spirv.array<20 x f32>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 5, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi64>
      %cst0_i64 = spirv.Constant 0 : i64
      %cst5_i64 = spirv.Constant 5 : i64
      %4 = spirv.IMul %cst5_i64, %1 : i64
      %5 = spirv.IAdd %cst0_i64, %4 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %6 = spirv.IMul %cst1_i64, %3 : i64
      %7 = spirv.IAdd %5, %6 : i64
      %8 = spirv.AccessChain %arg0[%7] : !spirv.ptr<!spirv.array<20 x i8>, CrossWorkgroup>, i64 -> !spirv.ptr<i8, CrossWorkgroup>
      %9 = spirv.Load "CrossWorkgroup" %8 : i8
      %cst1_i8 = spirv.Constant 1 : i8
      %10 = spirv.IEqual %9, %cst1_i8 : i8
      %cst0_i64_1 = spirv.Constant 0 : i64
      %cst5_i64_2 = spirv.Constant 5 : i64
      %11 = spirv.IMul %cst5_i64_2, %1 : i64
      %12 = spirv.IAdd %cst0_i64_1, %11 : i64
      %cst1_i64_3 = spirv.Constant 1 : i64
      %13 = spirv.IMul %cst1_i64_3, %3 : i64
      %14 = spirv.IAdd %12, %13 : i64
      %15 = spirv.AccessChain %arg1[%14] : !spirv.ptr<!spirv.array<20 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      %16 = spirv.Load "CrossWorkgroup" %15 : f32
      %17 = spirv.Select %10, %arg2, %16 : i1, f32
      %cst0_i64_4 = spirv.Constant 0 : i64
      %cst5_i64_5 = spirv.Constant 5 : i64
      %18 = spirv.IMul %cst5_i64_5, %1 : i64
      %19 = spirv.IAdd %cst0_i64_4, %18 : i64
      %cst1_i64_6 = spirv.Constant 1 : i64
      %20 = spirv.IMul %cst1_i64_6, %3 : i64
      %21 = spirv.IAdd %19, %20 : i64
      %22 = spirv.AccessChain %arg3[%21] : !spirv.ptr<!spirv.array<20 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %22, %17 : f32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel_0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<4x5xi1>, %arg1: memref<4x5xf32>, %arg2: f32, %arg3: memref<4x5xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 5, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<4x5xi1>
      %3 = memref.load %arg1[%0, %1] : memref<4x5xf32>
      %4 = arith.select %2, %arg2, %3 : f32
      memref.store %4, %arg3[%0, %1] : memref<4x5xf32>
      gpu.return
    }
  }
}
