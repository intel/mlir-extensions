// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck


module @quantize attributes {gpu.container_module} {
  memref.global "private" constant @__constant_3xf32 : memref<3xf32> = dense<[1.000000e-01, 4.000000e-01, 3.000000e-01]>
  func.func @test(%arg0: memref<3xf32>) -> memref<3xi32> attributes {llvm.emit_c_interface} {
    %c3 = arith.constant 3 : index
    %cst = arith.constant 2.560000e+02 : f32
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<3xf32>
    memref.copy %arg0, %memref : memref<3xf32> to memref<3xf32>
    %memref_0 = gpu.alloc  () : memref<3xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c3, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<3xf32>, %cst : f32, %memref_0 : memref<3xf32>)
    %memref_1 = gpu.alloc  host_shared () : memref<3xi32>
    gpu.launch_func  @test_kernel_0::@test_kernel blocks in (%c3, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref_0 : memref<3xf32>, %memref_1 : memref<3xi32>)
    gpu.dealloc  %memref_0 : memref<3xf32>
    gpu.dealloc  %memref : memref<3xf32>
    return %memref_1 : memref<3xi32>
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<3 x f32>, CrossWorkgroup>, %arg1: f32, %arg2: !spirv.ptr<!spirv.array<3 x f32>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 3, 1, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %cst0_i64 = spirv.Constant 0 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %2 = spirv.IMul %cst1_i64, %1 : i64
      %3 = spirv.IAdd %cst0_i64, %2 : i64
      %4 = spirv.AccessChain %arg0[%3] : !spirv.ptr<!spirv.array<3 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      %5 = spirv.Load "CrossWorkgroup" %4 : f32
      %6 = spirv.FMul %5, %arg1 : f32
      %cst0_i64_0 = spirv.Constant 0 : i64
      %cst1_i64_1 = spirv.Constant 1 : i64
      %7 = spirv.IMul %cst1_i64_1, %1 : i64
      %8 = spirv.IAdd %cst0_i64_0, %7 : i64
      %9 = spirv.AccessChain %arg2[%8] : !spirv.ptr<!spirv.array<3 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %9, %6 : f32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<3xf32>, %arg1: f32, %arg2: memref<3xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 3, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = memref.load %arg0[%0] : memref<3xf32>
      %2 = arith.mulf %1, %arg1 : f32
      memref.store %2, %arg2[%0] : memref<3xf32>
      gpu.return
    }
  }
  spirv.module @__spv__test_kernel_0 Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<3 x f32>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<3 x i32>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 3, 1, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %cst0_i64 = spirv.Constant 0 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %2 = spirv.IMul %cst1_i64, %1 : i64
      %3 = spirv.IAdd %cst0_i64, %2 : i64
      %4 = spirv.AccessChain %arg0[%3] : !spirv.ptr<!spirv.array<3 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      %5 = spirv.Load "CrossWorkgroup" %4 : f32
      %6 = spirv.ConvertFToS %5 : f32 to i32
      %cst0_i64_0 = spirv.Constant 0 : i64
      %cst1_i64_1 = spirv.Constant 1 : i64
      %7 = spirv.IMul %cst1_i64_1, %1 : i64
      %8 = spirv.IAdd %cst0_i64_0, %7 : i64
      %9 = spirv.AccessChain %arg1[%8] : !spirv.ptr<!spirv.array<3 x i32>, CrossWorkgroup>, i64 -> !spirv.ptr<i32, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %9, %6 : i32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel_0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<3xf32>, %arg1: memref<3xi32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 3, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = memref.load %arg0[%0] : memref<3xf32>
      %2 = arith.fptosi %1 : f32 to i32
      memref.store %2, %arg1[%0] : memref<3xi32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    scf.for %arg0 = %c0 to %c100 step %c1 {
      %0 = memref.get_global @__constant_3xf32 : memref<3xf32>
      %1 = func.call @test(%0) : (memref<3xf32>) -> memref<3xi32>
      %cast = memref.cast %1 : memref<3xi32> to memref<*xi32>
      func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
      // CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
      // CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [3] strides = {{.*}} data =
      // CHECK:   25
      // CHECK:   102
      // CHECK:   76
    }
    return
  }
  func.func private @printMemrefI32(memref<*xi32>) attributes {llvm.emit_c_interface}
}
