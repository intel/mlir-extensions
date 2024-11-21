// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck


module @explicit_padding attributes {gpu.container_module} {
  memref.global "private" constant @__constant_2x3xf32 : memref<2x3xf32> = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]>
  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index

    %0 = memref.get_global @__constant_2x3xf32 : memref<2x3xf32>
    scf.for %arg0 = %c0 to %c100 step %c1 {
      %1 = func.call @test(%0) : (memref<2x3xf32>) -> memref<6x5xf32>
      %cast = memref.cast %1 : memref<6x5xf32> to memref<*xf32>
      func.call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    }
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func @test(%arg0: memref<2x3xf32>) -> memref<6x5xf32> attributes {llvm.emit_c_interface} {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<2x3xf32>
    memref.copy %arg0, %memref : memref<2x3xf32> to memref<2x3xf32>
    %memref_0 = gpu.alloc  host_shared () : memref<6x5xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c6, %c5, %c1) threads in (%c1, %c1, %c1) args(%cst : f32, %memref_0 : memref<6x5xf32>)
    %memref_1 = gpu.alloc  host_shared () : memref<6x5xf32>
    memref.copy %memref_0, %memref_1 : memref<6x5xf32> to memref<6x5xf32>
    gpu.launch_func  @test_kernel_0::@test_kernel blocks in (%c2, %c3, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<2x3xf32>, %memref_1 : memref<6x5xf32>)
    gpu.dealloc  %memref_0 : memref<6x5xf32>
    gpu.dealloc  %memref : memref<2x3xf32>
    return %memref_1 : memref<6x5xf32>
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: f32, %arg1: !spirv.ptr<!spirv.array<30 x f32>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 6, 5, 1>, workgroup_attributions = 0 : i64} {
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
      %8 = spirv.AccessChain %arg1[%7] : !spirv.ptr<!spirv.array<30 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %8, %arg0 : f32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: f32, %arg1: memref<6x5xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 6, 5, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      memref.store %arg0, %arg1[%0, %1] : memref<6x5xf32>
      gpu.return
    }
  }
  spirv.module @__spv__test_kernel_0 Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<6 x f32>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<30 x f32>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 2, 3, 1>, workgroup_attributions = 0 : i64} {
      %cst1_i64 = spirv.Constant 1 : i64
      %cst2_i64 = spirv.Constant 2 : i64
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi64>
      %cst0_i64 = spirv.Constant 0 : i64
      %cst3_i64 = spirv.Constant 3 : i64
      %4 = spirv.IMul %cst3_i64, %1 : i64
      %5 = spirv.IAdd %cst0_i64, %4 : i64
      %cst1_i64_1 = spirv.Constant 1 : i64
      %6 = spirv.IMul %cst1_i64_1, %3 : i64
      %7 = spirv.IAdd %5, %6 : i64
      %8 = spirv.AccessChain %arg0[%7] : !spirv.ptr<!spirv.array<6 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      %9 = spirv.Load "CrossWorkgroup" %8 : f32
      %10 = spirv.IAdd %1, %cst2_i64 : i64
      %11 = spirv.IAdd %3, %cst1_i64 : i64
      %cst0_i64_2 = spirv.Constant 0 : i64
      %cst5_i64 = spirv.Constant 5 : i64
      %12 = spirv.IMul %cst5_i64, %10 : i64
      %13 = spirv.IAdd %cst0_i64_2, %12 : i64
      %cst1_i64_3 = spirv.Constant 1 : i64
      %14 = spirv.IMul %cst1_i64_3, %11 : i64
      %15 = spirv.IAdd %13, %14 : i64
      %16 = spirv.AccessChain %arg1[%15] : !spirv.ptr<!spirv.array<30 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      %17 = spirv.Load "CrossWorkgroup" %16 : f32
      %18 = spirv.FAdd %9, %17 : f32
      %cst0_i64_4 = spirv.Constant 0 : i64
      %cst5_i64_5 = spirv.Constant 5 : i64
      %19 = spirv.IMul %cst5_i64_5, %10 : i64
      %20 = spirv.IAdd %cst0_i64_4, %19 : i64
      %cst1_i64_6 = spirv.Constant 1 : i64
      %21 = spirv.IMul %cst1_i64_6, %11 : i64
      %22 = spirv.IAdd %20, %21 : i64
      %23 = spirv.AccessChain %arg1[%22] : !spirv.ptr<!spirv.array<30 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %23, %18 : f32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel_0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<2x3xf32>, %arg1: memref<6x5xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 2, 3, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<2x3xf32>
      %3 = arith.addi %0, %c2 : index
      %4 = arith.addi %1, %c1 : index
      %5 = memref.load %arg1[%3, %4] : memref<6x5xf32>
      %6 = arith.addf %2, %5 : f32
      memref.store %6, %arg1[%3, %4] : memref<6x5xf32>
      gpu.return
    }
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [6, 5] strides = {{.*}} data =
// CHECK-NEXT: [0, 0, 0, 0, 0]
// CHECK-NEXT: [0, 0, 0, 0, 0]
// CHECK-NEXT: [0, 1, 2, 3, 0]
// CHECK-NEXT: [0, 4, 5, 6, 0]
// CHECK-NEXT: [0, 0, 0, 0, 0]
// CHECK-NEXT: [0, 0, 0, 0, 0]
