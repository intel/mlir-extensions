// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @test attributes {gpu.container_module} {
  memref.global "private" constant @__constant_8x32xf32 : memref<8x32xf32> = dense<[
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [-1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [-2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1]
    ]>
  func.func @test(%arg0: memref<8x32xf32>) -> memref<8x32xf32> attributes {llvm.emit_c_interface} {
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<8x32xf32>
    memref.copy %arg0, %memref : memref<8x32xf32> to memref<8x32xf32>
    %memref_0 = gpu.alloc  host_shared () : memref<8x32xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c8, %c32, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8x32xf32>, %cst : f32, %memref_0 : memref<8x32xf32>)
    %alloc = memref.alloc() : memref<8x32xf32>
    memref.copy %memref_0, %alloc : memref<8x32xf32> to memref<8x32xf32>
    gpu.dealloc  %memref_0 : memref<8x32xf32>
    gpu.dealloc  %memref : memref<8x32xf32>
    return %alloc : memref<8x32xf32>
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__workgroup_mem__1 : !spirv.ptr<!spirv.array<512 x f32>, Workgroup>
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<256 x f32>, CrossWorkgroup>, %arg1: f32, %arg2: !spirv.ptr<!spirv.array<256 x f32>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 8, 32, 1>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi64>
      %cst0_i64 = spirv.Constant 0 : i64
      %cst32_i64 = spirv.Constant 32 : i64
      %4 = spirv.IMul %cst32_i64, %1 : i64
      %5 = spirv.IAdd %cst0_i64, %4 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %6 = spirv.IMul %cst1_i64, %3 : i64
      %7 = spirv.IAdd %5, %6 : i64
      %8 = spirv.AccessChain %arg0[%7] : !spirv.ptr<!spirv.array<256 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      %9 = spirv.Load "CrossWorkgroup" %8 : f32
      %10 = spirv.FUnordGreaterThan %9, %arg1 : f32
      %11 = spirv.Select %10, %9, %arg1 : i1, f32
      %__workgroup_mem__1_addr = spirv.mlir.addressof @__workgroup_mem__1 : !spirv.ptr<!spirv.array<512 x f32>, Workgroup>
      %cst0_i64_1 = spirv.Constant 0 : i64
      %cst32_i64_2 = spirv.Constant 32 : i64
      %12 = spirv.IMul %cst32_i64_2, %1 : i64
      %13 = spirv.IAdd %cst0_i64_1, %12 : i64
      %cst1_i64_3 = spirv.Constant 1 : i64
      %14 = spirv.IMul %cst1_i64_3, %3 : i64
      %15 = spirv.IAdd %13, %14 : i64
      %16 = spirv.AccessChain %__workgroup_mem__1_addr[%15] : !spirv.ptr<!spirv.array<512 x f32>, Workgroup>, i64 -> !spirv.ptr<f32, Workgroup>
      spirv.Store "Workgroup" %16, %11 : f32
      spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
      %cst0_i64_4 = spirv.Constant 0 : i64
      %cst32_i64_5 = spirv.Constant 32 : i64
      %17 = spirv.IMul %cst32_i64_5, %1 : i64
      %18 = spirv.IAdd %cst0_i64_4, %17 : i64
      %cst1_i64_6 = spirv.Constant 1 : i64
      %19 = spirv.IMul %cst1_i64_6, %3 : i64
      %20 = spirv.IAdd %18, %19 : i64
      %21 = spirv.AccessChain %__workgroup_mem__1_addr[%20] : !spirv.ptr<!spirv.array<512 x f32>, Workgroup>, i64 -> !spirv.ptr<f32, Workgroup>
      %22 = spirv.Load "Workgroup" %21 : f32
      %cst0_i64_7 = spirv.Constant 0 : i64
      %cst32_i64_8 = spirv.Constant 32 : i64
      %23 = spirv.IMul %cst32_i64_8, %1 : i64
      %24 = spirv.IAdd %cst0_i64_7, %23 : i64
      %cst1_i64_9 = spirv.Constant 1 : i64
      %25 = spirv.IMul %cst1_i64_9, %3 : i64
      %26 = spirv.IAdd %24, %25 : i64
      %27 = spirv.AccessChain %arg2[%26] : !spirv.ptr<!spirv.array<256 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %27, %22 : f32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<8x32xf32>, %arg1: f32, %arg2: memref<8x32xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 8, 32, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0, %1] : memref<8x32xf32>
      %3 = arith.cmpf ugt, %2, %arg1 : f32
      %4 = arith.select %3, %2, %arg1 : f32
      %alloc = memref.alloc() : memref<16x32xf32, #spirv.storage_class<Workgroup>>
      memref.store %4, %alloc[%0, %1] : memref<16x32xf32, #spirv.storage_class<Workgroup>>
      gpu.barrier
      %5 = memref.load %alloc[%0, %1] : memref<16x32xf32, #spirv.storage_class<Workgroup>>
      memref.store %5, %arg2[%0, %1] : memref<8x32xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_8x32xf32 : memref<8x32xf32>
    %1 = call @test(%0) : (memref<8x32xf32>) -> memref<8x32xf32>
    %cast = memref.cast %1 : memref<8x32xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    // CHECK: [0, 0
    // CHECK: [1.1, 1.1
    // CHECK: [0, 0
    // CHECK: [2.1, 2.1
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
