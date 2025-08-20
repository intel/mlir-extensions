// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

module @slm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_4x128xf32 : memref<4x128xf32> = dense<[
    [1.001, 1.002, 1.003, 1.004, 1.005, 1.006, 1.007, 1.008, 1.009,
  1.01 , 1.011, 1.012, 1.013, 1.014, 1.015, 1.016, 1.017, 1.018,
  1.019, 1.02 , 1.021, 1.022, 1.023, 1.024, 1.025, 1.026, 1.027,
  1.028, 1.029, 1.03 , 1.031, 1.032, 1.033, 1.034, 1.035, 1.036,
  1.037, 1.038, 1.039, 1.04 , 1.041, 1.042, 1.043, 1.044, 1.045,
  1.046, 1.047, 1.048, 1.049, 1.05 , 1.051, 1.052, 1.053, 1.054,
  1.055, 1.056, 1.057, 1.058, 1.059, 1.06 , 1.061, 1.062, 1.063,
  1.064, 1.065, 1.066, 1.067, 1.068, 1.069, 1.07 , 1.071, 1.072,
  1.073, 1.074, 1.075, 1.076, 1.077, 1.078, 1.079, 1.08 , 1.081,
  1.082, 1.083, 1.084, 1.085, 1.086, 1.087, 1.088, 1.089, 1.09 ,
  1.091, 1.092, 1.093, 1.094, 1.095, 1.096, 1.097, 1.098, 1.099,
  1.1  , 1.101, 1.102, 1.103, 1.104, 1.105, 1.106, 1.107, 1.108,
  1.109, 1.11 , 1.111, 1.112, 1.113, 1.114, 1.115, 1.116, 1.117,
  1.118, 1.119, 1.12 , 1.121, 1.122, 1.123, 1.124, 1.125, 1.126,
  1.127, 1.128],
    [3.001, 3.002, 3.003, 3.004, 3.005, 3.006, 3.007, 3.008, 3.009,
  3.01 , 3.011, 3.012, 3.013, 3.014, 3.015, 3.016, 3.017, 3.018,
  3.019, 3.02 , 3.021, 3.022, 3.023, 3.024, 3.025, 3.026, 3.027,
  3.028, 3.029, 3.03 , 3.031, 3.032, 3.033, 3.034, 3.035, 3.036,
  3.037, 3.038, 3.039, 3.04 , 3.041, 3.042, 3.043, 3.044, 3.045,
  3.046, 3.047, 3.048, 3.049, 3.05 , 3.051, 3.052, 3.053, 3.054,
  3.055, 3.056, 3.057, 3.058, 3.059, 3.06 , 3.061, 3.062, 3.063,
  3.064, 3.065, 3.066, 3.067, 3.068, 3.069, 3.07 , 3.071, 3.072,
  3.073, 3.074, 3.075, 3.076, 3.077, 3.078, 3.079, 3.08 , 3.081,
  3.082, 3.083, 3.084, 3.085, 3.086, 3.087, 3.088, 3.089, 3.09 ,
  3.091, 3.092, 3.093, 3.094, 3.095, 3.096, 3.097, 3.098, 3.099,
  3.1  , 3.101, 3.102, 3.103, 3.104, 3.105, 3.106, 3.107, 3.108,
  3.109, 3.11 , 3.111, 3.112, 3.113, 3.114, 3.115, 3.116, 3.117,
  3.118, 3.119, 3.12 , 3.121, 3.122, 3.123, 3.124, 3.125, 3.126,
  3.127, 3.128],
    [5.001, 5.002, 5.003, 5.004, 5.005, 5.006, 5.007, 5.008, 5.009,
  5.01 , 5.011, 5.012, 5.013, 5.014, 5.015, 5.016, 5.017, 5.018,
  5.019, 5.02 , 5.021, 5.022, 5.023, 5.024, 5.025, 5.026, 5.027,
  5.028, 5.029, 5.03 , 5.031, 5.032, 5.033, 5.034, 5.035, 5.036,
  5.037, 5.038, 5.039, 5.04 , 5.041, 5.042, 5.043, 5.044, 5.045,
  5.046, 5.047, 5.048, 5.049, 5.05 , 5.051, 5.052, 5.053, 5.054,
  5.055, 5.056, 5.057, 5.058, 5.059, 5.06 , 5.061, 5.062, 5.063,
  5.064, 5.065, 5.066, 5.067, 5.068, 5.069, 5.07 , 5.071, 5.072,
  5.073, 5.074, 5.075, 5.076, 5.077, 5.078, 5.079, 5.08 , 5.081,
  5.082, 5.083, 5.084, 5.085, 5.086, 5.087, 5.088, 5.089, 5.09 ,
  5.091, 5.092, 5.093, 5.094, 5.095, 5.096, 5.097, 5.098, 5.099,
  5.1  , 5.101, 5.102, 5.103, 5.104, 5.105, 5.106, 5.107, 5.108,
  5.109, 5.11 , 5.111, 5.112, 5.113, 5.114, 5.115, 5.116, 5.117,
  5.118, 5.119, 5.12 , 5.121, 5.122, 5.123, 5.124, 5.125, 5.126,
  5.127, 5.128],
    [7.001, 7.002, 7.003, 7.004, 7.005, 7.006, 7.007, 7.008, 7.009,
  7.01 , 7.011, 7.012, 7.013, 7.014, 7.015, 7.016, 7.017, 7.018,
  7.019, 7.02 , 7.021, 7.022, 7.023, 7.024, 7.025, 7.026, 7.027,
  7.028, 7.029, 7.03 , 7.031, 7.032, 7.033, 7.034, 7.035, 7.036,
  7.037, 7.038, 7.039, 7.04 , 7.041, 7.042, 7.043, 7.044, 7.045,
  7.046, 7.047, 7.048, 7.049, 7.05 , 7.051, 7.052, 7.053, 7.054,
  7.055, 7.056, 7.057, 7.058, 7.059, 7.06 , 7.061, 7.062, 7.063,
  7.064, 7.065, 7.066, 7.067, 7.068, 7.069, 7.07 , 7.071, 7.072,
  7.073, 7.074, 7.075, 7.076, 7.077, 7.078, 7.079, 7.08 , 7.081,
  7.082, 7.083, 7.084, 7.085, 7.086, 7.087, 7.088, 7.089, 7.09 ,
  7.091, 7.092, 7.093, 7.094, 7.095, 7.096, 7.097, 7.098, 7.099,
  7.1  , 7.101, 7.102, 7.103, 7.104, 7.105, 7.106, 7.107, 7.108,
  7.109, 7.11 , 7.111, 7.112, 7.113, 7.114, 7.115, 7.116, 7.117,
  7.118, 7.119, 7.12 , 7.121, 7.122, 7.123, 7.124, 7.125, 7.126,
  7.127, 7.128]
  ]>
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_4x128xf32 : memref<4x128xf32>
    %1 = call @test(%0) : (memref<4x128xf32>) -> memref<4x128xf32>
    %cast = memref.cast %1 : memref<4x128xf32> to memref<*xf32>
    //CHECK: [1.001,   1.002,   1.003,   1.004,   1.005,   1.006,   1.007,   1.008,   1.009,   1.01,   1.011,   1.012,   1.013,   1.014,   1.015,   1.016,   1.017,   1.018,   1.019,   1.02,   1.021,   1.022,   1.023,   1.024,   1.025,   1.026,   1.027,   1.028,   1.029,   1.03,   1.031,   1.032,   1.033,   1.034,   1.035,   1.036,   1.037,   1.038,   1.039,   1.04,   1.041,   1.042,   1.043,   1.044,   1.045,   1.046,   1.047,   1.048,   1.049,   1.05,   1.051,   1.052,   1.053,   1.054,   1.055,   1.056,   1.057,   1.058,   1.059,   1.06,   1.061,   1.062,   1.063,   1.064,   1.065,   1.066,   1.067,   1.068,   1.069,   1.07,   1.071,   1.072,   1.073,   1.074,   1.075,   1.076,   1.077,   1.078,   1.079,   1.08,   1.081,   1.082,   1.083,   1.084,   1.085,   1.086,   1.087,   1.088,   1.089,   1.09,   1.091,   1.092,   1.093,   1.094,   1.095,   1.096,   1.097,   1.098,   1.099,   1.1,   1.101,   1.102,   1.103,   1.104,   1.105,   1.106,   1.107,   1.108,   1.109,   1.11,   1.111,   1.112,   1.113,   1.114,   1.115,   1.116,   1.117,   1.118,   1.119,   1.12,   1.121,   1.122,   1.123,   1.124,   1.125,   1.126,   1.127,   1.128]
    //CHECK: [3.001,   3.002,   3.003,   3.004,   3.005,   3.006,   3.007,   3.008,   3.009,   3.01,   3.011,   3.012,   3.013,   3.014,   3.015,   3.016,   3.017,   3.018,   3.019,   3.02,   3.021,   3.022,   3.023,   3.024,   3.025,   3.026,   3.027,   3.028,   3.029,   3.03,   3.031,   3.032,   3.033,   3.034,   3.035,   3.036,   3.037,   3.038,   3.039,   3.04,   3.041,   3.042,   3.043,   3.044,   3.045,   3.046,   3.047,   3.048,   3.049,   3.05,   3.051,   3.052,   3.053,   3.054,   3.055,   3.056,   3.057,   3.058,   3.059,   3.06,   3.061,   3.062,   3.063,   3.064,   3.065,   3.066,   3.067,   3.068,   3.069,   3.07,   3.071,   3.072,   3.073,   3.074,   3.075,   3.076,   3.077,   3.078,   3.079,   3.08,   3.081,   3.082,   3.083,   3.084,   3.085,   3.086,   3.087,   3.088,   3.089,   3.09,   3.091,   3.092,   3.093,   3.094,   3.095,   3.096,   3.097,   3.098,   3.099,   3.1,   3.101,   3.102,   3.103,   3.104,   3.105,   3.106,   3.107,   3.108,   3.109,   3.11,   3.111,   3.112,   3.113,   3.114,   3.115,   3.116,   3.117,   3.118,   3.119,   3.12,   3.121,   3.122,   3.123,   3.124,   3.125,   3.126,   3.127,   3.128]
    //CHECK: [5.001,   5.002,   5.003,   5.004,   5.005,   5.006,   5.007,   5.008,   5.009,   5.01,   5.011,   5.012,   5.013,   5.014,   5.015,   5.016,   5.017,   5.018,   5.019,   5.02,   5.021,   5.022,   5.023,   5.024,   5.025,   5.026,   5.027,   5.028,   5.029,   5.03,   5.031,   5.032,   5.033,   5.034,   5.035,   5.036,   5.037,   5.038,   5.039,   5.04,   5.041,   5.042,   5.043,   5.044,   5.045,   5.046,   5.047,   5.048,   5.049,   5.05,   5.051,   5.052,   5.053,   5.054,   5.055,   5.056,   5.057,   5.058,   5.059,   5.06,   5.061,   5.062,   5.063,   5.064,   5.065,   5.066,   5.067,   5.068,   5.069,   5.07,   5.071,   5.072,   5.073,   5.074,   5.075,   5.076,   5.077,   5.078,   5.079,   5.08,   5.081,   5.082,   5.083,   5.084,   5.085,   5.086,   5.087,   5.088,   5.089,   5.09,   5.091,   5.092,   5.093,   5.094,   5.095,   5.096,   5.097,   5.098,   5.099,   5.1,   5.101,   5.102,   5.103,   5.104,   5.105,   5.106,   5.107,   5.108,   5.109,   5.11,   5.111,   5.112,   5.113,   5.114,   5.115,   5.116,   5.117,   5.118,   5.119,   5.12,   5.121,   5.122,   5.123,   5.124,   5.125,   5.126,   5.127,   5.128]
    //CHECK: [7.001,   7.002,   7.003,   7.004,   7.005,   7.006,   7.007,   7.008,   7.009,   7.01,   7.011,   7.012,   7.013,   7.014,   7.015,   7.016,   7.017,   7.018,   7.019,   7.02,   7.021,   7.022,   7.023,   7.024,   7.025,   7.026,   7.027,   7.028,   7.029,   7.03,   7.031,   7.032,   7.033,   7.034,   7.035,   7.036,   7.037,   7.038,   7.039,   7.04,   7.041,   7.042,   7.043,   7.044,   7.045,   7.046,   7.047,   7.048,   7.049,   7.05,   7.051,   7.052,   7.053,   7.054,   7.055,   7.056,   7.057,   7.058,   7.059,   7.06,   7.061,   7.062,   7.063,   7.064,   7.065,   7.066,   7.067,   7.068,   7.069,   7.07,   7.071,   7.072,   7.073,   7.074,   7.075,   7.076,   7.077,   7.078,   7.079,   7.08,   7.081,   7.082,   7.083,   7.084,   7.085,   7.086,   7.087,   7.088,   7.089,   7.09,   7.091,   7.092,   7.093,   7.094,   7.095,   7.096,   7.097,   7.098,   7.099,   7.1,   7.101,   7.102,   7.103,   7.104,   7.105,   7.106,   7.107,   7.108,   7.109,   7.11,   7.111,   7.112,   7.113,   7.114,   7.115,   7.116,   7.117,   7.118,   7.119,   7.12,   7.121,   7.122,   7.123,   7.124,   7.125,   7.126,   7.127,   7.128]
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func @test(%arg0: memref<4x128xf32>) -> memref<4x128xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c2048_i32 = arith.constant 2048 : i32
    %memref = gpu.alloc  host_shared () : memref<4x128xf32>
    memref.copy %arg0, %memref : memref<4x128xf32> to memref<4x128xf32>
    %memref_0 = gpu.alloc  host_shared () : memref<4x128xf32>
    %alloc = memref.alloc() : memref<4x128xf32, 3>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c256, %c1, %c1) dynamic_shared_memory_size %c2048_i32 args(%memref : memref<4x128xf32>, %memref_0 : memref<4x128xf32>, %alloc : memref<4x128xf32, 3>)
    return %memref_0 : memref<4x128xf32>
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<512 x f32>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<512 x f32>, CrossWorkgroup>, %arg2: !spirv.ptr<!spirv.array<512 x f32>, Workgroup>) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 256, 1, 1>, workgroup_attributions = 0 : i64} {
      %cst128_i64 = spirv.Constant 128 : i64
      %cst256_i64 = spirv.Constant 256 : i64
      %__builtin_var_LocalInvocationId___addr = spirv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_LocalInvocationId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %2 = spirv.IAdd %1, %cst256_i64 : i64
      %3 = spirv.SDiv %1, %cst128_i64 : i64
      %4 = spirv.CL.s_abs %1 : i64
      %5 = spirv.CL.s_abs %cst128_i64 : i64
      %6 = spirv.UMod %4, %5 : i64
      %7 = spirv.IEqual %1, %4 : i64
      %8 = spirv.SNegate %6 : i64
      %9 = spirv.Select %7, %6, %8 : i1, i64
      %10 = spirv.SDiv %2, %cst128_i64 : i64
      %11 = spirv.CL.s_abs %2 : i64
      %12 = spirv.CL.s_abs %cst128_i64 : i64
      %13 = spirv.UMod %11, %12 : i64
      %14 = spirv.IEqual %2, %11 : i64
      %15 = spirv.SNegate %13 : i64
      %16 = spirv.Select %14, %13, %15 : i1, i64
      %cst0_i64 = spirv.Constant 0 : i64
      %cst128_i64_0 = spirv.Constant 128 : i64
      %17 = spirv.IMul %cst128_i64_0, %3 : i64
      %18 = spirv.IAdd %cst0_i64, %17 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %19 = spirv.IMul %cst1_i64, %9 : i64
      %20 = spirv.IAdd %18, %19 : i64
      %21 = spirv.AccessChain %arg0[%20] : !spirv.ptr<!spirv.array<512 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      %22 = spirv.Load "CrossWorkgroup" %21 : f32
      %cst0_i64_1 = spirv.Constant 0 : i64
      %cst128_i64_2 = spirv.Constant 128 : i64
      %23 = spirv.IMul %cst128_i64_2, %10 : i64
      %24 = spirv.IAdd %cst0_i64_1, %23 : i64
      %cst1_i64_3 = spirv.Constant 1 : i64
      %25 = spirv.IMul %cst1_i64_3, %16 : i64
      %26 = spirv.IAdd %24, %25 : i64
      %27 = spirv.AccessChain %arg0[%26] : !spirv.ptr<!spirv.array<512 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      %28 = spirv.Load "CrossWorkgroup" %27 : f32
      %cst0_i64_4 = spirv.Constant 0 : i64
      %cst128_i64_5 = spirv.Constant 128 : i64
      %29 = spirv.IMul %cst128_i64_5, %3 : i64
      %30 = spirv.IAdd %cst0_i64_4, %29 : i64
      %cst1_i64_6 = spirv.Constant 1 : i64
      %31 = spirv.IMul %cst1_i64_6, %9 : i64
      %32 = spirv.IAdd %30, %31 : i64
      %33 = spirv.AccessChain %arg2[%32] : !spirv.ptr<!spirv.array<512 x f32>, Workgroup>, i64 -> !spirv.ptr<f32, Workgroup>
      spirv.Store "Workgroup" %33, %22 : f32
      %cst0_i64_7 = spirv.Constant 0 : i64
      %cst128_i64_8 = spirv.Constant 128 : i64
      %34 = spirv.IMul %cst128_i64_8, %10 : i64
      %35 = spirv.IAdd %cst0_i64_7, %34 : i64
      %cst1_i64_9 = spirv.Constant 1 : i64
      %36 = spirv.IMul %cst1_i64_9, %16 : i64
      %37 = spirv.IAdd %35, %36 : i64
      %38 = spirv.AccessChain %arg2[%37] : !spirv.ptr<!spirv.array<512 x f32>, Workgroup>, i64 -> !spirv.ptr<f32, Workgroup>
      spirv.Store "Workgroup" %38, %28 : f32
      spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
      %cst0_i64_10 = spirv.Constant 0 : i64
      %cst128_i64_11 = spirv.Constant 128 : i64
      %39 = spirv.IMul %cst128_i64_11, %3 : i64
      %40 = spirv.IAdd %cst0_i64_10, %39 : i64
      %cst1_i64_12 = spirv.Constant 1 : i64
      %41 = spirv.IMul %cst1_i64_12, %9 : i64
      %42 = spirv.IAdd %40, %41 : i64
      %43 = spirv.AccessChain %arg2[%42] : !spirv.ptr<!spirv.array<512 x f32>, Workgroup>, i64 -> !spirv.ptr<f32, Workgroup>
      %44 = spirv.Load "Workgroup" %43 : f32
      %cst0_i64_13 = spirv.Constant 0 : i64
      %cst128_i64_14 = spirv.Constant 128 : i64
      %45 = spirv.IMul %cst128_i64_14, %10 : i64
      %46 = spirv.IAdd %cst0_i64_13, %45 : i64
      %cst1_i64_15 = spirv.Constant 1 : i64
      %47 = spirv.IMul %cst1_i64_15, %16 : i64
      %48 = spirv.IAdd %46, %47 : i64
      %49 = spirv.AccessChain %arg2[%48] : !spirv.ptr<!spirv.array<512 x f32>, Workgroup>, i64 -> !spirv.ptr<f32, Workgroup>
      %50 = spirv.Load "Workgroup" %49 : f32
      %cst0_i64_16 = spirv.Constant 0 : i64
      %cst128_i64_17 = spirv.Constant 128 : i64
      %51 = spirv.IMul %cst128_i64_17, %3 : i64
      %52 = spirv.IAdd %cst0_i64_16, %51 : i64
      %cst1_i64_18 = spirv.Constant 1 : i64
      %53 = spirv.IMul %cst1_i64_18, %9 : i64
      %54 = spirv.IAdd %52, %53 : i64
      %55 = spirv.AccessChain %arg1[%54] : !spirv.ptr<!spirv.array<512 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %55, %44 : f32
      %cst0_i64_19 = spirv.Constant 0 : i64
      %cst128_i64_20 = spirv.Constant 128 : i64
      %56 = spirv.IMul %cst128_i64_20, %10 : i64
      %57 = spirv.IAdd %cst0_i64_19, %56 : i64
      %cst1_i64_21 = spirv.Constant 1 : i64
      %58 = spirv.IMul %cst1_i64_21, %16 : i64
      %59 = spirv.IAdd %57, %58 : i64
      %60 = spirv.AccessChain %arg1[%59] : !spirv.ptr<!spirv.array<512 x f32>, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %60, %50 : f32
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_LocalInvocationId__
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<4x128xf32>, %arg1: memref<4x128xf32>, %arg2: memref<4x128xf32, 3>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 256, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.thread_id  x
      %1 = arith.addi %0, %c256 : index
      %2 = arith.divsi %0, %c128 : index
      %3 = arith.remsi %0, %c128 : index
      %4 = arith.divsi %1, %c128 : index
      %5 = arith.remsi %1, %c128 : index
      %6 = memref.load %arg0[%2, %3] : memref<4x128xf32>
      %7 = memref.load %arg0[%4, %5] : memref<4x128xf32>
      memref.store %6, %arg2[%2, %3] : memref<4x128xf32, 3>
      memref.store %7, %arg2[%4, %5] : memref<4x128xf32, 3>
      gpu.barrier
      %8 = memref.load %arg2[%2, %3] : memref<4x128xf32, 3>
      %9 = memref.load %arg2[%4, %5] : memref<4x128xf32, 3>
      memref.store %8, %arg1[%2, %3] : memref<4x128xf32>
      memref.store %9, %arg1[%4, %5] : memref<4x128xf32>
      gpu.return
    }
  }
}
