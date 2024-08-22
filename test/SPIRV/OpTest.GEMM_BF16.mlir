// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=opencl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%opencl_runtime --filecheck

module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_3x3xbf16_result : memref<3x3xbf16> = dense<1.000000e+00>
  memref.global "private" constant @__constant_3x3xbf16_0: memref<3x3xbf16> = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [3.000000e+00, 4.000000e+00, 5.000000e-01], [3.000000e+00, 3.000000e+00, 3.000000e+00]]>
  memref.global "private" constant @__constant_3x3xbf16 : memref<3x3xbf16> = dense<[[5.000000e-01, 2.000000e-01, 4.000000e+00], [1.000000e+00, 1.000000e+00, 2.000000e+00], [3.000000e+00, 3.000000e+00, 3.000000e-01]]>
   func.func @main() {
    %0 = memref.get_global @__constant_3x3xbf16 : memref<3x3xbf16>
    %1 = memref.get_global @__constant_3x3xbf16_0 : memref<3x3xbf16>
    %2 = memref.get_global @__constant_3x3xbf16_result : memref<3x3xbf16>
    %3 = call @test(%0, %1, %2) : (memref<3x3xbf16>, memref<3x3xbf16>, memref<3x3xbf16>) -> memref<3x3xbf16>
    %cast = memref.cast %3 : memref<3x3xbf16> to memref<*xbf16>
    call @printMemrefBF16(%cast) : (memref<*xbf16>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT: [14.125,   14.8125,   14.625]
    // CHECK-NEXT: [11,   13,   10.5]
    // CHECK-NEXT: [13.875,   19.875,   12.375]
    return
  }
  func.func private @printMemrefBF16(memref<*xbf16>)
  func.func @test(%arg0: memref<3x3xbf16>, %arg1: memref<3x3xbf16>, %arg2: memref<3x3xbf16>) -> memref<3x3xbf16> {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index

    // Strategy 1: Do bf16 to i16 conversion just before imex-convert-gpu-to-spirv pass
    // Only convert the kernel parameters
    // Way to do it:
    // 1. Create i8 memrefs for all kernel args that uses bf16,
    // 2. Create a view of the args as bf16,
    // 3. Copy the original args to that view using memref.copy
    // 4. Create a view of the args as i16
    // 5. Pass the newly allocated i8/i16 args to the kernel
    // 6. Do necessary conversion inside the spirv module

    %memref_kernel_arg0_i8 = gpu.alloc  host_shared () : memref<18xi8>
    %memref_kernel_arg1_i8 = gpu.alloc  host_shared () : memref<18xi8>

    %memref_kernel_arg0_bf16 = memref.view %memref_kernel_arg0_i8[%c0][] : memref<18xi8> to memref<3x3xbf16>
    %memref_kernel_arg1_bf16 = memref.view %memref_kernel_arg1_i8[%c0][] : memref<18xi8> to memref<3x3xbf16>


    memref.copy %arg0, %memref_kernel_arg0_bf16 : memref<3x3xbf16> to memref<3x3xbf16>
    memref.copy %arg1, %memref_kernel_arg1_bf16 : memref<3x3xbf16> to memref<3x3xbf16>

    %memref_kernel_arg0_i16 = memref.view %memref_kernel_arg0_i8[%c0][] : memref<18xi8> to memref<3x3xi16>
    %memref_kernel_arg1_i16 = memref.view %memref_kernel_arg1_i8[%c0][] : memref<18xi8> to memref<3x3xi16>


    %memref_kernel_result_i8 = gpu.alloc  host_shared () : memref<18xi8>
    %memref_kernel_result_bf16 = memref.view %memref_kernel_result_i8[%c0][] : memref<18xi8> to memref<3x3xbf16>
    %memref_kernel_result_i16 = memref.view %memref_kernel_result_i8[%c0][] : memref<18xi8> to memref<3x3xi16>

    memref.copy %arg2, %memref_kernel_result_bf16 : memref<3x3xbf16> to memref<3x3xbf16>
    gpu.launch_func  @gemm_kernel::@test_kernel blocks in (%c3, %c3, %c1) threads in (%c1, %c1, %c1) args(%memref_kernel_arg0_i16 : memref<3x3xi16>, %memref_kernel_arg1_i16 : memref<3x3xi16>, %memref_kernel_result_i16 : memref<3x3xi16>, %c0 : index, %c3 : index, %c1 : index)
    gpu.dealloc  %memref_kernel_arg0_i16 : memref<3x3xi16>
    gpu.dealloc  %memref_kernel_arg1_i16 : memref<3x3xi16>
    return %memref_kernel_result_bf16 : memref<3x3xbf16>
  }

 spirv.module @__spv__gemm_kernel Physical64 OpenCL  requires #spirv.vce<v1.0, [Int64, Int16, Kernel, Addresses], []>  attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_AMD_shader_ballot]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<9 x i16>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<9 x i16>, CrossWorkgroup>, %arg2: !spirv.ptr<!spirv.array<9 x i16>, CrossWorkgroup>, %arg3: i64, %arg4: i64, %arg5: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 3, 3, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>, workgroup_attributions = 0 : i64} {
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi64>
      spirv.mlir.loop {
        spirv.Branch ^bb1(%arg3 : i64)
      ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb2
        %5 = spirv.SLessThan %4, %arg4 : i64
        spirv.BranchConditional %5, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %cst0_i64 = spirv.Constant 0 : i64
        %cst3_i64 = spirv.Constant 3 : i64
        %6 = spirv.IMul %cst3_i64, %1 : i64
        %7 = spirv.IAdd %cst0_i64, %6 : i64
        %cst1_i64 = spirv.Constant 1 : i64
        %8 = spirv.IMul %cst1_i64, %4 : i64
        %9 = spirv.IAdd %7, %8 : i64
        %10 = spirv.AccessChain %arg0[%9] : !spirv.ptr<!spirv.array<9 x i16>, CrossWorkgroup>, i64
        %11 = spirv.Load "CrossWorkgroup" %10 : i16
        %cst0_i64_1 = spirv.Constant 0 : i64
        %cst3_i64_2 = spirv.Constant 3 : i64
        %12 = spirv.IMul %cst3_i64_2, %4 : i64
        %13 = spirv.IAdd %cst0_i64_1, %12 : i64
        %cst1_i64_3 = spirv.Constant 1 : i64
        %14 = spirv.IMul %cst1_i64_3, %3 : i64
        %15 = spirv.IAdd %13, %14 : i64
        %16 = spirv.AccessChain %arg1[%15] : !spirv.ptr<!spirv.array<9 x i16>, CrossWorkgroup>, i64
        %17 = spirv.Load "CrossWorkgroup" %16 : i16
        %cst0_i64_4 = spirv.Constant 0 : i64
        %cst3_i64_5 = spirv.Constant 3 : i64
        %18 = spirv.IMul %cst3_i64_5, %1 : i64
        %19 = spirv.IAdd %cst0_i64_4, %18 : i64
        %cst1_i64_6 = spirv.Constant 1 : i64
        %20 = spirv.IMul %cst1_i64_6, %3 : i64
        %21 = spirv.IAdd %19, %20 : i64
        %22 = spirv.AccessChain %arg2[%21] : !spirv.ptr<!spirv.array<9 x i16>, CrossWorkgroup>, i64
        %23 = spirv.Load "CrossWorkgroup" %22 : i16
        %231 = spirv.INTEL.ConvertBF16ToF %23 : i16 to f32

        %111 = spirv.INTEL.ConvertBF16ToF %11 : i16 to f32
        %171 = spirv.INTEL.ConvertBF16ToF %17 : i16 to f32

        %24 = spirv.FMul %111, %171 : f32
        %25 = spirv.FAdd %231, %24 : f32

        %251 = spirv.INTEL.ConvertFToBF16 %25 : f32 to i16

        %cst0_i64_7 = spirv.Constant 0 : i64
        %cst3_i64_8 = spirv.Constant 3 : i64
        %26 = spirv.IMul %cst3_i64_8, %1 : i64
        %27 = spirv.IAdd %cst0_i64_7, %26 : i64
        %cst1_i64_9 = spirv.Constant 1 : i64
        %28 = spirv.IMul %cst1_i64_9, %3 : i64
        %29 = spirv.IAdd %27, %28 : i64
        %30 = spirv.AccessChain %arg2[%29] : !spirv.ptr<!spirv.array<9 x i16>, CrossWorkgroup>, i64
        spirv.Store "CrossWorkgroup" %30, %251 : i16
        %31 = spirv.IAdd %4, %arg5 : i64
        spirv.Branch ^bb1(%31 : i64)
      ^bb3:  // pred: ^bb1
        spirv.mlir.merge
      }
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @gemm_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<3x3xi16>, %arg1: memref<3x3xi16>, %arg2: memref<3x3xi16>, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 3, 3, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      scf.for %arg6 = %arg3 to %arg4 step %arg5 {
        %c0 = arith.constant 0 : index

        %2 = memref.load %arg0[%0, %arg6] : memref<3x3xi16>
        %3 = memref.load %arg1[%arg6, %1] : memref<3x3xi16>

        %222 = arith.bitcast %2 : i16 to bf16
        %333 = arith.bitcast %3 : i16 to bf16

        %22 = arith.extf %222 : bf16 to f32
        %33 = arith.extf %333 : bf16 to f32

        %4 = memref.load %arg2[%0, %1] : memref<3x3xi16>

        %444 = arith.bitcast %4 : i16 to bf16
        %432 = arith.extf %444 : bf16 to f32

        %5 = arith.mulf %22, %33 : f32
        %6 = arith.addf %432, %5 : f32

        %666 = arith.truncf %6: f32 to bf16
        %616 = arith.bitcast %666 : bf16 to i16

        memref.store %616, %arg2[%0, %1] : memref<3x3xi16>
      }
      gpu.return
    }
  }
}
