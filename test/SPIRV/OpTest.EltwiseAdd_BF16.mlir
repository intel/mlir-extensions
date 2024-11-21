// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @eltwise_add attributes {gpu.container_module} {
  memref.global "private" constant @__constant_10x20xbf16 : memref<10x20xbf16> = dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
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
  memref.global "private" constant @__constant_10x20xbf16_1 : memref<10x20xbf16> = dense<0.5>

func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_10x20xbf16 : memref<10x20xbf16>
    %1 = memref.get_global @__constant_10x20xbf16_1 : memref<10x20xbf16>
    %2 = call @test(%0, %1) : (memref<10x20xbf16>, memref<10x20xbf16>) -> memref<10x20xbf16>
    %cast = memref.cast %2 : memref<10x20xbf16> to memref<*xbf16>
    call @printMemrefBF16(%cast) : (memref<*xbf16>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-COUNT-10: [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5]
    return
  }
  func.func private @printMemrefBF16(memref<*xbf16>) attributes {llvm.emit_c_interface}

  func.func @test(%arg0: memref<10x20xbf16>, %arg1: memref<10x20xbf16>) -> memref<10x20xbf16> attributes {llvm.emit_c_interface} {
    %c20 = arith.constant 20 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %memref_arg0_bf16 = gpu.alloc  host_shared () : memref<10x20xbf16>
    %memref_arg1_bf16 = gpu.alloc  host_shared () : memref<10x20xbf16>

    %memref_result_i8 = gpu.alloc  host_shared () : memref<400xi8>

    // Copy the args to gpu memory
    memref.copy %arg0, %memref_arg0_bf16 : memref<10x20xbf16> to memref<10x20xbf16>
    memref.copy %arg1, %memref_arg1_bf16 : memref<10x20xbf16> to memref<10x20xbf16>

    // Strategy 1: Do bf16 to i16 conversion just before imex-convert-gpu-to-spirv pass
    // Only convert the kernel parameters
    // Way to do it:
    // 1. Create i8 memrefs for all kernel args that uses bf16,
    // 2. Create a view of the args as bf16,
    // 3. Copy the original args to that view using memref.copy
    // 4. Create a view of the args as i16
    // 5. Pass the newly allocated i8/i16 args to the kernel
    // 6. Do necessary conversion inside the spirv module
    %memref_kernel_arg0_i8 = gpu.alloc  host_shared () : memref<400xi8>
    %memref_kernel_arg1_i8 = gpu.alloc  host_shared () : memref<400xi8>

    %memref_kernel_arg0_bf16 = memref.view %memref_kernel_arg0_i8[%c0][] : memref<400xi8> to memref<10x20xbf16>
    %memref_kernel_arg1_bf16 = memref.view %memref_kernel_arg1_i8[%c0][] : memref<400xi8> to memref<10x20xbf16>

    // Copy the original values of the kernel args
    memref.copy %memref_arg0_bf16, %memref_kernel_arg0_bf16 : memref<10x20xbf16> to memref<10x20xbf16>
    memref.copy %memref_arg1_bf16, %memref_kernel_arg1_bf16 : memref<10x20xbf16> to memref<10x20xbf16>

    %memref_kernel_arg0_i16 = memref.view %memref_kernel_arg0_i8[%c0][] : memref<400xi8> to memref<10x20xi16>
    %memref_kernel_arg1_i16 = memref.view %memref_kernel_arg1_i8[%c0][] : memref<400xi8> to memref<10x20xi16>


    %memref_kernel_result_bf16 = memref.view %memref_result_i8[%c0][] : memref<400xi8> to memref<10x20xbf16>
    %memref_kernel_result_i16 = memref.view %memref_result_i8[%c0][] : memref<400xi8> to memref<10x20xi16>

    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c10, %c20, %c1) threads in (%c1, %c1, %c1) args(%memref_kernel_arg0_i16 : memref<10x20xi16>, %memref_kernel_arg1_i16 : memref<10x20xi16>, %memref_kernel_result_i16 : memref<10x20xi16>)

    gpu.dealloc  %memref_kernel_arg0_i16 : memref<10x20xi16>
    gpu.dealloc  %memref_kernel_arg1_i16 : memref<10x20xi16>
    return %memref_kernel_result_bf16 : memref<10x20xbf16>
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Int16, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_AMD_shader_ballot]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<200 x i16>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<200 x i16>, CrossWorkgroup>, %arg2: !spirv.ptr<!spirv.array<200 x i16>, CrossWorkgroup>) "None" attributes {workgroup_attributions = 0 : i64} {
      %cst20_i64 = spirv.Constant 20 : i64
      %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
      %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
      %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
      %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
      %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi64>
      %4 = spirv.IMul %1, %cst20_i64 : i64
      %5 = spirv.IAdd %4, %3 : i64
      %6 = spirv.AccessChain %arg0[%5] : !spirv.ptr<!spirv.array<200 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
      %7 = spirv.Load "CrossWorkgroup" %6 : i16
      %8 = spirv.IMul %1, %cst20_i64 : i64
      %9 = spirv.IAdd %8, %3 : i64
      %10 = spirv.AccessChain %arg1[%9] : !spirv.ptr<!spirv.array<200 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
      %11 = spirv.Load "CrossWorkgroup" %10 : i16
      // %12 = spirv.IAdd %7, %11 : i16
      // *************************************** //
      // Convert the operands to f32 from bf16
      %f32_7 = spirv.INTEL.ConvertBF16ToF %7 : i16 to f32
      %f32_11 = spirv.INTEL.ConvertBF16ToF %11 : i16 to f32
      // Do an FADD on the operands
      %f32_12 = spirv.FAdd %f32_7, %f32_11 : f32
      // Convert the result back to i16 for storing
      %12 = spirv.INTEL.ConvertFToBF16 %f32_12 : f32 to i16
      // *************************************** //

      %13 = spirv.IMul %1, %cst20_i64 : i64
      %14 = spirv.IAdd %13, %3 : i64
      %15 = spirv.AccessChain %arg2[%14] : !spirv.ptr<!spirv.array<200 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %15, %12 : i16
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_AMD_shader_ballot]>, api=OpenCL, #spirv.resource_limits<>>} {
    // Can't change the inputs here to bf16, since, the signature of the kernel has to match the spirv kernel
    gpu.func @test_kernel(%arg0: memref<10x20xi16>, %arg1: memref<10x20xi16>, %arg2: memref<10x20xi16>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %c0 = arith.constant 0 : index

      %2 = memref.load %arg0[%0, %1] : memref<10x20xi16>
      %3 = memref.load %arg1[%0, %1] : memref<10x20xi16>

      %bf16_2 = arith.bitcast %2 : i16 to bf16
      %bf16_3 = arith.bitcast %3 : i16 to bf16

      %f32_2 = arith.extf %bf16_2 : bf16 to f32
      %f32_3 = arith.extf %bf16_3 : bf16 to f32
      %4 = arith.addf %f32_2, %f32_3 : f32

      %5 = arith.truncf %4 : f32 to bf16
      %6 = arith.bitcast %5 : bf16 to i16
      memref.store %6, %arg2[%0, %1] : memref<10x20xi16>
      gpu.return
    }
  }

}
