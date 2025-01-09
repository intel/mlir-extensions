// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

module @broadcast_non_numpy attributes {gpu.container_module} {
   memref.global "private" constant @__constant_3xbf16 : memref<3xbf16> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]>
  func.func @main() {
    %0 = memref.get_global @__constant_3xbf16 : memref<3xbf16>
    %1 = call @test(%0) : (memref<3xbf16>) -> memref<3x4xbf16>
    %cast = memref.cast %1 : memref<3x4xbf16> to memref<*xbf16>
    call @printMemrefBF16(%cast) : (memref<*xbf16>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT: [1, 1, 1, 1]
    // CHECK-NEXT: [2, 2, 2, 2]
    // CHECK-NEXT: [3, 3, 3, 3]
    return
  }
  func.func private @printMemrefBF16(memref<*xbf16>)

  func.func @test(%arg0: memref<3xbf16>) -> memref<3x4xbf16> attributes {llvm.emit_c_interface} {
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index

    // 1. Create i8 memrefs for all kernel args that uses bf16,
    // 2. Create a view of the args as bf16,
    // 3. Copy the original args to that view using memref.copy
    // 4. Create a view of the args as i16
    %memref_kernel_arg0_i8 = gpu.alloc  host_shared () : memref<6xi8>
    %memref_kernel_arg0_bf16 = memref.view %memref_kernel_arg0_i8[%c0][] : memref<6xi8> to memref<3xbf16>
    memref.copy %arg0, %memref_kernel_arg0_bf16 : memref<3xbf16> to memref<3xbf16>
    %memref_kernel_arg0_i16 = memref.view %memref_kernel_arg0_i8[%c0][] : memref<6xi8> to memref<3xi16>

    %memref_kernel_result_i8 = gpu.alloc  host_shared () : memref<24xi8>
    %memref_kernel_result_bf16 = memref.view %memref_kernel_result_i8[%c0][] : memref<24xi8> to memref<3x4xbf16>
    %memref_kernel_result_i16 = memref.view %memref_kernel_result_i8[%c0][] : memref<24xi8> to memref<3x4xi16>

    // 5. Pass the newly allocated i16 args to the kernel
    gpu.launch_func  @broadcast_kernel::@test blocks in (%c3, %c4, %c1) threads in (%c1, %c1, %c1) args(%memref_kernel_arg0_i16 : memref<3xi16>, %memref_kernel_result_i16 : memref<3x4xi16>)
    gpu.dealloc  %memref_kernel_arg0_i8 : memref<6xi8>
    return %memref_kernel_result_bf16 : memref<3x4xbf16>
  }

spirv.module @__spv__broadcast_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64], []>, api=OpenCL, #spirv.resource_limits<>>} {
  spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
  spirv.func @test(%arg0: !spirv.ptr<!spirv.array<3 x i16>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<12 x i16>, CrossWorkgroup>) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 3, 4, 1>, workgroup_attributions = 0 : i64} {
    %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
    %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
    %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
    %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
    %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
    %3 = spirv.CompositeExtract %2[1 : i32] : vector<3xi64>
    %cst0_i64 = spirv.Constant 0 : i64
    %cst1_i64 = spirv.Constant 1 : i64
    %4 = spirv.IMul %cst1_i64, %1 : i64
    %5 = spirv.IAdd %cst0_i64, %4 : i64
    %6 = spirv.AccessChain %arg0[%5] : !spirv.ptr<!spirv.array<3 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
    %7 = spirv.Load "CrossWorkgroup" %6 : i16
    %cst0_i64_1 = spirv.Constant 0 : i64
    %cst4_i64 = spirv.Constant 4 : i64
    %8 = spirv.IMul %cst4_i64, %1 : i64
    %9 = spirv.IAdd %cst0_i64_1, %8 : i64
    %cst1_i64_2 = spirv.Constant 1 : i64
    %10 = spirv.IMul %cst1_i64_2, %3 : i64
    %11 = spirv.IAdd %9, %10 : i64
    %12 = spirv.AccessChain %arg1[%11] : !spirv.ptr<!spirv.array<12 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
    spirv.Store "CrossWorkgroup" %12, %7 : i16
    spirv.Return
  }
  spirv.EntryPoint "Kernel" @test, @__builtin_var_WorkgroupId__
}

  gpu.module @broadcast_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test(%arg0: memref<3xi16>, %arg1: memref<3x4xi16>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 3, 4, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = memref.load %arg0[%0] : memref<3xi16>
      memref.store %2, %arg1[%0, %1] : memref<3x4xi16>
      gpu.return
    }
  }

}
