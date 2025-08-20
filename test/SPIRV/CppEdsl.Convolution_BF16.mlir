// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%mlir_runner_utils,%irunner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%mlir_runner_utils,%irunner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

module @convolution attributes {gpu.container_module} {
  memref.global "private" constant @__constant_3x3x64x64xbf16 : memref<3x3x64x64xbf16> = dense<5.000000e-01>
  memref.global "private" constant @__constant_1x56x56x64xbf16 : memref<1x56x56x64xbf16> = dense<1.000000e+00>
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_1x56x56x64xbf16 : memref<1x56x56x64xbf16>
    %1 = memref.get_global @__constant_3x3x64x64xbf16 : memref<3x3x64x64xbf16>
    %2 = call @test(%0, %1) : (memref<1x56x56x64xbf16>, memref<3x3x64x64xbf16>) -> memref<1x56x56x64xbf16>
    %cast = memref.cast %2 : memref<1x56x56x64xbf16> to memref<*xbf16>
    call @printMemrefBF16(%cast) : (memref<*xbf16>) -> ()
     //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT: [128,     128,     128,     128,     128,     128,     128,     128,     128,     128,     128
    return
  }
  func.func private @printMemrefBF16(memref<*xbf16>)

  func.func @test(%arg0: memref<1x56x56x64xbf16>, %arg1: memref<3x3x64x64xbf16>) -> memref<1x56x56x64xbf16> attributes {llvm.emit_c_interface} {
    %c56 = arith.constant 56 : index
    %c1 = arith.constant 1 : index
    %c58 = arith.constant 58 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c3 = arith.constant 3 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index

    %memref_kernel_arg0_i8 = gpu.alloc  host_shared () : memref<401408xi8>
    %memref_kernel_arg1_i8 = gpu.alloc  host_shared () : memref<73728xi8>

    %memref_kernel_arg0_bf16 = memref.view %memref_kernel_arg0_i8[%c0][] : memref<401408xi8> to memref<1x56x56x64xbf16>
    %memref_kernel_arg1_bf16 = memref.view %memref_kernel_arg1_i8[%c0][] : memref<73728xi8> to memref<3x3x64x64xbf16>

    %memref_kernel_arg0_i16 = memref.view %memref_kernel_arg0_i8[%c0][] : memref<401408xi8> to memref<1x56x56x64xi16>
    %memref_kernel_arg1_i16 = memref.view %memref_kernel_arg1_i8[%c0][] : memref<73728xi8> to memref<3x3x64x64xi16>

    memref.copy %arg0, %memref_kernel_arg0_bf16 : memref<1x56x56x64xbf16> to memref<1x56x56x64xbf16>
    memref.copy %arg1, %memref_kernel_arg1_bf16 : memref<3x3x64x64xbf16> to memref<3x3x64x64xbf16>

    %memref_1_i8 = gpu.alloc  host_shared () : memref<401408xi8>
    %memref_1_bf16 = memref.view %memref_1_i8[%c0][] : memref<401408xi8> to memref<1x56x56x64xbf16>
    %memref_1_i16 = memref.view %memref_1_i8[%c0][] : memref<401408xi8> to memref<1x56x56x64xi16>

    gpu.launch_func  @copy_arg0_kernel::@test_kernel blocks in (%c1, %c56, %c56) threads in (%c1, %c1, %c1) args(%memref_kernel_arg0_i16 : memref<1x56x56x64xi16>, %c0 : index, %memref_1_i16 : memref<1x56x56x64xi16>, %c64 : index, %c1 : index)

    %memref_2_i8 = gpu.alloc  host_shared () : memref<430592xi8>
    %memref_2_bf16 = memref.view %memref_2_i8[%c0][] : memref<430592xi8> to memref<1x58x58x64xbf16>
    %memref_2_i16 = memref.view %memref_2_i8[%c0][] : memref<430592xi8> to memref<1x58x58x64xi16>

     %cst_i16 = arith.constant 0 : i16
    gpu.launch_func  @init_kernel_0::@test_kernel blocks in (%c1, %c58, %c58) threads in (%c1, %c1, %c1) args(%cst_i16 : i16, %memref_2_i16 : memref<1x58x58x64xi16>, %c0 : index, %c64 : index, %c1 : index)


    %memref_3_i8 = gpu.alloc  host_shared () : memref<430592xi8>
    %memref_3_bf16 = memref.view %memref_3_i8[%c0][] : memref<430592xi8> to memref<1x58x58x64xbf16>
    %memref_3_i16 = memref.view %memref_3_i8[%c0][] : memref<430592xi8> to memref<1x58x58x64xi16>

    memref.copy %memref_2_bf16, %memref_3_bf16 : memref<1x58x58x64xbf16> to memref<1x58x58x64xbf16>
    %subview = memref.subview %memref_3_bf16[0, 1, 1, 0] [1, 56, 56, 64] [1, 1, 1, 1] : memref<1x58x58x64xbf16> to memref<1x56x56x64xbf16, strided<[215296, 3712, 64, 1], offset: 3776>>
    memref.copy %memref_1_bf16, %subview : memref<1x56x56x64xbf16> to memref<1x56x56x64xbf16, strided<[215296, 3712, 64, 1], offset: 3776>>

    %memref_4_i8 = gpu.alloc  host_shared () : memref<401408xi8>
    %memref_4_bf16 = memref.view %memref_4_i8[%c0][] : memref<401408xi8> to memref<1x56x56x64xbf16>
    %memref_4_i16 = memref.view %memref_4_i8[%c0][] : memref<401408xi8> to memref<1x56x56x64xi16>
    gpu.launch_func  @init_kernel_1::@test_kernel blocks in (%c1, %c56, %c56) threads in (%c1, %c1, %c1) args(%cst_i16 : i16, %memref_4_i16 : memref<1x56x56x64xi16>, %c0 : index, %c64 : index, %c1 : index)

    %memref_5_i8 = gpu.alloc  host_shared () : memref<401408xi8>
    %memref_5_bf16 = memref.view %memref_5_i8[%c0][] : memref<401408xi8> to memref<1x56x56x64xbf16>
    %memref_5_i16 = memref.view %memref_5_i8[%c0][] : memref<401408xi8> to memref<1x56x56x64xi16>
    memref.copy %memref_4_bf16, %memref_5_bf16 : memref<1x56x56x64xbf16> to memref<1x56x56x64xbf16>
    gpu.launch_func  @conv_kernel::@test_kernel blocks in (%c1, %c56, %c56) threads in (%c1, %c1, %c1) args(%memref_3_i16 : memref<1x58x58x64xi16>, %c0 : index, %memref_kernel_arg1_i16 : memref<3x3x64x64xi16>, %memref_5_i16 : memref<1x56x56x64xi16>, %c64 : index, %c1 : index, %c3 : index)

    gpu.dealloc  %memref_1_i8 : memref<401408xi8>
    gpu.dealloc  %memref_2_i8 : memref<430592xi8>
    gpu.dealloc  %memref_4_i8 : memref<401408xi8>
    gpu.dealloc  %memref_kernel_arg0_i8 : memref<401408xi8>
    gpu.dealloc  %memref_kernel_arg1_i8 : memref<73728xi8>

    return %memref_5_bf16 : memref<1x56x56x64xbf16>

  }

spirv.module @__spv__init_kernel_0 Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
  spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
  spirv.func @test_kernel(%arg0: i16, %arg1: !spirv.ptr<!spirv.array<215296 x i16>, CrossWorkgroup>, %arg2: i64, %arg3: i64, %arg4: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 58, 58>, workgroup_attributions = 0 : i64} {
    %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
    %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
    %1 = spirv.CompositeExtract %0[1 : i32] : vector<3xi64>
    %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
    %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
    %3 = spirv.CompositeExtract %2[2 : i32] : vector<3xi64>
    spirv.mlir.loop {
      spirv.Branch ^bb1(%arg2 : i64)
    ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb2
      %5 = spirv.SLessThan %4, %arg3 : i64
      spirv.BranchConditional %5, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %cst0_i64 = spirv.Constant 0 : i64
      %cst215296_i64 = spirv.Constant 215296 : i64
      %6 = spirv.IMul %cst215296_i64, %arg2 : i64
      %7 = spirv.IAdd %cst0_i64, %6 : i64
      %cst3712_i64 = spirv.Constant 3712 : i64
      %8 = spirv.IMul %cst3712_i64, %1 : i64
      %9 = spirv.IAdd %7, %8 : i64
      %cst64_i64 = spirv.Constant 64 : i64
      %10 = spirv.IMul %cst64_i64, %3 : i64
      %11 = spirv.IAdd %9, %10 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %12 = spirv.IMul %cst1_i64, %4 : i64
      %13 = spirv.IAdd %11, %12 : i64
      %14 = spirv.AccessChain %arg1[%13] : !spirv.ptr<!spirv.array<215296 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %14, %arg0 : i16
      %15 = spirv.IAdd %4, %arg4 : i64
      spirv.Branch ^bb1(%15 : i64)
    ^bb3:  // pred: ^bb1
      spirv.mlir.merge
    }
    spirv.Return
  }
  spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
}

spirv.module @__spv__init_kernel_1 Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
  spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
  spirv.func @test_kernel(%arg0: i16, %arg1: !spirv.ptr<!spirv.array<200704 x i16>, CrossWorkgroup>, %arg2: i64, %arg3: i64, %arg4: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 56, 56>, workgroup_attributions = 0 : i64} {
    %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
    %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
    %1 = spirv.CompositeExtract %0[1 : i32] : vector<3xi64>
    %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
    %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
    %3 = spirv.CompositeExtract %2[2 : i32] : vector<3xi64>
    spirv.mlir.loop {
      spirv.Branch ^bb1(%arg2 : i64)
    ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb2
      %5 = spirv.SLessThan %4, %arg3 : i64
      spirv.BranchConditional %5, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %cst0_i64 = spirv.Constant 0 : i64
      %cst200704_i64 = spirv.Constant 200704 : i64
      %6 = spirv.IMul %cst200704_i64, %arg2 : i64
      %7 = spirv.IAdd %cst0_i64, %6 : i64
      %cst3584_i64 = spirv.Constant 3584 : i64
      %8 = spirv.IMul %cst3584_i64, %1 : i64
      %9 = spirv.IAdd %7, %8 : i64
      %cst64_i64 = spirv.Constant 64 : i64
      %10 = spirv.IMul %cst64_i64, %3 : i64
      %11 = spirv.IAdd %9, %10 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %12 = spirv.IMul %cst1_i64, %4 : i64
      %13 = spirv.IAdd %11, %12 : i64
      %14 = spirv.AccessChain %arg1[%13] : !spirv.ptr<!spirv.array<200704 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %14, %arg0 : i16
      %15 = spirv.IAdd %4, %arg4 : i64
      spirv.Branch ^bb1(%15 : i64)
    ^bb3:  // pred: ^bb1
      spirv.mlir.merge
    }
    spirv.Return
  }
  spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
}

spirv.module @__spv__copy_arg0_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
  spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
  spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<200704 x i16>, CrossWorkgroup>, %arg1: i64, %arg2: !spirv.ptr<!spirv.array<200704 x i16>, CrossWorkgroup>, %arg3: i64, %arg4: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 56, 56>, workgroup_attributions = 0 : i64} {
    %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
    %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
    %1 = spirv.CompositeExtract %0[1 : i32] : vector<3xi64>
    %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
    %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
    %3 = spirv.CompositeExtract %2[2 : i32] : vector<3xi64>
    spirv.mlir.loop {
      spirv.Branch ^bb1(%arg1 : i64)
    ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb2
      %5 = spirv.SLessThan %4, %arg3 : i64
      spirv.BranchConditional %5, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %cst0_i64 = spirv.Constant 0 : i64
      %cst200704_i64 = spirv.Constant 200704 : i64
      %6 = spirv.IMul %cst200704_i64, %arg1 : i64
      %7 = spirv.IAdd %cst0_i64, %6 : i64
      %cst3584_i64 = spirv.Constant 3584 : i64
      %8 = spirv.IMul %cst3584_i64, %1 : i64
      %9 = spirv.IAdd %7, %8 : i64
      %cst64_i64 = spirv.Constant 64 : i64
      %10 = spirv.IMul %cst64_i64, %3 : i64
      %11 = spirv.IAdd %9, %10 : i64
      %cst1_i64 = spirv.Constant 1 : i64
      %12 = spirv.IMul %cst1_i64, %4 : i64
      %13 = spirv.IAdd %11, %12 : i64
      %14 = spirv.AccessChain %arg0[%13] : !spirv.ptr<!spirv.array<200704 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
      %15 = spirv.Load "CrossWorkgroup" %14 : i16
      %cst0_i64_1 = spirv.Constant 0 : i64
      %cst200704_i64_2 = spirv.Constant 200704 : i64
      %16 = spirv.IMul %cst200704_i64_2, %arg1 : i64
      %17 = spirv.IAdd %cst0_i64_1, %16 : i64
      %cst3584_i64_3 = spirv.Constant 3584 : i64
      %18 = spirv.IMul %cst3584_i64_3, %1 : i64
      %19 = spirv.IAdd %17, %18 : i64
      %cst64_i64_4 = spirv.Constant 64 : i64
      %20 = spirv.IMul %cst64_i64_4, %3 : i64
      %21 = spirv.IAdd %19, %20 : i64
      %cst1_i64_5 = spirv.Constant 1 : i64
      %22 = spirv.IMul %cst1_i64_5, %4 : i64
      %23 = spirv.IAdd %21, %22 : i64
      %24 = spirv.AccessChain %arg2[%23] : !spirv.ptr<!spirv.array<200704 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
      spirv.Store "CrossWorkgroup" %24, %15 : i16
      %25 = spirv.IAdd %4, %arg4 : i64
      spirv.Branch ^bb1(%25 : i64)
    ^bb3:  // pred: ^bb1
      spirv.mlir.merge
    }
    spirv.Return
  }
  spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
}

spirv.module @__spv__conv_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
  spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
  spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<215296 x i16>, CrossWorkgroup>, %arg1: i64, %arg2: !spirv.ptr<!spirv.array<36864 x i16>, CrossWorkgroup>, %arg3: !spirv.ptr<!spirv.array<200704 x i16>, CrossWorkgroup>, %arg4: i64, %arg5: i64, %arg6: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 56, 56>, workgroup_attributions = 0 : i64} {
    %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
    %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
    %1 = spirv.CompositeExtract %0[1 : i32] : vector<3xi64>
    %__builtin_var_WorkgroupId___addr_0 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
    %2 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr_0 : vector<3xi64>
    %3 = spirv.CompositeExtract %2[2 : i32] : vector<3xi64>
    spirv.mlir.loop {
      spirv.Branch ^bb1(%arg1 : i64)
    ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb2
      %5 = spirv.SLessThan %4, %arg4 : i64
      spirv.BranchConditional %5, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      spirv.mlir.loop {
        spirv.Branch ^bb1(%arg1 : i64)
      ^bb1(%7: i64):  // 2 preds: ^bb0, ^bb2
        %8 = spirv.SLessThan %7, %arg6 : i64
        spirv.BranchConditional %8, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        spirv.mlir.loop {
          spirv.Branch ^bb1(%arg1 : i64)
        ^bb1(%10: i64):  // 2 preds: ^bb0, ^bb2
          %11 = spirv.SLessThan %10, %arg6 : i64
          spirv.BranchConditional %11, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          spirv.mlir.loop {
            spirv.Branch ^bb1(%arg1 : i64)
          ^bb1(%13: i64):  // 2 preds: ^bb0, ^bb2
            %14 = spirv.SLessThan %13, %arg4 : i64
            spirv.BranchConditional %14, ^bb2, ^bb3
          ^bb2:  // pred: ^bb1
            %15 = spirv.IAdd %1, %7 : i64
            %16 = spirv.IAdd %3, %10 : i64
            %cst0_i64 = spirv.Constant 0 : i64
            %cst215296_i64 = spirv.Constant 215296 : i64
            %17 = spirv.IMul %cst215296_i64, %arg1 : i64
            %18 = spirv.IAdd %cst0_i64, %17 : i64
            %cst3712_i64 = spirv.Constant 3712 : i64
            %19 = spirv.IMul %cst3712_i64, %15 : i64
            %20 = spirv.IAdd %18, %19 : i64
            %cst64_i64 = spirv.Constant 64 : i64
            %21 = spirv.IMul %cst64_i64, %16 : i64
            %22 = spirv.IAdd %20, %21 : i64
            %cst1_i64 = spirv.Constant 1 : i64
            %23 = spirv.IMul %cst1_i64, %13 : i64
            %24 = spirv.IAdd %22, %23 : i64
            %25 = spirv.AccessChain %arg0[%24] : !spirv.ptr<!spirv.array<215296 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
            %26 = spirv.Load "CrossWorkgroup" %25 : i16
            %cst0_i64_1 = spirv.Constant 0 : i64
            %cst12288_i64 = spirv.Constant 12288 : i64
            %27 = spirv.IMul %cst12288_i64, %7 : i64
            %28 = spirv.IAdd %cst0_i64_1, %27 : i64
            %cst4096_i64 = spirv.Constant 4096 : i64
            %29 = spirv.IMul %cst4096_i64, %10 : i64
            %30 = spirv.IAdd %28, %29 : i64
            %cst64_i64_2 = spirv.Constant 64 : i64
            %31 = spirv.IMul %cst64_i64_2, %13 : i64
            %32 = spirv.IAdd %30, %31 : i64
            %cst1_i64_3 = spirv.Constant 1 : i64
            %33 = spirv.IMul %cst1_i64_3, %4 : i64
            %34 = spirv.IAdd %32, %33 : i64
            %35 = spirv.AccessChain %arg2[%34] : !spirv.ptr<!spirv.array<36864 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
            %36 = spirv.Load "CrossWorkgroup" %35 : i16
            %cst0_i64_4 = spirv.Constant 0 : i64
            %cst200704_i64 = spirv.Constant 200704 : i64
            %37 = spirv.IMul %cst200704_i64, %arg1 : i64
            %38 = spirv.IAdd %cst0_i64_4, %37 : i64
            %cst3584_i64 = spirv.Constant 3584 : i64
            %39 = spirv.IMul %cst3584_i64, %1 : i64
            %40 = spirv.IAdd %38, %39 : i64
            %cst64_i64_5 = spirv.Constant 64 : i64
            %41 = spirv.IMul %cst64_i64_5, %3 : i64
            %42 = spirv.IAdd %40, %41 : i64
            %cst1_i64_6 = spirv.Constant 1 : i64
            %43 = spirv.IMul %cst1_i64_6, %4 : i64
            %44 = spirv.IAdd %42, %43 : i64
            %45 = spirv.AccessChain %arg3[%44] : !spirv.ptr<!spirv.array<200704 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
            %46 = spirv.Load "CrossWorkgroup" %45 : i16

            %f32_26 = spirv.INTEL.ConvertBF16ToF %26 : i16 to f32
            %f32_36 = spirv.INTEL.ConvertBF16ToF %36 : i16 to f32
            %f32_46 = spirv.INTEL.ConvertBF16ToF %46 : i16 to f32

            %47 = spirv.FMul %f32_26, %f32_36 : f32

            %477 = spirv.INTEL.ConvertFToBF16 %47 : f32 to i16
            %478= spirv.INTEL.ConvertBF16ToF %477 : i16 to f32

            %48 = spirv.FAdd %f32_46, %478 : f32

            %bf16_48 = spirv.INTEL.ConvertFToBF16 %48 : f32 to i16

            %cst0_i64_7 = spirv.Constant 0 : i64
            %cst200704_i64_8 = spirv.Constant 200704 : i64
            %49 = spirv.IMul %cst200704_i64_8, %arg1 : i64
            %50 = spirv.IAdd %cst0_i64_7, %49 : i64
            %cst3584_i64_9 = spirv.Constant 3584 : i64
            %51 = spirv.IMul %cst3584_i64_9, %1 : i64
            %52 = spirv.IAdd %50, %51 : i64
            %cst64_i64_10 = spirv.Constant 64 : i64
            %53 = spirv.IMul %cst64_i64_10, %3 : i64
            %54 = spirv.IAdd %52, %53 : i64
            %cst1_i64_11 = spirv.Constant 1 : i64
            %55 = spirv.IMul %cst1_i64_11, %4 : i64
            %56 = spirv.IAdd %54, %55 : i64
            %57 = spirv.AccessChain %arg3[%56] : !spirv.ptr<!spirv.array<200704 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
            spirv.Store "CrossWorkgroup" %57, %bf16_48 : i16
            %58 = spirv.IAdd %13, %arg5 : i64
            spirv.Branch ^bb1(%58 : i64)
          ^bb3:  // pred: ^bb1
            spirv.mlir.merge
          }
          %12 = spirv.IAdd %10, %arg5 : i64
          spirv.Branch ^bb1(%12 : i64)
        ^bb3:  // pred: ^bb1
          spirv.mlir.merge
        }
        %9 = spirv.IAdd %7, %arg5 : i64
        spirv.Branch ^bb1(%9 : i64)
      ^bb3:  // pred: ^bb1
        spirv.mlir.merge
      }
      %6 = spirv.IAdd %4, %arg5 : i64
      spirv.Branch ^bb1(%6 : i64)
    ^bb3:  // pred: ^bb1
      spirv.mlir.merge
    }
    spirv.Return
  }
  spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
}


  gpu.module @copy_arg0_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<1x56x56x64xi16>, %arg1: index, %arg2: memref<1x56x56x64xi16>, %arg3: index, %arg4: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 56, 56>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  y
      %1 = gpu.block_id  z
      scf.for %arg5 = %arg1 to %arg3 step %arg4 {
        %2 = memref.load %arg0[%arg1, %0, %1, %arg5] : memref<1x56x56x64xi16>
        memref.store %2, %arg2[%arg1, %0, %1, %arg5] : memref<1x56x56x64xi16>
      }
      gpu.return
    }
  }
  gpu.module @init_kernel_0 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: i16, %arg1: memref<1x58x58x64xi16>, %arg2: index, %arg3: index, %arg4: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 58, 58>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  y
      %1 = gpu.block_id  z
      scf.for %arg5 = %arg2 to %arg3 step %arg4 {
        memref.store %arg0, %arg1[%arg2, %0, %1, %arg5] : memref<1x58x58x64xi16>
      }
      gpu.return
    }
  }

  gpu.module @init_kernel_1 attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: i16, %arg1: memref<1x56x56x64xi16>, %arg2: index, %arg3: index, %arg4: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 56, 56>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  y
      %1 = gpu.block_id  z
      scf.for %arg5 = %arg2 to %arg3 step %arg4 {
        memref.store %arg0, %arg1[%arg2, %0, %1, %arg5] : memref<1x56x56x64xi16>
      }
      gpu.return
    }
  }
  gpu.module @conv_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<1x58x58x64xi16>, %arg1: index, %arg2: memref<3x3x64x64xi16>, %arg3: memref<1x56x56x64xi16>, %arg4: index, %arg5: index, %arg6: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 56, 56>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  y
      %1 = gpu.block_id  z
      scf.for %arg7 = %arg1 to %arg4 step %arg5 {
        scf.for %arg8 = %arg1 to %arg6 step %arg5 {
          scf.for %arg9 = %arg1 to %arg6 step %arg5 {
            scf.for %arg10 = %arg1 to %arg4 step %arg5 {
              %2 = arith.addi %0, %arg8 : index
              %3 = arith.addi %1, %arg9 : index
              %4 = memref.load %arg0[%arg1, %2, %3, %arg10] : memref<1x58x58x64xi16>
              %5 = memref.load %arg2[%arg8, %arg9, %arg10, %arg7] : memref<3x3x64x64xi16>
              %6 = memref.load %arg3[%arg1, %0, %1, %arg7] : memref<1x56x56x64xi16>

              %bf16_4 = arith.bitcast %4 : i16 to bf16
              %bf16_5 = arith.bitcast %5 : i16 to bf16
              %bf16_6 = arith.bitcast %6 : i16 to bf16

              %f32_4 = arith.extf %bf16_4 : bf16 to f32
              %f32_5 = arith.extf %bf16_5 : bf16 to f32
              %f32_6 = arith.extf %bf16_6 : bf16 to f32

              %7 = arith.mulf %f32_4, %f32_5 : f32
              %8 = arith.addf %f32_6, %7 : f32

              %9 = arith.truncf %8 : f32 to bf16
              %10 = arith.bitcast %9 : bf16 to i16
              memref.store %10, %arg3[%arg1, %0, %1, %arg7] : memref<1x56x56x64xi16>
            }
          }
        }
      }
      gpu.return
    }
  }

}
