// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

module @argmax attributes {gpu.container_module} {
  memref.global "private" constant @__constant_1x4x4x3xbf16 : memref<1x4x4x3xbf16> = dense<[[[[9.000000e+00, 8.000000e+00, 0.000000e+00], [1.000000e+00, 5.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 7.000000e+00], [8.000000e+00, 2.000000e+00, 2.000000e+00]], [[8.000000e+00, 0.000000e+00, 4.000000e+00], [7.000000e+00, 5.000000e+00, 5.000000e+00], [8.000000e+00, 2.000000e+00, 0.000000e+00], [0.000000e+00, 9.000000e+00, 5.000000e+00]], [[4.000000e+00, 7.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00, 1.000000e+00], [3.000000e+00, 3.000000e+00, 6.000000e+00], [8.000000e+00, 0.000000e+00, 1.000000e+00]], [[2.000000e+00, 8.000000e+00, 4.000000e+00], [0.000000e+00, 5.000000e+00, 5.000000e+00], [6.000000e+00, 1.000000e+00, 1.000000e+00], [3.000000e+00, 3.000000e+00, 1.000000e+00]]]]>

  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_1x4x4x3xbf16 : memref<1x4x4x3xbf16>
    %1 = call @test(%0) : (memref<1x4x4x3xbf16>) -> memref<i32>
    %cast = memref.cast %1 : memref<i32> to memref<*xi32>
    call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
    // CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
    // CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [] strides = {{.*}} data =
    // CHECK:   0
    return
  }
  func.func private @printMemrefI32(memref<*xi32>) attributes {llvm.emit_c_interface}

  func.func @test(%arg0: memref<1x4x4x3xbf16>) -> memref<i32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0xFF80 : bf16
    %c0_i32 = arith.constant 0 : i32
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index


    %memref_arg0_bf16 = gpu.alloc  host_shared () : memref<1x4x4x3xbf16>
    memref.copy %arg0, %memref_arg0_bf16 : memref<1x4x4x3xbf16> to memref<1x4x4x3xbf16>
    %memref_kernel_arg0_i8 = gpu.alloc  host_shared () : memref<96xi8>
    %memref_kernel_arg0_bf16 = memref.view %memref_kernel_arg0_i8[%c0][] : memref<96xi8> to memref<1x4x4x3xbf16>
    memref.copy %memref_arg0_bf16, %memref_kernel_arg0_bf16 : memref<1x4x4x3xbf16> to memref<1x4x4x3xbf16>
    %memref_kernel_arg0_i16 = memref.view %memref_kernel_arg0_i8[%c0][] : memref<96xi8> to memref<1x4x4x3xi16>


    %alloc = memref.alloc() {alignment = 64 : i64} : memref<bf16>
    memref.store %cst, %alloc[] : memref<bf16>
    %memref_0 = gpu.alloc  host_shared () : memref<bf16>
    memref.copy %alloc, %memref_0 : memref<bf16> to memref<bf16>

    %memref_0_kernel_i8 = gpu.alloc  host_shared () : memref<2xi8>
    %memref_0_kernel_bf16 = memref.view %memref_0_kernel_i8[%c0][] : memref<2xi8> to memref<bf16>
    memref.copy %memref_0, %memref_0_kernel_bf16 : memref<bf16> to memref<bf16>
    %memref_0_kernel_i16 = memref.view %memref_0_kernel_i8[%c0][] : memref<2xi8> to memref<i16>

    gpu.launch_func  @cmpf_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref_kernel_arg0_i16 : memref<1x4x4x3xi16>, %c0 : index, %memref_0_kernel_i16 : memref<i16>, %c3 : index, %c1 : index, %c4 : index)

    %memref_1 = gpu.alloc  () : memref<1x4x4x3xi32>
    gpu.launch_func  @store_i32_kernel::@test_kernel blocks in (%c1, %c4, %c4) threads in (%c1, %c1, %c1) args(%c0_i32 : i32, %memref_1 : memref<1x4x4x3xi32>, %c0 : index, %c3 : index, %c1 : index)


    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<i32>
    memref.store %c0_i32, %alloc_2[] : memref<i32>
    %memref_3 = gpu.alloc  host_shared () : memref<i32>
    memref.copy %alloc_2, %memref_3 : memref<i32> to memref<i32>
    gpu.launch_func  @argmax_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref_kernel_arg0_i16 : memref<1x4x4x3xi16>, %c0 : index, %memref_0_kernel_i16 : memref<i16>, %memref_1 : memref<1x4x4x3xi32>, %memref_3 : memref<i32>, %c0_i32 : i32, %c3 : index, %c1 : index, %c4 : index)
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<i32>
    %0 = memref.load %memref_3[] : memref<i32>
    memref.store %0, %alloc_4[] : memref<i32>
    gpu.dealloc  %memref_1 : memref<1x4x4x3xi32>
    gpu.dealloc  %memref_arg0_bf16 : memref<1x4x4x3xbf16>
    gpu.dealloc %memref_0_kernel_i8 : memref<2xi8>

    return %alloc_4 : memref<i32>
  }

    spirv.module @__spv__cmpf_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<48 x i16>, CrossWorkgroup>, %arg1: i64, %arg2: !spirv.ptr<!spirv.array<1 x i16>, CrossWorkgroup>, %arg3: i64, %arg4: i64, %arg5: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, workgroup_attributions = 0 : i64} {
        spirv.mlir.loop {
        spirv.Branch ^bb1(%arg1 : i64)
        ^bb1(%0: i64):  // 2 preds: ^bb0, ^bb2
        %1 = spirv.SLessThan %0, %arg5 : i64
        spirv.BranchConditional %1, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
        spirv.mlir.loop {
            spirv.Branch ^bb1(%arg1 : i64)
        ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb2
            %4 = spirv.SLessThan %3, %arg5 : i64
            spirv.BranchConditional %4, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
            spirv.mlir.loop {
            spirv.Branch ^bb1(%arg1 : i64)
            ^bb1(%6: i64):  // 2 preds: ^bb0, ^bb2
            %7 = spirv.SLessThan %6, %arg3 : i64
            spirv.BranchConditional %7, ^bb2, ^bb3
            ^bb2:  // pred: ^bb1
            %cst0_i64 = spirv.Constant 0 : i64
            %cst48_i64 = spirv.Constant 48 : i64
            %8 = spirv.IMul %cst48_i64, %arg1 : i64
            %9 = spirv.IAdd %cst0_i64, %8 : i64
            %cst12_i64 = spirv.Constant 12 : i64
            %10 = spirv.IMul %cst12_i64, %0 : i64
            %11 = spirv.IAdd %9, %10 : i64
            %cst3_i64 = spirv.Constant 3 : i64
            %12 = spirv.IMul %cst3_i64, %3 : i64
            %13 = spirv.IAdd %11, %12 : i64
            %cst1_i64 = spirv.Constant 1 : i64
            %14 = spirv.IMul %cst1_i64, %6 : i64
            %15 = spirv.IAdd %13, %14 : i64
            %16 = spirv.AccessChain %arg0[%15] : !spirv.ptr<!spirv.array<48 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
            %17 = spirv.Load "CrossWorkgroup" %16 : i16
            %fp32_17 = spirv.INTEL.ConvertBF16ToF %17 : i16 to f32

            %cst0_i64_0 = spirv.Constant 0 : i64
            %18 = spirv.AccessChain %arg2[%cst0_i64_0] : !spirv.ptr<!spirv.array<1 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
            %19 = spirv.Load "CrossWorkgroup" %18 : i16
            %fp32_19 = spirv.INTEL.ConvertBF16ToF %19 : i16 to f32

            %20 = spirv.FOrdGreaterThan %fp32_19, %fp32_17 : f32
            %21 = spirv.Select %20, %fp32_19, %fp32_17 : i1, f32

            %bf16_21 = spirv.INTEL.ConvertFToBF16 %21 : f32 to i16
            %cst0_i64_1 = spirv.Constant 0 : i64
            %22 = spirv.AccessChain %arg2[%cst0_i64_1] : !spirv.ptr<!spirv.array<1 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
            spirv.Store "CrossWorkgroup" %22, %bf16_21 : i16
            %23 = spirv.IAdd %6, %arg4 : i64
            spirv.Branch ^bb1(%23 : i64)
            ^bb3:  // pred: ^bb1
            spirv.mlir.merge
            }
            %5 = spirv.IAdd %3, %arg4 : i64
            spirv.Branch ^bb1(%5 : i64)
        ^bb3:  // pred: ^bb1
            spirv.mlir.merge
        }
        %2 = spirv.IAdd %0, %arg4 : i64
        spirv.Branch ^bb1(%2 : i64)
        ^bb3:  // pred: ^bb1
        spirv.mlir.merge
        }
        spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel
    }

    spirv.module @__spv__store_i32_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    spirv.func @test_kernel(%arg0: i32, %arg1: !spirv.ptr<!spirv.array<48 x i32>, CrossWorkgroup>, %arg2: i64, %arg3: i64, %arg4: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 4, 4>, workgroup_attributions = 0 : i64} {
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
        %cst48_i64 = spirv.Constant 48 : i64
        %6 = spirv.IMul %cst48_i64, %arg2 : i64
        %7 = spirv.IAdd %cst0_i64, %6 : i64
        %cst12_i64 = spirv.Constant 12 : i64
        %8 = spirv.IMul %cst12_i64, %1 : i64
        %9 = spirv.IAdd %7, %8 : i64
        %cst3_i64 = spirv.Constant 3 : i64
        %10 = spirv.IMul %cst3_i64, %3 : i64
        %11 = spirv.IAdd %9, %10 : i64
        %cst1_i64 = spirv.Constant 1 : i64
        %12 = spirv.IMul %cst1_i64, %4 : i64
        %13 = spirv.IAdd %11, %12 : i64
        %14 = spirv.AccessChain %arg1[%13] : !spirv.ptr<!spirv.array<48 x i32>, CrossWorkgroup>, i64 -> !spirv.ptr<i32, CrossWorkgroup>
        spirv.Store "CrossWorkgroup" %14, %arg0 : i32
        %15 = spirv.IAdd %4, %arg4 : i64
        spirv.Branch ^bb1(%15 : i64)
        ^bb3:  // pred: ^bb1
        spirv.mlir.merge
        }
        spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin_var_WorkgroupId__
    }

    spirv.module @__spv__argmax_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int64, Kernel, Addresses], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<48 x i16>, CrossWorkgroup>, %arg1: i64, %arg2: !spirv.ptr<!spirv.array<1 x i16>, CrossWorkgroup>, %arg3: !spirv.ptr<!spirv.array<48 x i32>, CrossWorkgroup>, %arg4: !spirv.ptr<!spirv.array<1 x i32>, CrossWorkgroup>, %arg5: i32, %arg6: i64, %arg7: i64, %arg8: i64) "None" attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, workgroup_attributions = 0 : i64} {
        spirv.mlir.loop {
        spirv.Branch ^bb1(%arg1 : i64)
        ^bb1(%0: i64):  // 2 preds: ^bb0, ^bb2
        %1 = spirv.SLessThan %0, %arg8 : i64
        spirv.BranchConditional %1, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
        spirv.mlir.loop {
            spirv.Branch ^bb1(%arg1 : i64)
        ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb2
            %4 = spirv.SLessThan %3, %arg8 : i64
            spirv.BranchConditional %4, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
            spirv.mlir.loop {
            spirv.Branch ^bb1(%arg1 : i64)
            ^bb1(%6: i64):  // 2 preds: ^bb0, ^bb2
            %7 = spirv.SLessThan %6, %arg6 : i64
            spirv.BranchConditional %7, ^bb2, ^bb3
            ^bb2:  // pred: ^bb1
            %cst0_i64 = spirv.Constant 0 : i64
            %cst48_i64 = spirv.Constant 48 : i64
            %8 = spirv.IMul %cst48_i64, %arg1 : i64
            %9 = spirv.IAdd %cst0_i64, %8 : i64
            %cst12_i64 = spirv.Constant 12 : i64
            %10 = spirv.IMul %cst12_i64, %0 : i64
            %11 = spirv.IAdd %9, %10 : i64
            %cst3_i64 = spirv.Constant 3 : i64
            %12 = spirv.IMul %cst3_i64, %3 : i64
            %13 = spirv.IAdd %11, %12 : i64
            %cst1_i64 = spirv.Constant 1 : i64
            %14 = spirv.IMul %cst1_i64, %6 : i64
            %15 = spirv.IAdd %13, %14 : i64
            %16 = spirv.AccessChain %arg0[%15] : !spirv.ptr<!spirv.array<48 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
            %17 = spirv.Load "CrossWorkgroup" %16 : i16
            %f32_17 = spirv.INTEL.ConvertBF16ToF %17 : i16 to f32

            %cst0_i64_0 = spirv.Constant 0 : i64
            %18 = spirv.AccessChain %arg2[%cst0_i64_0] : !spirv.ptr<!spirv.array<1 x i16>, CrossWorkgroup>, i64 -> !spirv.ptr<i16, CrossWorkgroup>
            %19 = spirv.Load "CrossWorkgroup" %18 : i16
            %f32_19 = spirv.INTEL.ConvertBF16ToF %19 : i16 to f32

            %cst0_i64_1 = spirv.Constant 0 : i64
            %cst48_i64_2 = spirv.Constant 48 : i64
            %20 = spirv.IMul %cst48_i64_2, %arg1 : i64
            %21 = spirv.IAdd %cst0_i64_1, %20 : i64
            %cst12_i64_3 = spirv.Constant 12 : i64
            %22 = spirv.IMul %cst12_i64_3, %0 : i64
            %23 = spirv.IAdd %21, %22 : i64
            %cst3_i64_4 = spirv.Constant 3 : i64
            %24 = spirv.IMul %cst3_i64_4, %3 : i64
            %25 = spirv.IAdd %23, %24 : i64
            %cst1_i64_5 = spirv.Constant 1 : i64
            %26 = spirv.IMul %cst1_i64_5, %6 : i64
            %27 = spirv.IAdd %25, %26 : i64
            %28 = spirv.AccessChain %arg3[%27] : !spirv.ptr<!spirv.array<48 x i32>, CrossWorkgroup>, i64 -> !spirv.ptr<i32, CrossWorkgroup>
            %29 = spirv.Load "CrossWorkgroup" %28 : i32
            %cst0_i64_6 = spirv.Constant 0 : i64
            %30 = spirv.AccessChain %arg4[%cst0_i64_6] : !spirv.ptr<!spirv.array<1 x i32>, CrossWorkgroup>, i64 -> !spirv.ptr<i32, CrossWorkgroup>
            %31 = spirv.Load "CrossWorkgroup" %30 : i32
            %32 = spirv.FOrdEqual %f32_17, %f32_19 : f32
            %33 = spirv.Select %32, %29, %arg5 : i1, i32
            %34 = spirv.UGreaterThan %31, %33 : i32
            %35 = spirv.Select %34, %31, %33 : i1, i32
            %cst0_i64_7 = spirv.Constant 0 : i64
            %36 = spirv.AccessChain %arg4[%cst0_i64_7] : !spirv.ptr<!spirv.array<1 x i32>, CrossWorkgroup>, i64 -> !spirv.ptr<i32, CrossWorkgroup>
            spirv.Store "CrossWorkgroup" %36, %35 : i32
            %37 = spirv.IAdd %6, %arg7 : i64
            spirv.Branch ^bb1(%37 : i64)
            ^bb3:  // pred: ^bb1
            spirv.mlir.merge
            }
            %5 = spirv.IAdd %3, %arg7 : i64
            spirv.Branch ^bb1(%5 : i64)
        ^bb3:  // pred: ^bb1
            spirv.mlir.merge
        }
        %2 = spirv.IAdd %0, %arg7 : i64
        spirv.Branch ^bb1(%2 : i64)
        ^bb3:  // pred: ^bb1
        spirv.mlir.merge
        }
        spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel
    }

  gpu.module @cmpf_kernel attributes {gpu.binary = "", spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<1x4x4x3xi16>, %arg1: index, %arg2: memref<i16>, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      scf.for %arg6 = %arg1 to %arg5 step %arg4 {
        scf.for %arg7 = %arg1 to %arg5 step %arg4 {
          scf.for %arg8 = %arg1 to %arg3 step %arg4 {
            %0 = memref.load %arg0[%arg1, %arg6, %arg7, %arg8] : memref<1x4x4x3xi16>
            %1 = memref.load %arg2[] : memref<i16>
            %bf16_0 = arith.bitcast %0 : i16 to bf16
            %bf16_1 = arith.bitcast %1 : i16 to bf16

            %f32_0 = arith.extf %bf16_0 : bf16 to f32
            %f32_1 = arith.extf %bf16_1 : bf16 to f32

            %2 = arith.cmpf ogt, %f32_1, %f32_0 : f32
            %3 = arith.select %2, %f32_1, %f32_0 : f32

            %5 = arith.truncf %3 : f32 to bf16
            %6 = arith.bitcast %5 : bf16 to i16
            memref.store %6, %arg2[] : memref<i16>
          }
        }
      }
      gpu.return
    }
  }

  gpu.module @store_i32_kernel attributes {gpu.binary = "", spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: i32, %arg1: memref<1x4x4x3xi32>, %arg2: index, %arg3: index, %arg4: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 4, 4>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = gpu.block_id  y
      %1 = gpu.block_id  z
      scf.for %arg5 = %arg2 to %arg3 step %arg4 {
        memref.store %arg0, %arg1[%arg2, %0, %1, %arg5] : memref<1x4x4x3xi32>
      }
      gpu.return
    }
  }
  gpu.module @argmax_kernel attributes {gpu.binary = "", spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<1x4x4x3xi16>, %arg1: index, %arg2: memref<i16>, %arg3: memref<1x4x4x3xi32>, %arg4: memref<i32>, %arg5: i32, %arg6: index, %arg7: index, %arg8: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      scf.for %arg9 = %arg1 to %arg8 step %arg7 {
        scf.for %arg10 = %arg1 to %arg8 step %arg7 {
          scf.for %arg11 = %arg1 to %arg6 step %arg7 {
            %0 = memref.load %arg0[%arg1, %arg9, %arg10, %arg11] : memref<1x4x4x3xi16>
            %1 = memref.load %arg2[] : memref<i16>
            %2 = memref.load %arg3[%arg1, %arg9, %arg10, %arg11] : memref<1x4x4x3xi32>
            %3 = memref.load %arg4[] : memref<i32>
            %bf16_0 = arith.bitcast %0 : i16 to bf16
            %bf16_1 = arith.bitcast %1 : i16 to bf16

            %f32_0 = arith.extf %bf16_0 : bf16 to f32
            %f32_1 = arith.extf %bf16_1 : bf16 to f32

            %4 = arith.cmpf oeq, %f32_0, %f32_1 : f32
            %5 = arith.select %4, %2, %arg5 : i32
            %6 = arith.cmpi ugt, %3, %5 : i32
            %7 = arith.select %6, %3, %5 : i32
            memref.store %7, %arg4[] : memref<i32>
          }
        }
      }
      gpu.return
    }
  }
}
