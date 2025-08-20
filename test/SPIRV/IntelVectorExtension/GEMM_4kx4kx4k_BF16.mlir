// RUN: IMEX_ENABLE_LARGE_REG_FILE=1 %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%mlir_runner_utils,%irunner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: IMEX_ENABLE_LARGE_REG_FILE=1 %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%mlir_runner_utils,%irunner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

module attributes {gpu.container_module}  {

  // function to setup the launch and launch the kernel
  // args: size_t systolic_depth, size_t repeat_cnt, size_t N
  func.func @gemm4k_gpu(%arg_M: index, %arg_N: index, %arg_K: index, %arg_C : memref<?xi16>, %arg_B : memref<?xi16>, %arg_A : memref<?xi16>) {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c96 = arith.constant 96 : index
    %c256 = arith.constant 256 : index
    %stride_am = arith.constant 4096 : i32
    %stride_bk = arith.constant 4096 : i32
    %stride_cm = arith.constant 4096 : i32
    %M = arith.constant 4096 : i32
    %N = arith.constant 4096 : i32
    %K = arith.constant 4096 : i32

    // Since we are using only one DPAS instruction we are launching,
    // 256 workgroup and, 32 thread per workgroup
    gpu.launch_func @dpas_module::@gemm4k_kernel blocks in (%c256, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg_A : memref<?xi16>, %arg_B : memref<?xi16>, %arg_C : memref<?xi16>, %M : i32, %N : i32, %K : i32, %stride_am : i32, %stride_bk : i32, %stride_cm : i32)
    return
  }

  spirv.module @__spv__dpas_module Physical64 OpenCL requires #spirv.vce<v1.0, [Int8, Int16, Int64, Float16, Kernel, Addresses, Linkage, Vector16, VectorAnyINTEL, Float16Buffer, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, FunctionFloatControlINTEL], [SPV_INTEL_float_controls2, SPV_INTEL_vector_compute]> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorAnyINTEL, VectorComputeINTEL, FunctionFloatControlINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    spirv.GlobalVariable @__builtin_var_SubgroupId__ built_in("SubgroupId") : !spirv.ptr<i32, Input>
    spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>

    spirv.func @gemm4k_kernel(%arg0_: !spirv.ptr<i16, CrossWorkgroup>, %arg1_: !spirv.ptr<i16, CrossWorkgroup>, %arg2_: !spirv.ptr<i16, CrossWorkgroup>, %arg3: i32, %arg4: i32, %arg5: i32 , %arg6: i32, %arg7: i32, %arg8: i32) "None" attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>, workgroup_attributions = 0 : i64} {
        %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
        // get WG ID for x dim
        %wg_id = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi64>
        %wg_id_x_i64 = spirv.CompositeExtract %wg_id[0 : i32] : vector<3xi64>
        %wg_id_x = spirv.UConvert %wg_id_x_i64 : i64 to i32

        %cst_vec_128xf32 = spirv.Constant dense<0.000000e+00> : vector<128xf32>
        %0 = spirv.Undef : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %1 = spirv.CompositeInsert %cst_vec_128xf32, %0[0 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %2 = spirv.CompositeInsert %cst_vec_128xf32, %1[1 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %3 = spirv.CompositeInsert %cst_vec_128xf32, %2[2 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %4 = spirv.CompositeInsert %cst_vec_128xf32, %3[3 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %5 = spirv.CompositeInsert %cst_vec_128xf32, %4[4 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %6 = spirv.CompositeInsert %cst_vec_128xf32, %5[5 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %7 = spirv.CompositeInsert %cst_vec_128xf32, %6[6 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %8 = spirv.CompositeInsert %cst_vec_128xf32, %7[7 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %9 = spirv.CompositeInsert %cst_vec_128xf32, %8[8 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %10 = spirv.CompositeInsert %cst_vec_128xf32, %9[9 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %11 = spirv.CompositeInsert %cst_vec_128xf32, %10[10 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %12 = spirv.CompositeInsert %cst_vec_128xf32, %11[11 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %13 = spirv.CompositeInsert %cst_vec_128xf32, %12[12 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %14 = spirv.CompositeInsert %cst_vec_128xf32, %13[13 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %15 = spirv.CompositeInsert %cst_vec_128xf32, %14[14 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %16 = spirv.CompositeInsert %cst_vec_128xf32, %15[15 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %cst32_i32 = spirv.Constant 32 : i32
        %cst31_i32 = spirv.Constant 31 : i32
        %cst1_i32 = spirv.Constant 1 : i32
        %cst0_i32 = spirv.Constant 0 : i32
        %17 = spirv.Undef : !spirv.struct<(i32)>
        %18 = spirv.IAdd %arg5, %cst31_i32 : i32
        %19 = spirv.SDiv %18, %cst32_i32 : i32
        %20 = spirv.IMul %cst0_i32, %arg6 : i32
        %21 = spirv.IAdd %20, %cst0_i32 : i32
        %22 = spirv.Undef : !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>
        %23 = spirv.ConvertUToPtr %21 : i32 to !spirv.ptr<i16, CrossWorkgroup>
        %24 = spirv.CompositeInsert %arg0_, %22[0 : i32] : !spirv.ptr<i16, CrossWorkgroup> into !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>
        %25 = spirv.CompositeInsert %23, %24[1 : i32] : !spirv.ptr<i16, CrossWorkgroup> into !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>
        %26 = spirv.IMul %cst0_i32, %arg7 : i32
        %27 = spirv.IAdd %26, %cst0_i32 : i32
        %28 = spirv.ConvertUToPtr %27 : i32 to !spirv.ptr<i16, CrossWorkgroup>
        %29 = spirv.CompositeInsert %arg1_, %22[0 : i32] : !spirv.ptr<i16, CrossWorkgroup> into !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>
        %30 = spirv.CompositeInsert %28, %29[1 : i32] : !spirv.ptr<i16, CrossWorkgroup> into !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>
        // convert local SG ID to global
        %__builtin_var_SubgroupId___addr = spirv.mlir.addressof @__builtin_var_SubgroupId__ : !spirv.ptr<i32, Input>
        %t0_31 = spirv.Load "Input" %__builtin_var_SubgroupId___addr : i32
        %t1_31 = spirv.IMul %wg_id_x, %cst32_i32 : i32
        %31 = spirv.IAdd %t0_31, %t1_31 : i32

        %true = spirv.Constant true
        %cst2_i8 = spirv.Constant 2 : i8
        %cst1_i8 = spirv.Constant 1 : i8
        %cst8_i32 = spirv.Constant 8 : i32
        %cst0_i8 = spirv.Constant 0 : i8
        %cst8191_i32 = spirv.Constant 8191 : i32
        %cst4095_i32 = spirv.Constant 4095 : i32
        %cst4_i32 = spirv.Constant 4 : i32
        %cst16_i32 = spirv.Constant 16 : i32
        %cst64_i32 = spirv.Constant 64 : i32
        %cst256_i32 = spirv.Constant 256 : i32
        %32 = spirv.UDiv %31, %cst32_i32 : i32
        %33 = spirv.UDiv %32, %cst16_i32 : i32
        %34 = spirv.IMul %33, %cst256_i32 : i32
        %35 = spirv.UMod %31, %cst32_i32 : i32
        %36 = spirv.UDiv %35, %cst4_i32 : i32
        %37 = spirv.UMod %35, %cst4_i32 : i32
        %38 = spirv.IMul %36, %cst32_i32 : i32
        %39 = spirv.ConvertPtrToU %arg0_ : !spirv.ptr<i16, CrossWorkgroup> to i64
        %40 = spirv.IMul %37, %cst8_i32 : i32
        %41 = spirv.IAdd %38, %40 : i32
        %42 = spirv.IAdd %34, %41 : i32

        // prefetch matrix a for 3 stages
        %43 = spirv.FunctionCall @llvm_genx_lsc_prefetch2d_stateless_v64i32_i1_i64(%true, %cst2_i8, %cst2_i8, %cst2_i8, %cst1_i8, %cst1_i8, %cst32_i32, %cst8_i32, %cst0_i8, %39, %cst8191_i32, %cst4095_i32, %cst8191_i32, %cst0_i32, %42) : (i1, i8, i8, i8, i8, i8, i32, i32, i8, i64, i32, i32, i32, i32, i32) -> vector<64xi32>
        %44 = spirv.IAdd %cst0_i32, %cst32_i32 : i32
        %45 = spirv.FunctionCall @llvm_genx_lsc_prefetch2d_stateless_v64i32_i1_i64(%true, %cst2_i8, %cst2_i8, %cst2_i8, %cst1_i8, %cst1_i8, %cst32_i32, %cst8_i32, %cst0_i8, %39, %cst8191_i32, %cst4095_i32, %cst8191_i32, %44, %42) : (i1, i8, i8, i8, i8, i8, i32, i32, i8, i64, i32, i32, i32, i32, i32) -> vector<64xi32>
        %46 = spirv.IAdd %cst0_i32, %cst64_i32 : i32
        %47 = spirv.FunctionCall @llvm_genx_lsc_prefetch2d_stateless_v64i32_i1_i64(%true, %cst2_i8, %cst2_i8, %cst2_i8, %cst1_i8, %cst1_i8, %cst32_i32, %cst8_i32, %cst0_i8, %39, %cst8191_i32, %cst4095_i32, %cst8191_i32, %46, %42) : (i1, i8, i8, i8, i8, i8, i32, i32, i8, i64, i32, i32, i32, i32, i32) -> vector<64xi32>
        %48 = spirv.UMod %32, %cst16_i32 : i32
        %49 = spirv.IMul %48, %cst256_i32 : i32
        %50 = spirv.IMul %37, %cst64_i32 : i32
        %51 = spirv.UDiv %36, %cst4_i32 : i32
        %52 = spirv.UMod %36, %cst4_i32 : i32
        %53 = spirv.ConvertPtrToU %arg1_ : !spirv.ptr<i16, CrossWorkgroup> to i64
        %54 = spirv.IMul %52, %cst16_i32 : i32
        %55 = spirv.IAdd %50, %54 : i32
        %56 = spirv.IAdd %49, %55 : i32
        %57 = spirv.IMul %51, %cst16_i32 : i32
        %58 = spirv.IAdd %cst0_i32, %57 : i32

        // prefetch matrix b for 3 stages
        %59 = spirv.FunctionCall @llvm_genx_lsc_prefetch2d_stateless_v64i64_i1_i64(%true, %cst2_i8, %cst2_i8, %cst2_i8, %cst1_i8, %cst1_i8, %cst16_i32, %cst16_i32, %cst1_i8, %53, %cst8191_i32, %cst4095_i32, %cst8191_i32, %56, %58) : (i1, i8, i8, i8, i8, i8, i32, i32, i8, i64, i32, i32, i32, i32, i32) -> vector<64xi64>
        %60 = spirv.IAdd %58, %cst32_i32 : i32
        %61 = spirv.FunctionCall @llvm_genx_lsc_prefetch2d_stateless_v64i64_i1_i64(%true, %cst2_i8, %cst2_i8, %cst2_i8, %cst1_i8, %cst1_i8, %cst16_i32, %cst16_i32, %cst1_i8, %53, %cst8191_i32, %cst4095_i32, %cst8191_i32, %56, %60) : (i1, i8, i8, i8, i8, i8, i32, i32, i8, i64, i32, i32, i32, i32, i32) -> vector<64xi64>
        %62 = spirv.IAdd %58, %cst64_i32 : i32
        %63 = spirv.FunctionCall @llvm_genx_lsc_prefetch2d_stateless_v64i64_i1_i64(%true, %cst2_i8, %cst2_i8, %cst2_i8, %cst1_i8, %cst1_i8, %cst16_i32, %cst16_i32, %cst1_i8, %53, %cst8191_i32, %cst4095_i32, %cst8191_i32, %56, %62) : (i1, i8, i8, i8, i8, i8, i32, i32, i8, i64, i32, i32, i32, i32, i32) -> vector<64xi64>
        spirv.Branch ^bb1(%cst0_i32, %16, %25, %30 : i32, !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>, !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>, !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>)
    ^bb1(%64: i32, %65: !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>, %66: !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>, %67: !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>):  // 2 preds: ^bb0, ^bb2
        %68 = spirv.SLessThan %64, %19 : i32
        spirv.BranchConditional %68, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
        %69 = spirv.CompositeExtract %66[0 : i32] : !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>
        %70 = spirv.CompositeExtract %66[1 : i32] : !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>
        // convert local SG ID to global SG ID
        %t0_71 = spirv.Load "Input" %__builtin_var_SubgroupId___addr : i32
        %t1_71 = spirv.IMul %wg_id_x, %cst32_i32 : i32
        %71 = spirv.IAdd %t0_71, %t1_71 : i32

        %cst_vec_8xi32 = spirv.Constant dense<0> : vector<8xi32>
        %cst2_i32 = spirv.Constant 2 : i32
        %cst538968065_i32 = spirv.Constant 538968065 : i32
        %72 = spirv.VectorInsertDynamic %cst538968065_i32, %cst_vec_8xi32[%cst2_i32] : vector<8xi32>, i32
        %cst3_i8 = spirv.Constant 3 : i8
        %cst33554436_i32 = spirv.Constant 33554436 : i32

        //named_barrier signal
        spirv.FunctionCall @llvm_genx_raw_send2_noresult_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst3_i8, %cst0_i32, %cst33554436_i32, %72) : (i8, i8, i1, i8, i8, i32, i32, vector<8xi32>) -> ()
        %73 = spirv.ConvertPtrToU %69 : !spirv.ptr<i16, CrossWorkgroup> to i64
        %74 = spirv.ConvertPtrToU %70 : !spirv.ptr<i16, CrossWorkgroup> to i32
        %75 = spirv.UDiv %71, %cst32_i32 : i32
        %76 = spirv.UDiv %75, %cst16_i32 : i32
        %77 = spirv.IMul %76, %cst256_i32 : i32
        %78 = spirv.UMod %71, %cst32_i32 : i32
        %79 = spirv.UDiv %78, %cst4_i32 : i32
        %80 = spirv.UMod %78, %cst4_i32 : i32
        %81 = spirv.IMul %79, %cst32_i32 : i32
        %cst16_i8 = spirv.Constant 16 : i8
        %cst15_i8 = spirv.Constant 15 : i8
        %cst50856451_i32 = spirv.Constant 50856451 : i32
        %cst32_i64 = spirv.Constant 32 : i64
        %82 = spirv.ShiftRightLogical %73, %cst32_i64 : i64, i64
        %83 = spirv.UConvert %82 : i64 to i32
        %84 = spirv.UConvert %73 : i64 to i32
        %85 = spirv.VectorInsertDynamic %84, %cst_vec_8xi32[%cst0_i32] : vector<8xi32>, i32
        %86 = spirv.VectorInsertDynamic %83, %85[%cst1_i32] : vector<8xi32>, i32
        %87 = spirv.VectorInsertDynamic %cst8191_i32, %86[%cst2_i32] : vector<8xi32>, i32
        %cst3_i32 = spirv.Constant 3 : i32
        %88 = spirv.VectorInsertDynamic %cst4095_i32, %87[%cst3_i32] : vector<8xi32>, i32
        %89 = spirv.VectorInsertDynamic %cst8191_i32, %88[%cst4_i32] : vector<8xi32>, i32
        %cst7_i32 = spirv.Constant 7 : i32
        %cst7951_i32 = spirv.Constant 7951 : i32
        %90 = spirv.VectorInsertDynamic %cst7951_i32, %89[%cst7_i32] : vector<8xi32>, i32
        %cst_vec_256xi32 = spirv.Constant dense<0> : vector<256xi32>
        %91 = spirv.IAdd %77, %81 : i32
        %92 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32] %90 : vector<8xi32>, %90 : vector<8xi32> -> vector<8xi32>
        %cst5_i32 = spirv.Constant 5 : i32
        %93 = spirv.VectorInsertDynamic %74, %92[%cst5_i32] : vector<8xi32>, i32
        %cst6_i32 = spirv.Constant 6 : i32
        %94 = spirv.VectorInsertDynamic %91, %93[%cst6_i32] : vector<8xi32>, i32

        //load matrix a 2x32x16, spilt into 2x4x8x16
        %95 = spirv.FunctionCall @llvm_genx_raw_send2_v256i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst16_i8, %cst15_i8, %cst0_i32, %cst50856451_i32, %94, %cst_vec_256xi32) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<256xi32>) -> vector<256xi32>
        %96 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32] %95 : vector<256xi32>, %95 : vector<256xi32> -> vector<64xi32>
        %97 = spirv.VectorShuffle [64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %95 : vector<256xi32>, %95 : vector<256xi32> -> vector<64xi32>
        %98 = spirv.VectorShuffle [128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32] %95 : vector<256xi32>, %95 : vector<256xi32> -> vector<64xi32>
        %99 = spirv.VectorShuffle [192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %95 : vector<256xi32>, %95 : vector<256xi32> -> vector<64xi32>
        %100 = spirv.IAdd %74, %cst16_i32 : i32
        %101 = spirv.VectorInsertDynamic %100, %92[%cst5_i32] : vector<8xi32>, i32
        %102 = spirv.VectorInsertDynamic %91, %101[%cst6_i32] : vector<8xi32>, i32
        %103 = spirv.FunctionCall @llvm_genx_raw_send2_v256i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst16_i8, %cst15_i8, %cst0_i32, %cst50856451_i32, %102, %cst_vec_256xi32) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<256xi32>) -> vector<256xi32>
        %104 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32] %103 : vector<256xi32>, %103 : vector<256xi32> -> vector<64xi32>
        %105 = spirv.VectorShuffle [64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %103 : vector<256xi32>, %103 : vector<256xi32> -> vector<64xi32>
        %106 = spirv.VectorShuffle [128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32] %103 : vector<256xi32>, %103 : vector<256xi32> -> vector<64xi32>
        %107 = spirv.VectorShuffle [192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %103 : vector<256xi32>, %103 : vector<256xi32> -> vector<64xi32>
        %cst96_i32 = spirv.Constant 96 : i32
        %108 = spirv.IAdd %74, %cst96_i32 : i32
        %109 = spirv.IMul %80, %cst8_i32 : i32
        %110 = spirv.IAdd %81, %109 : i32
        %111 = spirv.IAdd %77, %110 : i32
        %cst34079235_i32 = spirv.Constant 34079235 : i32
        %112 = spirv.VectorInsertDynamic %108, %90[%cst5_i32] : vector<8xi32>, i32
        %113 = spirv.VectorInsertDynamic %111, %112[%cst6_i32] : vector<8xi32>, i32
        %cst1823_i32 = spirv.Constant 1823 : i32
        %114 = spirv.VectorInsertDynamic %cst1823_i32, %113[%cst7_i32] : vector<8xi32>, i32

        //prefetch a each subgroup in N dim prefetch 8x32
        spirv.FunctionCall @llvm_genx_raw_send2_noresult_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst15_i8, %cst0_i32, %cst34079235_i32, %114) : (i8, i8, i1, i8, i8, i32, i32, vector<8xi32>) -> ()
        %115 = spirv.Undef : !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %116 = spirv.CompositeInsert %96, %115[0 : i32] : vector<64xi32> into !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %117 = spirv.CompositeInsert %104, %116[1 : i32] : vector<64xi32> into !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %118 = spirv.CompositeInsert %97, %117[2 : i32] : vector<64xi32> into !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %119 = spirv.CompositeInsert %105, %118[3 : i32] : vector<64xi32> into !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %120 = spirv.CompositeInsert %98, %119[4 : i32] : vector<64xi32> into !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %121 = spirv.CompositeInsert %106, %120[5 : i32] : vector<64xi32> into !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %122 = spirv.CompositeInsert %99, %121[6 : i32] : vector<64xi32> into !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %123 = spirv.CompositeInsert %107, %122[7 : i32] : vector<64xi32> into !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %124 = spirv.CompositeExtract %67[0 : i32] : !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>
        %125 = spirv.CompositeExtract %67[1 : i32] : !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>
        // convert local SG ID to global SG ID
        %t0_126 = spirv.Load "Input" %__builtin_var_SubgroupId___addr : i32
        %t1_126 = spirv.IMul %wg_id_x, %cst32_i32 : i32
        %126 = spirv.IAdd %t0_126, %t1_126 : i32

        %127 = spirv.ConvertPtrToU %124 : !spirv.ptr<i16, CrossWorkgroup> to i64
        %128 = spirv.ConvertPtrToU %125 : !spirv.ptr<i16, CrossWorkgroup> to i32
        %129 = spirv.UDiv %126, %cst32_i32 : i32
        %130 = spirv.UMod %129, %cst16_i32 : i32
        %131 = spirv.IMul %130, %cst256_i32 : i32
        %132 = spirv.UMod %126, %cst32_i32 : i32
        %133 = spirv.UDiv %132, %cst4_i32 : i32
        %134 = spirv.UMod %132, %cst4_i32 : i32
        %135 = spirv.IMul %134, %cst64_i32 : i32
        %136 = spirv.UDiv %133, %cst4_i32 : i32
        %137 = spirv.UMod %133, %cst4_i32 : i32
        %cst50856579_i32 = spirv.Constant 50856579 : i32
        %138 = spirv.ShiftRightLogical %127, %cst32_i64 : i64, i64
        %139 = spirv.UConvert %138 : i64 to i32
        %140 = spirv.UConvert %127 : i64 to i32
        %141 = spirv.VectorInsertDynamic %140, %cst_vec_8xi32[%cst0_i32] : vector<8xi32>, i32
        %142 = spirv.VectorInsertDynamic %139, %141[%cst1_i32] : vector<8xi32>, i32
        %143 = spirv.VectorInsertDynamic %cst8191_i32, %142[%cst2_i32] : vector<8xi32>, i32
        %144 = spirv.VectorInsertDynamic %cst4095_i32, %143[%cst3_i32] : vector<8xi32>, i32
        %145 = spirv.VectorInsertDynamic %cst8191_i32, %144[%cst4_i32] : vector<8xi32>, i32
        %146 = spirv.VectorInsertDynamic %cst7951_i32, %145[%cst7_i32] : vector<8xi32>, i32
        %cst_vec_128xi64 = spirv.Constant dense<0> : vector<128xi64>
        %147 = spirv.IAdd %131, %135 : i32
        %148 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32] %146 : vector<8xi32>, %146 : vector<8xi32> -> vector<8xi32>
        %149 = spirv.VectorInsertDynamic %147, %148[%cst5_i32] : vector<8xi32>, i32
        %150 = spirv.VectorInsertDynamic %128, %149[%cst6_i32] : vector<8xi32>, i32

        //load matrix b 4x32x16, spilt into 4x2x16x16
        %151 = spirv.FunctionCall @llvm_genx_raw_send2_v128i64_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst16_i8, %cst15_i8, %cst0_i32, %cst50856579_i32, %150, %cst_vec_128xi64) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi64>) -> vector<128xi64>
        %152 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32] %151 : vector<128xi64>, %151 : vector<128xi64> -> vector<64xi64>
        %153 = spirv.VectorShuffle [64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %151 : vector<128xi64>, %151 : vector<128xi64> -> vector<64xi64>
        %154 = spirv.IAdd %135, %cst16_i32 : i32
        %155 = spirv.IAdd %131, %154 : i32
        %156 = spirv.VectorInsertDynamic %155, %148[%cst5_i32] : vector<8xi32>, i32
        %157 = spirv.VectorInsertDynamic %128, %156[%cst6_i32] : vector<8xi32>, i32
        %158 = spirv.FunctionCall @llvm_genx_raw_send2_v128i64_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst16_i8, %cst15_i8, %cst0_i32, %cst50856579_i32, %157, %cst_vec_128xi64) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi64>) -> vector<128xi64>
        %159 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32] %158 : vector<128xi64>, %158 : vector<128xi64> -> vector<64xi64>
        %160 = spirv.VectorShuffle [64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %158 : vector<128xi64>, %158 : vector<128xi64> -> vector<64xi64>
        %161 = spirv.IAdd %135, %cst32_i32 : i32
        %162 = spirv.IAdd %131, %161 : i32
        %163 = spirv.VectorInsertDynamic %162, %148[%cst5_i32] : vector<8xi32>, i32
        %164 = spirv.VectorInsertDynamic %128, %163[%cst6_i32] : vector<8xi32>, i32
        %165 = spirv.FunctionCall @llvm_genx_raw_send2_v128i64_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst16_i8, %cst15_i8, %cst0_i32, %cst50856579_i32, %164, %cst_vec_128xi64) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi64>) -> vector<128xi64>
        %166 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32] %165 : vector<128xi64>, %165 : vector<128xi64> -> vector<64xi64>
        %167 = spirv.VectorShuffle [64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %165 : vector<128xi64>, %165 : vector<128xi64> -> vector<64xi64>
        %cst48_i32 = spirv.Constant 48 : i32
        %168 = spirv.IAdd %135, %cst48_i32 : i32
        %169 = spirv.IAdd %131, %168 : i32
        %170 = spirv.VectorInsertDynamic %169, %148[%cst5_i32] : vector<8xi32>, i32
        %171 = spirv.VectorInsertDynamic %128, %170[%cst6_i32] : vector<8xi32>, i32
        %172 = spirv.FunctionCall @llvm_genx_raw_send2_v128i64_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst16_i8, %cst15_i8, %cst0_i32, %cst50856579_i32, %171, %cst_vec_128xi64) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi64>) -> vector<128xi64>
        %173 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32] %172 : vector<128xi64>, %172 : vector<128xi64> -> vector<64xi64>
        %174 = spirv.VectorShuffle [64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %172 : vector<128xi64>, %172 : vector<128xi64> -> vector<64xi64>
        %175 = spirv.IMul %137, %cst16_i32 : i32
        %176 = spirv.IAdd %135, %175 : i32
        %177 = spirv.IAdd %131, %176 : i32
        %178 = spirv.IMul %136, %cst16_i32 : i32
        %179 = spirv.IAdd %128, %178 : i32
        %180 = spirv.IAdd %179, %cst96_i32 : i32
        %cst34079363_i32 = spirv.Constant 34079363 : i32
        %181 = spirv.VectorInsertDynamic %177, %146[%cst5_i32] : vector<8xi32>, i32
        %182 = spirv.VectorInsertDynamic %180, %181[%cst6_i32] : vector<8xi32>, i32
        %cst3855_i32 = spirv.Constant 3855 : i32
        %183 = spirv.VectorInsertDynamic %cst3855_i32, %182[%cst7_i32] : vector<8xi32>, i32

        //prefetch b, each subgroup in M dim prefetch 16 x 16
        spirv.FunctionCall @llvm_genx_raw_send2_noresult_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst15_i8, %cst0_i32, %cst34079363_i32, %183) : (i8, i8, i1, i8, i8, i32, i32, vector<8xi32>) -> ()
        %cst-128_i8 = spirv.Constant -128 : i8
        spirv.FunctionCall @llvm_genx_fence(%cst-128_i8) : (i8) -> ()
        %184 = spirv.Undef : !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        %185 = spirv.CompositeInsert %152, %184[0 : i32] : vector<64xi64> into !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        %186 = spirv.CompositeInsert %159, %185[1 : i32] : vector<64xi64> into !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        %187 = spirv.CompositeInsert %166, %186[2 : i32] : vector<64xi64> into !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        %188 = spirv.CompositeInsert %173, %187[3 : i32] : vector<64xi64> into !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        %189 = spirv.CompositeInsert %153, %188[4 : i32] : vector<64xi64> into !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        %190 = spirv.CompositeInsert %160, %189[5 : i32] : vector<64xi64> into !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        %191 = spirv.CompositeInsert %167, %190[6 : i32] : vector<64xi64> into !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        %192 = spirv.CompositeInsert %174, %191[7 : i32] : vector<64xi64> into !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        // convert local SG ID to global SG ID
        %t0_193 = spirv.Load "Input" %__builtin_var_SubgroupId___addr : i32
        %t1_193 = spirv.IMul %wg_id_x, %cst32_i32 : i32
        %193 = spirv.IAdd %t0_193, %t1_193 : i32

        %cst9_i32 = spirv.Constant 9 : i32
        %194 = spirv.CompositeExtract %123[0 : i32] : !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %195 = spirv.CompositeExtract %123[1 : i32] : !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %196 = spirv.CompositeExtract %123[2 : i32] : !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %197 = spirv.CompositeExtract %123[3 : i32] : !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %198 = spirv.CompositeExtract %123[4 : i32] : !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %199 = spirv.CompositeExtract %123[5 : i32] : !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %200 = spirv.CompositeExtract %123[6 : i32] : !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %201 = spirv.CompositeExtract %123[7 : i32] : !spirv.struct<(vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>, vector<64xi32>)>
        %202 = spirv.CompositeExtract %192[0 : i32] : !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        %203 = spirv.CompositeExtract %192[1 : i32] : !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        %204 = spirv.CompositeExtract %192[2 : i32] : !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        %205 = spirv.CompositeExtract %192[3 : i32] : !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        %206 = spirv.CompositeExtract %192[4 : i32] : !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        %207 = spirv.CompositeExtract %192[5 : i32] : !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        %208 = spirv.CompositeExtract %192[6 : i32] : !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        %209 = spirv.CompositeExtract %192[7 : i32] : !spirv.struct<(vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>, vector<64xi64>)>
        %210 = spirv.CompositeExtract %65[0 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %211 = spirv.CompositeExtract %65[1 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %212 = spirv.CompositeExtract %65[2 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %213 = spirv.CompositeExtract %65[3 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %214 = spirv.CompositeExtract %65[4 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %215 = spirv.CompositeExtract %65[5 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %216 = spirv.CompositeExtract %65[6 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %217 = spirv.CompositeExtract %65[7 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %218 = spirv.CompositeExtract %65[8 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %219 = spirv.CompositeExtract %65[9 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %220 = spirv.CompositeExtract %65[10 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %221 = spirv.CompositeExtract %65[11 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %222 = spirv.CompositeExtract %65[12 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %223 = spirv.CompositeExtract %65[13 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %224 = spirv.CompositeExtract %65[14 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %225 = spirv.CompositeExtract %65[15 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %226 = spirv.Bitcast %202 : vector<64xi64> to vector<128xi32>
        %227 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%210, %226, %194, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %228 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%214, %226, %196, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %229 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%218, %226, %198, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %230 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%222, %226, %200, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %231 = spirv.Bitcast %206 : vector<64xi64> to vector<128xi32>
        %232 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%227, %231, %195, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %233 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%228, %231, %197, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %234 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%229, %231, %199, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %235 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%230, %231, %201, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %236 = spirv.Bitcast %203 : vector<64xi64> to vector<128xi32>
        %237 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%211, %236, %194, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %238 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%215, %236, %196, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %239 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%219, %236, %198, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %240 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%223, %236, %200, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %241 = spirv.Bitcast %207 : vector<64xi64> to vector<128xi32>
        %242 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%237, %241, %195, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %243 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%238, %241, %197, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %244 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%239, %241, %199, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %245 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%240, %241, %201, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %246 = spirv.Bitcast %204 : vector<64xi64> to vector<128xi32>
        %247 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%212, %246, %194, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %248 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%216, %246, %196, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %249 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%220, %246, %198, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %250 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%224, %246, %200, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %251 = spirv.Bitcast %208 : vector<64xi64> to vector<128xi32>
        %252 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%247, %251, %195, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %253 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%248, %251, %197, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %254 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%249, %251, %199, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %255 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%250, %251, %201, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %256 = spirv.Bitcast %205 : vector<64xi64> to vector<128xi32>
        %257 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%213, %256, %194, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %258 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%217, %256, %196, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %259 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%221, %256, %198, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %260 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%225, %256, %200, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %261 = spirv.Bitcast %209 : vector<64xi64> to vector<128xi32>
        %262 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%257, %261, %195, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %263 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%258, %261, %197, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %264 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%259, %261, %199, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %265 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(%260, %261, %201, %cst9_i32, %cst9_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>

        //named_barrier wait
        spirv.FunctionCall @llvm_genx_nbarrier(%cst0_i8, %cst1_i8, %cst0_i8) : (i8, i8, i8) -> ()
        %266 = spirv.CompositeInsert %232, %0[0 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %267 = spirv.CompositeInsert %242, %266[1 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %268 = spirv.CompositeInsert %252, %267[2 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %269 = spirv.CompositeInsert %262, %268[3 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %270 = spirv.CompositeInsert %233, %269[4 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %271 = spirv.CompositeInsert %243, %270[5 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %272 = spirv.CompositeInsert %253, %271[6 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %273 = spirv.CompositeInsert %263, %272[7 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %274 = spirv.CompositeInsert %234, %273[8 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %275 = spirv.CompositeInsert %244, %274[9 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %276 = spirv.CompositeInsert %254, %275[10 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %277 = spirv.CompositeInsert %264, %276[11 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %278 = spirv.CompositeInsert %235, %277[12 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %279 = spirv.CompositeInsert %245, %278[13 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %280 = spirv.CompositeInsert %255, %279[14 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %281 = spirv.CompositeInsert %265, %280[15 : i32] : vector<128xf32> into !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %282 = spirv.ConvertPtrToU %70 : !spirv.ptr<i16, CrossWorkgroup> to i32
        %283 = spirv.IAdd %282, %cst32_i32 : i32
        %284 = spirv.ConvertUToPtr %283 : i32 to !spirv.ptr<i16, CrossWorkgroup>
        %285 = spirv.CompositeInsert %69, %22[0 : i32] : !spirv.ptr<i16, CrossWorkgroup> into !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>
        %286 = spirv.CompositeInsert %284, %285[1 : i32] : !spirv.ptr<i16, CrossWorkgroup> into !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>
        %287 = spirv.ConvertPtrToU %125 : !spirv.ptr<i16, CrossWorkgroup> to i32
        %288 = spirv.IAdd %287, %cst32_i32 : i32
        %289 = spirv.ConvertUToPtr %288 : i32 to !spirv.ptr<i16, CrossWorkgroup>
        %290 = spirv.CompositeInsert %124, %22[0 : i32] : !spirv.ptr<i16, CrossWorkgroup> into !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>
        %291 = spirv.CompositeInsert %289, %290[1 : i32] : !spirv.ptr<i16, CrossWorkgroup> into !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>
        %292 = spirv.IAdd %64, %cst1_i32 : i32
        spirv.Branch ^bb1(%292, %281, %286, %291 : i32, !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>, !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>, !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>)
    ^bb3:  // pred: ^bb1
        %293 = spirv.CompositeExtract %65[0 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %294 = spirv.CompositeExtract %65[1 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %295 = spirv.CompositeExtract %65[2 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %296 = spirv.CompositeExtract %65[3 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %297 = spirv.CompositeExtract %65[4 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %298 = spirv.CompositeExtract %65[5 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %299 = spirv.CompositeExtract %65[6 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %300 = spirv.CompositeExtract %65[7 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %301 = spirv.CompositeExtract %65[8 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %302 = spirv.CompositeExtract %65[9 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %303 = spirv.CompositeExtract %65[10 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %304 = spirv.CompositeExtract %65[11 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %305 = spirv.CompositeExtract %65[12 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %306 = spirv.CompositeExtract %65[13 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %307 = spirv.CompositeExtract %65[14 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %308 = spirv.CompositeExtract %65[15 : i32] : !spirv.struct<(vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>, vector<128xf32>)>
        %309 = spirv.FunctionCall @llvm_genx_bf_cvt_v128f16_v128f32(%293) : (vector<128xf32>) -> vector<128xf16>
        %310 = spirv.FunctionCall @llvm_genx_bf_cvt_v128f16_v128f32(%294) : (vector<128xf32>) -> vector<128xf16>
        %311 = spirv.FunctionCall @llvm_genx_bf_cvt_v128f16_v128f32(%295) : (vector<128xf32>) -> vector<128xf16>
        %312 = spirv.FunctionCall @llvm_genx_bf_cvt_v128f16_v128f32(%296) : (vector<128xf32>) -> vector<128xf16>
        %313 = spirv.FunctionCall @llvm_genx_bf_cvt_v128f16_v128f32(%297) : (vector<128xf32>) -> vector<128xf16>
        %314 = spirv.FunctionCall @llvm_genx_bf_cvt_v128f16_v128f32(%298) : (vector<128xf32>) -> vector<128xf16>
        %315 = spirv.FunctionCall @llvm_genx_bf_cvt_v128f16_v128f32(%299) : (vector<128xf32>) -> vector<128xf16>
        %316 = spirv.FunctionCall @llvm_genx_bf_cvt_v128f16_v128f32(%300) : (vector<128xf32>) -> vector<128xf16>
        %317 = spirv.FunctionCall @llvm_genx_bf_cvt_v128f16_v128f32(%301) : (vector<128xf32>) -> vector<128xf16>
        %318 = spirv.FunctionCall @llvm_genx_bf_cvt_v128f16_v128f32(%302) : (vector<128xf32>) -> vector<128xf16>
        %319 = spirv.FunctionCall @llvm_genx_bf_cvt_v128f16_v128f32(%303) : (vector<128xf32>) -> vector<128xf16>
        %320 = spirv.FunctionCall @llvm_genx_bf_cvt_v128f16_v128f32(%304) : (vector<128xf32>) -> vector<128xf16>
        %321 = spirv.FunctionCall @llvm_genx_bf_cvt_v128f16_v128f32(%305) : (vector<128xf32>) -> vector<128xf16>
        %322 = spirv.FunctionCall @llvm_genx_bf_cvt_v128f16_v128f32(%306) : (vector<128xf32>) -> vector<128xf16>
        %323 = spirv.FunctionCall @llvm_genx_bf_cvt_v128f16_v128f32(%307) : (vector<128xf32>) -> vector<128xf16>
        %324 = spirv.FunctionCall @llvm_genx_bf_cvt_v128f16_v128f32(%308) : (vector<128xf32>) -> vector<128xf16>
        %325 = spirv.Undef : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %326 = spirv.CompositeInsert %309, %325[0 : i32] : vector<128xf16> into !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %327 = spirv.CompositeInsert %310, %326[1 : i32] : vector<128xf16> into !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %328 = spirv.CompositeInsert %311, %327[2 : i32] : vector<128xf16> into !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %329 = spirv.CompositeInsert %312, %328[3 : i32] : vector<128xf16> into !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %330 = spirv.CompositeInsert %313, %329[4 : i32] : vector<128xf16> into !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %331 = spirv.CompositeInsert %314, %330[5 : i32] : vector<128xf16> into !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %332 = spirv.CompositeInsert %315, %331[6 : i32] : vector<128xf16> into !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %333 = spirv.CompositeInsert %316, %332[7 : i32] : vector<128xf16> into !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %334 = spirv.CompositeInsert %317, %333[8 : i32] : vector<128xf16> into !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %335 = spirv.CompositeInsert %318, %334[9 : i32] : vector<128xf16> into !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %336 = spirv.CompositeInsert %319, %335[10 : i32] : vector<128xf16> into !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %337 = spirv.CompositeInsert %320, %336[11 : i32] : vector<128xf16> into !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %338 = spirv.CompositeInsert %321, %337[12 : i32] : vector<128xf16> into !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %339 = spirv.CompositeInsert %322, %338[13 : i32] : vector<128xf16> into !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %340 = spirv.CompositeInsert %323, %339[14 : i32] : vector<128xf16> into !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %341 = spirv.CompositeInsert %324, %340[15 : i32] : vector<128xf16> into !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %342 = spirv.IMul %arg8, %cst0_i32 : i32
        %343 = spirv.ConvertUToPtr %342 : i32 to !spirv.ptr<i16, CrossWorkgroup>
        %344 = spirv.CompositeInsert %arg2_, %22[0 : i32] : !spirv.ptr<i16, CrossWorkgroup> into !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>
        %345 = spirv.CompositeExtract %344[1 : i32] : !spirv.struct<(!spirv.ptr<i16, CrossWorkgroup>, !spirv.ptr<i16, CrossWorkgroup>)>
        %346 = spirv.ConvertPtrToU %345 : !spirv.ptr<i16, CrossWorkgroup> to i32
        %347 = spirv.IAdd %346, %cst0_i32 : i32
        %348 = spirv.ConvertUToPtr %347 : i32 to !spirv.ptr<i16, CrossWorkgroup>
        %350 = spirv.Undef : !spirv.struct<(i1)>
        %352 = spirv.ConvertPtrToU %arg2_ : !spirv.ptr<i16, CrossWorkgroup> to i64
        // convert local SG ID to global SG ID
        %t0_353 = spirv.Load "Input" %__builtin_var_SubgroupId___addr : i32
        %t1_353 = spirv.IMul %wg_id_x, %cst32_i32 : i32
        %353 = spirv.IAdd %t0_353, %t1_353 : i32

        %354 = spirv.UDiv %353, %cst32_i32 : i32
        %355 = spirv.UDiv %354, %cst16_i32 : i32
        %356 = spirv.UMod %354, %cst16_i32 : i32
        %357 = spirv.IMul %356, %cst256_i32 : i32
        %358 = spirv.IMul %355, %cst256_i32 : i32
        %359 = spirv.UMod %353, %cst32_i32 : i32
        %360 = spirv.UDiv %359, %cst4_i32 : i32
        %361 = spirv.UMod %359, %cst4_i32 : i32
        %362 = spirv.IMul %361, %cst64_i32 : i32
        %363 = spirv.IMul %360, %cst32_i32 : i32
        %364 = spirv.CompositeExtract %341[0 : i32] : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %365 = spirv.CompositeExtract %341[1 : i32] : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %366 = spirv.CompositeExtract %341[2 : i32] : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %367 = spirv.CompositeExtract %341[3 : i32] : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %368 = spirv.CompositeExtract %341[4 : i32] : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %369 = spirv.CompositeExtract %341[5 : i32] : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %370 = spirv.CompositeExtract %341[6 : i32] : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %371 = spirv.CompositeExtract %341[7 : i32] : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %372 = spirv.CompositeExtract %341[8 : i32] : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %373 = spirv.CompositeExtract %341[9 : i32] : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %374 = spirv.CompositeExtract %341[10 : i32] : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %375 = spirv.CompositeExtract %341[11 : i32] : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %376 = spirv.CompositeExtract %341[12 : i32] : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %377 = spirv.CompositeExtract %341[13 : i32] : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %378 = spirv.CompositeExtract %341[14 : i32] : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %379 = spirv.CompositeExtract %341[15 : i32] : !spirv.struct<(vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>, vector<128xf16>)>
        %cst4_i8 = spirv.Constant 4 : i8
        %cst15_i8_0 = spirv.Constant 15 : i8
        %cst34472455_i32 = spirv.Constant 34472455 : i32
        %cst_vec_8xi32_1 = spirv.Constant dense<0> : vector<8xi32>
        %cst32_i64_2 = spirv.Constant 32 : i64
        %380 = spirv.ShiftRightLogical %352, %cst32_i64_2 : i64, i64
        %381 = spirv.UConvert %380 : i64 to i32
        %382 = spirv.UConvert %352 : i64 to i32
        %383 = spirv.VectorInsertDynamic %382, %cst_vec_8xi32_1[%cst0_i32] : vector<8xi32>, i32
        %384 = spirv.VectorInsertDynamic %381, %383[%cst1_i32] : vector<8xi32>, i32
        %cst2_i32_3 = spirv.Constant 2 : i32
        %385 = spirv.VectorInsertDynamic %cst8191_i32, %384[%cst2_i32_3] : vector<8xi32>, i32
        %cst3_i32_4 = spirv.Constant 3 : i32
        %386 = spirv.VectorInsertDynamic %cst4095_i32, %385[%cst3_i32_4] : vector<8xi32>, i32
        %387 = spirv.VectorInsertDynamic %cst8191_i32, %386[%cst4_i32] : vector<8xi32>, i32
        %cst7_i32_5 = spirv.Constant 7 : i32
        %cst1807_i32 = spirv.Constant 1807 : i32
        %388 = spirv.VectorInsertDynamic %cst1807_i32, %387[%cst7_i32_5] : vector<8xi32>, i32
        %389 = spirv.IMul %cst0_i32, %cst16_i32 : i32
        %390 = spirv.IAdd %362, %389 : i32
        %391 = spirv.IAdd %357, %390 : i32
        %392 = spirv.IMul %cst0_i32, %cst8_i32 : i32
        %393 = spirv.IAdd %363, %392 : i32
        %394 = spirv.IAdd %358, %393 : i32
        %395 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32] %388 : vector<8xi32>, %388 : vector<8xi32> -> vector<8xi32>
        %cst5_i32_6 = spirv.Constant 5 : i32
        %396 = spirv.VectorInsertDynamic %391, %395[%cst5_i32_6] : vector<8xi32>, i32
        %cst6_i32_7 = spirv.Constant 6 : i32
        %397 = spirv.VectorInsertDynamic %394, %396[%cst6_i32_7] : vector<8xi32>, i32
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8_0, %cst0_i32, %cst34472455_i32, %397, %364) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
        %398 = spirv.IMul %cst1_i32, %cst16_i32 : i32
        %399 = spirv.IAdd %362, %398 : i32
        %400 = spirv.IAdd %357, %399 : i32
        %401 = spirv.VectorInsertDynamic %400, %395[%cst5_i32_6] : vector<8xi32>, i32
        %402 = spirv.VectorInsertDynamic %394, %401[%cst6_i32_7] : vector<8xi32>, i32
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8_0, %cst0_i32, %cst34472455_i32, %402, %365) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
        %403 = spirv.IMul %cst2_i32_3, %cst16_i32 : i32
        %404 = spirv.IAdd %362, %403 : i32
        %405 = spirv.IAdd %357, %404 : i32
        %406 = spirv.VectorInsertDynamic %405, %395[%cst5_i32_6] : vector<8xi32>, i32
        %407 = spirv.VectorInsertDynamic %394, %406[%cst6_i32_7] : vector<8xi32>, i32
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8_0, %cst0_i32, %cst34472455_i32, %407, %366) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
        %408 = spirv.IMul %cst3_i32_4, %cst16_i32 : i32
        %409 = spirv.IAdd %362, %408 : i32
        %410 = spirv.IAdd %357, %409 : i32
        %411 = spirv.VectorInsertDynamic %410, %395[%cst5_i32_6] : vector<8xi32>, i32
        %412 = spirv.VectorInsertDynamic %394, %411[%cst6_i32_7] : vector<8xi32>, i32
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8_0, %cst0_i32, %cst34472455_i32, %412, %367) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
        %413 = spirv.IMul %cst1_i32, %cst8_i32 : i32
        %414 = spirv.IAdd %363, %413 : i32
        %415 = spirv.IAdd %358, %414 : i32
        %416 = spirv.VectorInsertDynamic %415, %396[%cst6_i32_7] : vector<8xi32>, i32
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8_0, %cst0_i32, %cst34472455_i32, %416, %368) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
        %417 = spirv.VectorInsertDynamic %415, %401[%cst6_i32_7] : vector<8xi32>, i32
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8_0, %cst0_i32, %cst34472455_i32, %417, %369) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
        %418 = spirv.VectorInsertDynamic %415, %406[%cst6_i32_7] : vector<8xi32>, i32
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8_0, %cst0_i32, %cst34472455_i32, %418, %370) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
        %419 = spirv.VectorInsertDynamic %415, %411[%cst6_i32_7] : vector<8xi32>, i32
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8_0, %cst0_i32, %cst34472455_i32, %419, %371) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
        %420 = spirv.IMul %cst2_i32_3, %cst8_i32 : i32
        %421 = spirv.IAdd %363, %420 : i32
        %422 = spirv.IAdd %358, %421 : i32
        %423 = spirv.VectorInsertDynamic %422, %396[%cst6_i32_7] : vector<8xi32>, i32
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8_0, %cst0_i32, %cst34472455_i32, %423, %372) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
        %424 = spirv.VectorInsertDynamic %422, %401[%cst6_i32_7] : vector<8xi32>, i32
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8_0, %cst0_i32, %cst34472455_i32, %424, %373) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
        %425 = spirv.VectorInsertDynamic %422, %406[%cst6_i32_7] : vector<8xi32>, i32
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8_0, %cst0_i32, %cst34472455_i32, %425, %374) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
        %426 = spirv.VectorInsertDynamic %422, %411[%cst6_i32_7] : vector<8xi32>, i32
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8_0, %cst0_i32, %cst34472455_i32, %426, %375) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
        %427 = spirv.IMul %cst3_i32_4, %cst8_i32 : i32
        %428 = spirv.IAdd %363, %427 : i32
        %429 = spirv.IAdd %358, %428 : i32
        %430 = spirv.VectorInsertDynamic %429, %396[%cst6_i32_7] : vector<8xi32>, i32
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8_0, %cst0_i32, %cst34472455_i32, %430, %376) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
        %431 = spirv.VectorInsertDynamic %429, %401[%cst6_i32_7] : vector<8xi32>, i32
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8_0, %cst0_i32, %cst34472455_i32, %431, %377) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
        %432 = spirv.VectorInsertDynamic %429, %406[%cst6_i32_7] : vector<8xi32>, i32
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8_0, %cst0_i32, %cst34472455_i32, %432, %378) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
        %433 = spirv.VectorInsertDynamic %429, %411[%cst6_i32_7] : vector<8xi32>, i32
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8_0, %cst0_i32, %cst34472455_i32, %433, %379) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
        spirv.Return
    }

    spirv.EntryPoint "Kernel" @gemm4k_kernel
    spirv.ExecutionMode @gemm4k_kernel "ContractionOff"
    spirv.ExecutionMode @gemm4k_kernel "SharedLocalMemorySizeINTEL", 0
    spirv.ExecutionMode @gemm4k_kernel "NamedBarrierCountINTEL", 16

    spirv.func @llvm_genx_address_convert_i64_p1i16(!spirv.ptr<i16, CrossWorkgroup>) -> i64 "None" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.address.convert.i64.p1i16",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL
    }

    spirv.func @llvm_genx_lsc_prefetch2d_stateless_v64i64_i1_i64(i1, i8, i8, i8, i8, i8, i32, i32, i8, i64, i32, i32, i32, i32, i32) -> vector<64xi64> "None" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.lsc.prefetch2d.stateless.v64i64.i1.i64",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL
    }

    spirv.func @llvm_genx_lsc_prefetch2d_stateless_v64i32_i1_i64(i1, i8, i8, i8, i8, i8, i32, i32, i8, i64, i32, i32, i32, i32, i32) -> vector<64xi32> "None" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.lsc.prefetch2d.stateless.v64i32.i1.i64",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL
    }

    spirv.func @llvm_genx_raw_send2_v256i32_i1_v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<256xi32>) -> vector<256xi32> "None" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.raw.send2.v256i32.i1.v8i32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL
    }

    spirv.func @llvm_genx_raw_send2_noresult_i1_v8i32(i8, i8, i1, i8, i8, i32, i32, vector<8xi32>) "None" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.raw.send2.noresult.i1.v8i32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL
    }

    spirv.func @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f16(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) "None" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.raw.sends2.noresult.i1.v8i32.v128f16",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL
    }

    spirv.func @llvm_genx_raw_send2_v128i64_i1_v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi64>) -> vector<128xi64> "None" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.raw.send2.v128i64.i1.v8i32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL
    }

    spirv.func @llvm_genx_dpas2_v128f32_v128f32_v128i32_v64i32(vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32> "None" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.dpas2.v128f32.v128f32.v128i32.v64i32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL
    }

    spirv.func @llvm_genx_lsc_store2d_stateless_i1_i64_v128f16(i1, i8, i8, i8, i8, i8, i32, i32, i8, i64, i32, i32, i32, i32, i32, vector<128xf16>) "None" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.lsc.store2d.stateless.i1.i64.v128f16",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL
    }

    spirv.func @llvm_genx_bf_cvt_v128f16_v128f32(vector<128xf32>) -> vector<128xf16> "None" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.bf.cvt.v128f16.v128f32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL
    }

    spirv.func @llvm_genx_nbarrier(i8, i8, i8) "None" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.nbarrier",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL
    }

    spirv.func @llvm_genx_fence(i8) "None" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.fence",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL
    }
  }

  // GPU module, almost same as the SPIR-V module but without 'spirv' dialect specific properties
  gpu.module @dpas_module {
    gpu.func @gemm4k_kernel(%arg0: memref<?xi16>, %arg1: memref<?xi16>, %arg2: memref<?xi16>, %arg3: i32, %arg4: i32, %arg5: i32 , %arg6: i32, %arg7: i32, %arg8: i32) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      gpu.return
    }
  }

  func.func @gemm4k_test(%arg_M: index, %arg_N: index, %arg_K: index){
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 2.000000e+00 : f32
    %c_gen_int = arith.constant 0 : i1
    %cf_lower = arith.constant -0.5 : f32
    %cf_upper = arith.constant 0.5 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c4096 = arith.constant 4096 : index

    %c4096_i32 = arith.constant 4096 : i32

    %int_0 = arith.constant 0 : i32
    %int_1 = arith.constant 1 : i32

    // Allocate vectors to be passed to function

    // Setting up Vector C
    %C_size = arith.muli %arg_M, %arg_N : index
    %C_size_i8 = arith.muli %C_size, %c2 : index

    %memref_C_i8 = gpu.alloc host_shared (%C_size_i8) : memref<?xi8>
    %memref_C_bf16 = memref.view %memref_C_i8[%c0][%C_size] : memref<?xi8> to memref<?xbf16>
    %memref_C = memref.view %memref_C_i8[%c0][%C_size] : memref<?xi8> to memref<?xi16>

    // allocate cpu reference
    %memref_C_f32_ref = gpu.alloc host_shared (%C_size) : memref<?xf32>

    // Initialize C to 0
    call @fillResource1DBF16(%memref_C_bf16, %cst_0) : (memref<*xbf16>, f32) -> ()
    call @fillResource1DF32(%memref_C_f32_ref, %cst_0) : (memref<?xf32>, f32) -> ()

    // Setting up the Vector B & A
    // B is setup slightly differently than other vectors, since B is
    // expected to be bf16 by the dpas instruction, but can not be passed
    // in SPIR-V (SPIR-V does not support bf16), we first allocate B
    // as i16 and then change the type (create view) to bf16. We use the bf16
    // view to initialize the vectors. We finally pass the i16 pointer to the
    // kernel and load bf16 from that using the intel vc-intrinsic

    // Alternative ways:, we could also create a i8 view and pass that.
    // This way, both views point to the same vector, but accessed
    // differently based what view is used
    // Since, in our case, the vector is essentially bf16, but needed to
    // have a view of i16 just be passed in SPIR-V and inside DPAS
    // reinterpreted back bf16, we can safely use this approach
    //            / bf16 (initialization)         \
    // B = i8 -                                   ->
    //            \ i16 (passed to SPIR-V kernel) /

    %B_size = arith.muli %arg_K, %arg_N : index

    // Since, we are allocating bf16 as i8, %B_size * 2 is used
    // for allocation size
    %B_size_i8 =  arith.muli %B_size, %c2 : index

    %memref_B = gpu.alloc  host_shared (%B_size_i8) : memref<?xi8>

    // Create a view of bf16 vector
    %memref_B_i16 = memref.view %memref_B[%c0][%B_size] : memref<?xi8> to memref<?xi16>
    %memref_B_bf16 = memref.view %memref_B[%c0][%B_size] : memref<?xi8> to memref<?xbf16>

    // Initialize B matrix to 2.0
    // call @fillResource1DBF16(%memref_B_bf16, %cst_2) : (memref<*xbf16>, f32) -> ()
    // Or initialize B matrix to random values in (-0.5 , 0.5)
    call @fillResource1DRandomBF16(%memref_B_bf16, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xbf16>, f32, f32, i1) -> ()

    // Setting up the Vector A
    %A_size = arith.muli %arg_M, %arg_K : index

    // Since, we are allocating bf16 as i8, %A_size * 2 is used
    // for allocation size
    %A_size_i8 =  arith.muli %A_size, %c2 : index

    %memref_A = gpu.alloc  host_shared (%A_size_i8) : memref<?xi8>
    // Create a view of bf16 vector
    %memref_A_i16 = memref.view %memref_A[%c0][%A_size] : memref<? x i8> to memref<? x i16>
    %memref_A_bf16 = memref.view %memref_A[%c0][%A_size] : memref<? x i8> to memref<? x bf16>

    // SPIR-V type does not support bf16, hence passing vector 1, and vector 2 as i16, will load bf16 from this vector using the intel vc-intrinsic

    // Initialize A to 1.0
    // call @fillResource1DBF16(%memref_A_bf16, %cst_1) : (memref<*xbf16>, f32) -> ()
    // Or initialize A to random values in (-0.5, 0.5)
    call @fillResource1DRandomBF16(%memref_A_bf16, %cf_lower, %cf_upper, %c_gen_int) : (memref<*xbf16>, f32, f32, i1) -> ()

    // Calling the GPU version, using bf16 view of B and A vector
    call @gemm4k_gpu(%arg_M, %arg_N, %arg_K, %memref_C, %memref_B_i16, %memref_A_i16) : (index, index, index, memref<?xi16>, memref<?xi16>, memref<?xi16>) -> ()


    // Compute the CPU reference (takes minutes)
    scf.for %i = %c0 to %c4096 step %c1 {
      scf.for %j = %c0 to %c4096  step %c1 {
        %c_idx = arith.muli %i, %c4096 : index
        %c_idx_1 = arith.addi %c_idx, %j : index
        %acc = memref.load %memref_C_f32_ref[%c_idx_1] : memref<?xf32>
        %res = scf.for %k = %c0 to %c4096 step %c1 iter_args(%arg3 = %acc) -> f32  {
          %a_idx = arith.muli %i, %c4096 : index
          %a_idx_1 = arith.addi %a_idx, %k : index
          %b_idx = arith.muli %k, %c4096 : index
          %b_idx_1 = arith.addi %b_idx, %j : index
          %a = memref.load %memref_A_bf16[%a_idx_1] : memref<?xbf16>
          %b = memref.load %memref_B_bf16[%b_idx_1] : memref<?xbf16>
          %a_f32 = arith.extf %a : bf16 to f32
          %b_f32 = arith.extf %b : bf16 to f32
          %cc_ = arith.mulf %a_f32, %b_f32 : f32
          %ccc = arith.addf %cc_, %arg3 : f32
          scf.yield %ccc : f32
        }
        memref.store %res, %memref_C_f32_ref[%c_idx_1] : memref<?xf32>
      }
    }

    // Print allClose for GPU and CPU values
    %result = memref.cast %memref_C_bf16 : memref<?xbf16> to memref<*xbf16>
    %result_ref = memref.cast %memref_C_f32_ref : memref<?xf32> to memref<*xf32>
    %A = memref.cast %memref_A_bf16 : memref<?xbf16> to memref<*xbf16>
    %B = memref.cast %memref_B_bf16 : memref<?xbf16> to memref<*xbf16>
    // call @printMemrefBF16(%A) : (memref<*xbf16>) -> ()
    // call @printMemrefBF16(%B) : (memref<*xbf16>) -> ()
    // call @printMemrefBF16(%result) : (memref<*xbf16>) -> ()
    // call @printMemrefF32(%result_ref) : (memref<*xf32>) -> ()
    call @printAllcloseBF16(%result, %result_ref): (memref<*xbf16>, memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]

    return
  }

  // main function
  func.func @main() {
    %cst_M = arith.constant 4096 : index
    %cst_N = arith.constant 4096 : index
    %cst_K = arith.constant 4096 : index

    call @gemm4k_test(%cst_M, %cst_N, %cst_K) : (index, index, index) -> ()
    return
  }

  // Helper functions
  func.func private @fillResource1DBF16(memref<*xbf16>, f32) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF16(memref<*xf16>, f32) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
  func.func private @printMemrefBF16(memref<*xbf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseBF16(memref<*xbf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DRandomF16(memref<*xf16>, f32, f32, i1) attributes {llvm.emit_c_interface}
}
