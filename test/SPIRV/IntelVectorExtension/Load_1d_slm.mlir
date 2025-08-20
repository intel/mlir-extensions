// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%mlir_runner_utils,%irunner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%mlir_runner_utils,%irunner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck


module attributes {gpu.container_module}  {

  // function to setup the launch and launch the kernel
  func.func @lsc_load_1d_slm_gpu(%arg_A : memref<256xi8>, %arg_B : memref<256xi8>) {
    %c1 = arith.constant 1 : index

    // 1 workgroup and, 1 thread per workgroup
    gpu.launch_func @lsc_load_1d_slm_module::@lsc_load_1d_slm_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg_A : memref<256xi8>, %arg_B : memref<256xi8>)
    return
  }

  spirv.module @__spv__lsc_load_1d_slm_module Physical64 OpenCL requires #spirv.vce<v1.0, [Int8, Int16, Int64, Float16, Kernel, Addresses, Linkage, Vector16, VectorAnyINTEL, Float16Buffer, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, FunctionFloatControlINTEL], [SPV_INTEL_float_controls2, SPV_INTEL_vector_compute]> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorAnyINTEL, VectorComputeINTEL, FunctionFloatControlINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {

    spirv.func @lsc_load_1d_slm_kernel(%arg1: !spirv.ptr<i8, CrossWorkgroup>, %arg2: !spirv.ptr<i8, CrossWorkgroup>)  "DontInline"  attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>, workgroup_attributions = 0 : i64, VectorComputeFunctionINTEL}  {
        %uchar_0 = spirv.Constant 0 : i8
        %ushort_1 = spirv.Constant 1 : i16
        %uint_0 = spirv.Constant 0 : i32
        %uchar_3 = spirv.Constant 3 : i8
        %uchar_8 = spirv.Constant 8 : i8
        %uchar_2 = spirv.Constant 2 : i8
        %uchar_4 = spirv.Constant 4 : i8
        %uchar_7 = spirv.Constant 7 : i8
        %uint_9 =  spirv.Constant 9 :  i32
        %uint_8 =  spirv.Constant 8 :  i32
        %uint_4 =  spirv.Constant 4 :  i32
        %true = spirv.Constant true

        // Cast the uchar pointers (i8 ptr) to ulongs (i64)
        %arg_1 = spirv.FunctionCall @llvm_genx_address_convert_i64_p1i8(%arg1) : (!spirv.ptr<i8, CrossWorkgroup>) -> i64
        %arg_2 = spirv.FunctionCall @llvm_genx_address_convert_i64_p1i8(%arg2) : (!spirv.ptr<i8, CrossWorkgroup>) -> i64

        // Load A from global
        %load_from_global =  spirv.FunctionCall @llvm_genx_lsc_load_stateless_v64f32_i1_i64(%true, %uchar_0, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_8, %uchar_2, %uchar_0, %arg_1, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, i32) -> vector<64 x f32>

        // store A in (slm)
        spirv.FunctionCall @llvm_genx_lsc_store_slm_i1_i64_v64f32(%true, %uchar_4, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_8, %uchar_2, %uchar_0, %uint_0, %load_from_global, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i32, vector<64 x f32>, i32) -> () // -> mlir::NoneType
        // For SLM we need to only pass offset. We dont need to pass pointer. Need to double check.

        // Load A from slm
        %load_from_slm =  spirv.FunctionCall @llvm_genx_lsc_load_slm_v64f32_i1_i64(%true, %uchar_0, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_8, %uchar_2, %uchar_0, %uint_0, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i32, i32) -> vector<64xf32>

       // store A in B (global)
        spirv.FunctionCall @llvm_genx_lsc_store_stateless_i1_i64_v64f32(%true, %uchar_4, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_8, %uchar_2, %uchar_0, %arg_2, %load_from_slm, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, vector<64 x f32>, i32) -> () // -> mlir::NoneType
        spirv.Return
    }

    spirv.EntryPoint "Kernel" @lsc_load_1d_slm_kernel
    spirv.ExecutionMode @lsc_load_1d_slm_kernel "ContractionOff"
    spirv.ExecutionMode @lsc_load_1d_slm_kernel "SharedLocalMemorySizeINTEL", 2048
    // Utility function declarations (Intel vc-intrinsics)

    spirv.func @llvm_genx_address_convert_i64_p1i8(%arg: !spirv.ptr<i8, CrossWorkgroup>) -> i64 "Pure" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.address.convert.i64.p1i8",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_lsc_load_slm_v64f32_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i32, %arg11 : i32) -> vector<64 x f32> "Const" attributes{
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.lsc.load.slm.v64f32.i1.i64",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_lsc_load_stateless_v64f32_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i64, %arg11 : i32) -> vector<64 x f32> "Const" attributes{
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.lsc.load.stateless.v64f32.i1.i64",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_lsc_store_slm_i1_i64_v64f32(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i32, %arg11 : vector<64 x f32>, %arg12 : i32)  "None" attributes{
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.lsc.store.slm.i1.i64.v64f32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_lsc_store_stateless_i1_i64_v64f32(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i64, %arg11 : vector<64 x f32>, %arg12 : i32)  "None" attributes{
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.lsc.store.stateless.i1.i64.v64f32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}
  }

  // GPU module, almost same as the SPIR-V module but without 'spirv' dialect specific properties
  gpu.module @lsc_load_1d_slm_module {
    gpu.func @lsc_load_1d_slm_kernel(%arg1: memref<256xi8>, %arg2: memref<256xi8>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      gpu.return
    }
  }

  func.func @lsc_load_1d_slm_test(%arg_sys_dpth: index, %arg_rpt_cnt: index, %arg_N: index){
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.100000e+00 : f32
    %cst_2 = arith.constant 2.200000e+00 : f32

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %memref_A = gpu.alloc host_shared () :memref<256xi8>
    %memref_A_i8 = memref.view %memref_A[%c0][] : memref<256xi8> to memref<64xf32>
    %memref_A_i8_1D = memref.cast %memref_A_i8 : memref<64xf32> to memref<?xf32>
    // Initialize it to 2.2
    call @fillResource1DF32(%memref_A_i8_1D, %cst_2) : (memref<?xf32>, f32) -> ()

    %memref_B_i8 = gpu.alloc host_shared () : memref<256xi8>
    %memref_B = memref.view %memref_B_i8[%c0][] : memref<256xi8> to memref<64xf32>
    %memref_B_1D = memref.cast %memref_B : memref<64xf32> to memref<?xf32>
    call @fillResource1DF32(%memref_B_1D, %cst_1) : (memref<?xf32>, f32) -> ()

    call @lsc_load_1d_slm_gpu(%memref_A, %memref_B_i8) : (memref<256xi8>, memref<256xi8>) -> ()

    // Print the result
    %result = memref.cast %memref_B : memref<64xf32> to memref<*xf32>
    call @printMemrefF32(%result) : (memref<*xf32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-COUNT-64: 2.2
    return
  }

  // main function
  func.func @main() {
    %cst_sys_dpth = arith.constant 8 : index
    %cst_rpt_cnt = arith.constant 4 : index
    %cst_N = arith.constant 16 : index

    call @lsc_load_1d_slm_test(%cst_sys_dpth, %cst_rpt_cnt, %cst_N) : (index, index, index) -> ()
    return
  }

  // Helper functions
  func.func private @fillResource1DF16(memref<*xf16>, f32) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}

}
