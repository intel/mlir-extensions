// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%mlir_runner_utils,%irunner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%mlir_runner_utils,%irunner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

/// A simple Matrix Multiplication using DPAS instruction
/// A and B are in bf16, while the result C is f32
/// C[4x16] =  A[4x16] x B[16x16]


module attributes {gpu.container_module}  {

  // function to setup the launch and launch the kernel
  // args: size_t systolic_depth, size_t repeat_cnt, size_t N
  func.func @dpas_gpu(%arg_sys_dpth: index, %arg_rpt_cnt: index, %arg_N: index, %arg_C : memref<256xi8>, %arg_B : memref<512xi8>, %arg_A : memref<128xi8>) {
    %c1 = arith.constant 1 : index

    // Since we are using only one DPAS instruction we are launching:
    // 1 workgroup and, 1 thread per workgroup
    gpu.launch_func @dpas_module::@dpas_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg_C : memref<256xi8>, %arg_B : memref<512xi8>, %arg_A : memref<128xi8>)
    return
  }

  // SPIR-V DPAS module, it holds the DPAS kernel
  spirv.module @__spv__dpas_module Physical64 OpenCL requires #spirv.vce<v1.0, [Int8, Int16, Int64, Float16, Kernel, Addresses, Linkage, Vector16, VectorAnyINTEL, Float16Buffer, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, FunctionFloatControlINTEL], [SPV_INTEL_float_controls2, SPV_INTEL_vector_compute]> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorAnyINTEL, VectorComputeINTEL, FunctionFloatControlINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    // DPAS kernel
    spirv.func @dpas_kernel(%arg0: !spirv.ptr<i8, CrossWorkgroup>, %arg1: !spirv.ptr<i8, CrossWorkgroup>, %arg2: !spirv.ptr<i8, CrossWorkgroup>)  "DontInline"  attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>, workgroup_attributions = 0 : i64, VectorComputeFunctionINTEL}  {
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
        %arg_0 = spirv.FunctionCall @llvm_genx_address_convert_i64_p1i8(%arg0) : (!spirv.ptr<i8, CrossWorkgroup>) -> i64
        %arg_1 = spirv.FunctionCall @llvm_genx_address_convert_i64_p1i8(%arg1) : (!spirv.ptr<i8, CrossWorkgroup>) -> i64
        %arg_2 = spirv.FunctionCall @llvm_genx_address_convert_i64_p1i8(%arg2) : (!spirv.ptr<i8, CrossWorkgroup>) -> i64

        // Load vector C using load stateless
        // Signless version
        %C =  spirv.FunctionCall @llvm_genx_lsc_load_stateless_v64f32_i1_i64(%true, %uchar_0, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_8, %uchar_2, %uchar_0, %arg_0, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, i32) -> vector<64xf32>
        // Load vector 1 using load stateless
        %B =  spirv.FunctionCall @llvm_genx_lsc_load_stateless_v64i64_i1_i64(%true, %uchar_0, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_4, %uchar_8, %uchar_2, %uchar_0, %arg_1, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, i32) -> vector<64 x i64>
        // Cast the vector B as i32 from i64
        %B_uint_cast = spirv.Bitcast %B : vector<64 x i64> to vector<128 x i32>
        // Load vector 2 using load stateless
        %A =  spirv.FunctionCall @llvm_genx_lsc_load_stateless_v32i32_i1_i64(%true, %uchar_0, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_7, %uchar_2, %uchar_0, %arg_2, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, i32) -> vector<32 x i32>
        // Call dpas2
        %dpas_result =  spirv.FunctionCall @llvm_genx_dpas2_v64f32_v64f32_v128i32_v32i32(%C, %B_uint_cast, %A, %uint_9, %uint_9, %uint_8, %uint_4, %uint_0, %uint_0): (vector<64 x f32>, vector<128 x i32>, vector<32 x i32>, i32, i32, i32, i32, i32, i32) -> vector<64 x f32>
        // Store the result
        spirv.FunctionCall @llvm_genx_lsc_store_stateless_i1_i64_v64f32(%true, %uchar_4, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_8, %uchar_2, %uchar_0, %arg_0, %dpas_result, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, vector<64 x f32>, i32) -> () // -> mlir::NoneType
        spirv.Return
    }

    spirv.EntryPoint "Kernel" @dpas_kernel
    spirv.ExecutionMode @dpas_kernel "ContractionOff"
    spirv.ExecutionMode @dpas_kernel "SharedLocalMemorySizeINTEL", 0
    // Utility function declarations (Intel vc-intrinsics)
    spirv.func @llvm_genx_address_convert_i64_p1i8(%arg: !spirv.ptr<i8, CrossWorkgroup>) -> i64 "Pure" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.address.convert.i64.p1i8",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_lsc_load_stateless_v64f32_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i64, %arg11 : i32) -> vector<64 x f32> "Const" attributes{
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.lsc.load.stateless.v64f32.i1.i64",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_lsc_load_stateless_v64i64_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i64, %arg11 : i32) -> vector<64 x i64> "Const" attributes{
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.lsc.load.stateless.v64i64.i1.i64",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_lsc_load_stateless_v32i32_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i64, %arg11 : i32) -> vector<32 x i32> "Const" attributes{
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.lsc.load.stateless.v32i32.i1.i64",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_dpas2_v64f32_v64f32_v128i32_v32i32(%arg0 : vector<64 x f32>, %arg1 : vector<128 x i32>, %arg2 : vector<32 x i32>, %arg3 : i32, %arg4 : i32, %arg5 : i32, %arg6 : i32, %arg7 : i32, %arg8 : i32) -> vector<64 x f32> "Pure" attributes{
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.dpas2.v64f32.v64f32.v128i32.v32i32",
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
  gpu.module @dpas_module {
    gpu.func @dpas_kernel(%arg0: memref<256xi8>, %arg1: memref<512xi8>, %arg2: memref<128xi8>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      gpu.return
    }
  }

  func.func @dpas_ref(%arg_sys_dpth: index, %arg_rpt_cnt: index, %arg_N: index, %arg_C : memref<?xf32>, %arg_B : memref<?xbf16>, %arg_A : memref<?xbf16>){
    return
  }

  func.func @dpas_test(%arg_sys_dpth: index, %arg_rpt_cnt: index, %arg_N: index){
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.100000e+00 : f32
    %cst_2 = arith.constant 2.200000e+00 : f32

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index

    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index

    // Allocate vectors to be passed to function
    // Setting up Vector C
    // @INFO: Keeping some information as comment, so that it is
    // easy to understand how rpt_cnt, sys_dpth, and N are getting
    // used.

    // %C_size = arith.muli %arg_rpt_cnt, %arg_N : index
    // C_size_i8 = 4*16*4 = 256
    // %C_size_i8 = arith.muli %C_size, %c4 : index

    %memref_C_i8 = gpu.alloc host_shared () : memref<256xi8>
    %memref_C = memref.view %memref_C_i8[%c0][] : memref<256xi8> to memref<64xf32>
    // Initialize C to 0
    %memref_C_1D = memref.cast %memref_C : memref<64xf32> to memref<?xf32>
    call @fillResource1DF32(%memref_C_1D, %cst_0) : (memref<?xf32>, f32) -> ()

    // Setting up the Vector B & A
    // B is setup slightly differently than other vectors, since B is
    // expected to be bf16 by the dpas instruction, but can not be passed
    // in SPIR-V (SPIR-V does not support bf16), we first allocate B
    // as i8 and then change the type (create view) to bf16. We use the bf16
    // view to initialize the vectors. We finally pass the i8 pointer to the
    // kernel and load bf16 from that using the intel vc-intrinsic

    // Alternative ways:, we could also create a i16 view and pass that.
    // This way, both views point to the same vector, but accessed
    // differently based what view is used
    // Since, in our case, the vector is essentially bf16, but needed to
    // have a view of i16 just be passed in SPIR-V and inside DPAS
    // reinterpreted back bf16, we can safely use this approach
    //            / bf16 (initialization)         \
    // B = i8 -                                   ->
    //            \ i16 (passed to SPIR-V kernel) /

    // %tmp_sys_dpth = arith.muli %arg_sys_dpth, %c2 : index
    // %B_size = arith.muli %tmp_sys_dpth, %arg_N : index
    // %B_size = 8 * 2 * 16 = 256
    // Since, we are allocating bf16 as i8, %B_size * 2 is used
    // for allocation size
    // %B_size_i8 =  arith.muli %B_size, %c2 : index
    //  %B_size_i8 = 256 * 2 = 512
    %memref_B = gpu.alloc host_shared () : memref<512xi8>

    // Create a view of bf16 vector
    %memref_B_bf16 = memref.view %memref_B[%c0][] : memref<512xi8> to memref<256xbf16>

    %memref_B_bf16_1D = memref.cast %memref_B_bf16 : memref<256xbf16> to memref<?xbf16>
    // Initialize it to 1.1 as bf16, since that's the original data type for B
    call @fillResource1DBF16(%memref_B_bf16_1D, %cst_1) : (memref<*xbf16>, f32) -> ()

    // Setting up the Vector A
    // %A_size = arith.muli %tmp_sys_dpth, %arg_rpt_cnt : index
    // %A_size = 16 * 4 = 64
    // Since, we are allocating bf16 as i8, %A_size * 2 is used
    // for allocation size
    // %A_size_i8 =  arith.muli %A_size, %c2 : index
    // %A_size_i8 = 64 * 2 = 128

    %memref_A = gpu.alloc host_shared () : memref<128xi8>
    // Create a view of bf16 vector
    %memref_A_bf16 = memref.view %memref_A[%c0][] : memref<128 x i8> to memref<64 x bf16>

    %memref_A_bf16_1D = memref.cast %memref_A_bf16 : memref<64xbf16> to memref<?xbf16>
    // Initialize it to 2.2 as bf16, since that's the original data type for A
    call @fillResource1DBF16(%memref_A_bf16_1D, %cst_2) : (memref<*xbf16>, f32) -> ()

    // Calling the reference function/CPU version
    // call @dpas_ref(%arg_sys_dpth, %arg_rpt_cnt,  %arg_N, %memref_C, %memref_B_bf16, %memref_A_bf16) : (index, index, index, memref<?xf32>, memref<?xbf16>, memref<?xbf16>) -> ()

    // Calling the GPU version, using f16 view of B and A vector
    call @dpas_gpu(%arg_sys_dpth, %arg_rpt_cnt,  %arg_N, %memref_C_i8, %memref_B, %memref_A) : (index, index, index, memref<256xi8>, memref<512xi8>, memref<128xi8>) -> ()

    // Print the result
    %result = memref.cast %memref_C : memref<64xf32> to memref<*xf32>
    call @printMemrefF32(%result) : (memref<*xf32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-COUNT-64: 38.8301
    return
  }

  // main function
  func.func @main() {
    %cst_sys_dpth = arith.constant 8 : index
    %cst_rpt_cnt = arith.constant 4 : index
    %cst_N = arith.constant 16 : index

    call @dpas_test(%cst_sys_dpth, %cst_rpt_cnt, %cst_N) : (index, index, index) -> ()
    return
  }

  // Helper functions
  func.func private @fillResource1DBF16(memref<*xbf16>, f32) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF16(memref<*xf16>, f32) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
  func.func private @printMemrefBF16(memref<*xbf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}

}
