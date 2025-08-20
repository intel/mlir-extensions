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
/// This example uses 2d block load/store

module attributes {gpu.container_module}  {

  // function to setup the launch and launch the kernel
  // args: size_t systolic_depth, size_t repeat_cnt, size_t N
  func.func @dpas_gpu(%arg_sys_dpth: index, %arg_rpt_cnt: index, %arg_N: index, %arg_C : memref<?xi8>, %arg_B : memref<?xi8>, %arg_A : memref<?xi8>) {
    %c1 = arith.constant 1 : index

    // Since we are using only one DPAS instruction we are launching,
    // 1 workgroup and, 1 thread per workgroup
    gpu.launch_func @dpas_module::@dpas_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg_C : memref<?xi8>, %arg_B : memref<?xi8>, %arg_A : memref<?xi8>)
    return
  }

  // SPIR-V DPAS module, it holds the DPAS kernel
  spirv.module @__spv__dpas_module Physical64 OpenCL requires #spirv.vce<v1.0, [Int8, Int16, Int64, Float16, Kernel, Addresses, Linkage, Vector16, VectorAnyINTEL, Float16Buffer, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, FunctionFloatControlINTEL], [SPV_INTEL_float_controls2, SPV_INTEL_vector_compute]> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorAnyINTEL, VectorComputeINTEL, FunctionFloatControlINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    // DPAS kernel
    spirv.func @dpas_kernel(%arg0: !spirv.ptr<i8, CrossWorkgroup>, %arg1: !spirv.ptr<i8, CrossWorkgroup>, %arg2: !spirv.ptr<i8, CrossWorkgroup>)  "DontInline"  attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>, workgroup_attributions = 0 : i64, VectorComputeFunctionINTEL}  {
        %true = spirv.Constant true

        %uchar_0 = spirv.Constant 0 : i8
        %uchar_1 = spirv.Constant 1 : i8
        %uchar_2 = spirv.Constant 2 : i8
        %uchar_3 = spirv.Constant 3 : i8
        %uchar_4 = spirv.Constant 4 : i8
        %uchar_7 = spirv.Constant 7 : i8
        %uchar_8 = spirv.Constant 8 : i8
        %uchar_9 = spirv.Constant 9 : i8
        %uchar_10 = spirv.Constant 10 : i8
        %uchar_15 = spirv.Constant 15 : i8
        %uchar_16 = spirv.Constant 16 : i8

        %ushort_1 = spirv.Constant 1 : i16

        %uint_0 = spirv.Constant 0 : i32
        %uint_1 = spirv.Constant 1 : i32
        %uint_2 = spirv.Constant 2 : i32
        %uint_3 = spirv.Constant 3 : i32
        %uint_4 =  spirv.Constant 4 :  i32
        %uint_5 = spirv.Constant 5 : i32
        %uint_6 = spirv.Constant 6 : i32
        %uint_7 =  spirv.Constant 7 :  i32
        %uint_8 =  spirv.Constant 8 :  i32
        %uint_9 =  spirv.Constant 9 :  i32
        %uint_15 =  spirv.Constant 15 :  i32
        %uint_16 =  spirv.Constant 16 :  i32
        %uint_31 =  spirv.Constant 31 :  i32
        %uint_32 =  spirv.Constant 32 :  i32
        %uint_63 =  spirv.Constant 63 :  i32
        %uint_64 =  spirv.Constant 64 :  i32
        %uint_783 = spirv.Constant 783 :  i32
        %ulong_4294967295 = spirv.Constant 4294967295 :  i64  // 0xFFFFFFFF

        // Load Message descriptor : 0 00 0001 00100 000 0 0 000 010 0 0 0 000011
        // https://gfxspecs.intel.com/Predator/Home/Index/53680
        %uint_37749763 = spirv.Constant 37749763 : i32

        // Store Message descriptor : 0 0001 00000 000 00000 010 000 000111
        // https://gfxspecs.intel.com/Predator/Home/Index/53530
        %uint_33555463 = spirv.Constant 33555463 : i32
        %zero_vector = spirv.Constant dense<0.0> : vector<64xf32>
        %ulong_32 = spirv.Constant 32 : i64

        %addr_payload_vector_store = spirv.Constant dense<[0,0,0,0,0,0,0,0]> : vector<8xi32>

        // Cast the uchar pointers (i8 ptr) to ulongs (i64)
        %arg_0 = spirv.FunctionCall @llvm_genx_address_convert_i64_p1i8(%arg0) : (!spirv.ptr<i8, CrossWorkgroup>) -> i64
        %arg_1 = spirv.FunctionCall @llvm_genx_address_convert_i64_p1i8(%arg1) : (!spirv.ptr<i8, CrossWorkgroup>) -> i64
        %arg_2 = spirv.FunctionCall @llvm_genx_address_convert_i64_p1i8(%arg2) : (!spirv.ptr<i8, CrossWorkgroup>) -> i64

         // --------------- STORE USING RAW SEND -------------------
        // STORE: Extract the LSB and MSB of the address and convert them to 32 bits
        %addr_payload_msb_32_or = spirv.ShiftRightLogical %arg_0, %ulong_32 : i64, i64
        %addr_payload_msb_32 = spirv.UConvert %addr_payload_msb_32_or : i64 to i32
        %addr_payload_lsb_and = spirv.BitwiseAnd %arg_0, %ulong_4294967295 : i64
        %addr_payload_lsb_32 = spirv.UConvert %addr_payload_lsb_and : i64 to i32

        // https://gfxspecs.intel.com/Predator/Home/Index/53567
        // For storing a vector<16x16> f32
        // %addr_payload = vector<8xi32>
        // vector[0] = %addr_payload_lsb_32
        // vector[1] = %addr_payload_msb_32
        // vector[2] =  63 (bytes) ..width is 16*32 = 512/8 = 64 - 1 = 63
        // vector[3] = 16 (height in number of elements) - 1 = 3
        // vector[4] = pitch = 63. distance between two rows in number of bytes
        // vector[5] = block start X = 0;
        // vector[6] = block start Y = 0;
        // vector[7] = block width 224:231 (15), block height 232:239 (15), array_length 240:243 (0) = 0000 00000011 00001111 = 783
        %0 = spirv.VectorInsertDynamic %addr_payload_lsb_32, %addr_payload_vector_store[%uint_0] : vector<8xi32>, i32
        %1 = spirv.VectorInsertDynamic %addr_payload_msb_32, %0[%uint_1] : vector<8xi32>, i32
        %2 = spirv.VectorInsertDynamic %uint_63, %1[%uint_2] : vector<8xi32>, i32
        %3 = spirv.VectorInsertDynamic %uint_3, %2[%uint_3] : vector<8xi32>, i32
        %4 = spirv.VectorInsertDynamic %uint_63, %3[%uint_4] : vector<8xi32>, i32
        %5 = spirv.VectorInsertDynamic %uint_0, %4[%uint_5] : vector<8xi32>, i32
        %6 = spirv.VectorInsertDynamic %uint_0, %5[%uint_6] : vector<8xi32>, i32
        %7 = spirv.VectorInsertDynamic %uint_783, %6[%uint_7] : vector<8xi32>, i32


        // Load vector C using raw_send2
        %C = spirv.FunctionCall @llvm_genx_raw_send2_v64f32_i1_v8i32(%uchar_0, %uchar_1, %true, %uchar_1, %uchar_4, %uchar_15, %uint_0, %uint_37749763, %7, %zero_vector) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<64xf32>) -> vector<64xf32>

        // Load vector B using load stateless, this load uses internal VNNI transformation while loading
        %B = spirv.FunctionCall @llvm_genx_lsc_load2d_stateless_v128i32_i1_i64(%true, %uchar_0, %uchar_0, %uchar_2, %uchar_1, %uchar_1, %uint_16, %uint_16, %uchar_1, %arg_1, %uint_31, %uint_15, %uint_31, %uint_0, %uint_0) : (i1, i8, i8, i8, i8, i8, i32, i32, i8, i64, i32, i32, i32, i32, i32) -> vector<128xi32>

        // Load Vector A

        %A = spirv.FunctionCall @llvm_genx_lsc_load2d_stateless_v32i32_i1_i64(%true, %uchar_0, %uchar_0, %uchar_2, %uchar_1, %uchar_1, %uint_16, %uint_4, %uchar_0, %arg_2, %uint_31, %uint_3, %uint_31, %uint_0, %uint_0) : (i1, i8, i8, i8, i8, i8, i32, i32, i8, i64, i32, i32, i32, i32, i32) -> vector<32xi32>

        // Call dpas2
        %dpas_result =  spirv.FunctionCall @llvm_genx_dpas2_v64f32_v64f32_v128i32_v32i32(%C, %B, %A, %uint_9, %uint_9, %uint_8, %uint_4, %uint_0, %uint_0): (vector<64 x f32>, vector<128 x i32>, vector<32 x i32>, i32, i32, i32, i32, i32, i32) -> vector<64 x f32>


        //store using raw_send2_no_result
        spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v64f32(%uchar_0, %uchar_0, %true, %uchar_1, %uchar_4, %uchar_15, %uint_0, %uint_33555463, %7, %dpas_result) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<64xf32>) -> ()

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

    spirv.func @llvm_genx_lsc_load2d_stateless_v128i32_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i8, %arg5 : i8, %arg6 : i32, %arg7 : i32, %arg8 : i8, %arg9 : i64, %arg10 : i32, %arg11 : i32, %arg12 : i32, %arg13 : i32, %arg14 : i32) -> vector<128xi32> "Pure" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.lsc.load2d.stateless.v128i32.i1.i64",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_lsc_load2d_stateless_v64f32_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i8, %arg5 : i8, %arg6 : i32, %arg7 : i32, %arg8 : i8, %arg9 : i64, %arg10 : i32, %arg11 : i32, %arg12 : i32, %arg13 : i32, %arg14 : i32) -> vector<64xf32> "Pure" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.lsc.load2d.stateless.v64f32.i1.i64",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_lsc_load2d_stateless_v32i32_i1_i64(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i8, %arg5 : i8, %arg6 : i32, %arg7 : i32, %arg8 : i8, %arg9 : i64, %arg10 : i32, %arg11 : i32, %arg12 : i32, %arg13 : i32, %arg14 : i32) -> vector<32xi32> "Pure" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.lsc.load2d.stateless.v32i32.i1.i64",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_lsc_store2d_stateless_i1_i64_v64f32(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i8, %arg5 : i8, %arg6 : i32, %arg7 : i32, %arg8 : i8, %arg9 : i64, %arg10 : i32, %arg11 : i32, %arg12 : i32, %arg13 : i32, %arg14 : i32, %arg15 : vector<64xf32>) "None" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.lsc.store2d.stateless.i1.i64.v64f32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_dpas2_v64f32_v64f32_v128i32_v32i32(%arg0 : vector<64 x f32>, %arg1 : vector<128 x i32>, %arg2 : vector<32 x i32>, %arg3 : i32, %arg4 : i32, %arg5 : i32, %arg6 : i32, %arg7 : i32, %arg8 : i32) -> vector<64 x f32> "Pure" attributes{
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.dpas2.v64f32.v64f32.v128i32.v32i32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_raw_send2_v64f32_i1_v8i32(%arg0 : i8, %arg1 : i8, %arg2 : i1, %arg3 : i8, %arg4 : i8, %arg5 : i8, %arg6 : i32, %arg7 : i32, %arg8 : vector<8xi32>, %arg9 : vector<64xf32>) -> vector<64xf32> "Pure" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.raw.send2.v64f32.i1.v8i32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_raw_sends2_noresult_i1_v8i32_v64f32(%arg0 : i8, %arg1 : i8, %arg2 : i1, %arg3 : i8, %arg4 : i8, %arg5 : i8, %arg6 : i32, %arg7 : i32, %arg8 : vector<8xi32>, %arg9 : vector<64xf32>) "None" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.raw.sends2.noresult.i1.v8i32.v64f32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}
  }


  // GPU module, almost same as the SPIR-V module but without 'spirv' dialect specific properties
  gpu.module @dpas_module {
    gpu.func @dpas_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
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

    // Allocate vectors to be passed to function

    // Setting up Vector C
    %C_size = arith.muli %arg_rpt_cnt, %arg_N : index
    %C_size_i8 = arith.muli %C_size, %c4 : index

    %memref_C_i8 = gpu.alloc host_shared (%C_size_i8) : memref<?xi8>
    %memref_C = memref.view %memref_C_i8[%c0][%C_size] : memref<?xi8> to memref<?xf32>
    // Initialize C to 0
    call @fillResource1DF32(%memref_C, %cst_0) : (memref<?xf32>, f32) -> ()

    // Setting up the Vector B & A
    // B and A is setup slightly differently than other vectors, since B is
    // expected to be bf16 by the dpas instruction, but can not be passed
    // in SPIR-V (SPIR-V does not support bf16), we first allocate B
    // as i8 and then change the type (create view) to bf16. We use the bf16
    // view to initialize the vectors. We finally pass the i8 pointer to the
    // kernel and load bf16 from that using the intel vc-intrinsic

    // Alternative ways:, we could also create a i16 view and pass that.
    // This way, both views point to the same vector, but accessed
    // differently based what view is used
    // Since, in our case, the vector is essentially bf16, but needed to
    // have a view of i16 just to be passed in SPIR-V and inside DPAS
    // reinterpreted back bf16, we can safely use this approach
    //            / bf16 (initialization)         \
    // B = i8 -                                   ->
    //            \ i16 (passed to SPIR-V kernel) /

    %tmp_sys_dpth = arith.muli %arg_sys_dpth, %c2 : index
    %B_size = arith.muli %tmp_sys_dpth, %arg_N : index

    // Since, we are allocating bf16 as i8, %B_size * 2 is used
    // for allocation size
    %B_size_i8 =  arith.muli %B_size, %c2 : index

    %memref_B = gpu.alloc  host_shared (%B_size_i8) : memref<?xi8>

    // Create a view of bf16 vector
    %memref_B_bf16 = memref.view %memref_B[%c0][%B_size] : memref<?xi8> to memref<?xbf16>

    // Initialize it to 1.1 as bf16, since that's the original data type for B
    call @fillResource1DBF16(%memref_B_bf16, %cst_1) : (memref<*xbf16>, f32) -> ()

    // Setting up the Vector A
    %A_size = arith.muli %tmp_sys_dpth, %arg_rpt_cnt : index

    // Since, we are allocating bf16 as i8, %A_size * 2 is used
    // for allocation size
    %A_size_i8 =  arith.muli %A_size, %c2 : index

    %memref_A = gpu.alloc  host_shared (%A_size_i8) : memref<?xi8>
    // Create a view of bf16 vector
    %memref_A_bf16 = memref.view %memref_A[%c0][%A_size] : memref<? x i8> to memref<? x bf16>

    // SPIR-V type does not support bf16, hence passing vector 1, and vector 2 as i8, will load bf16 from this vector using the intel vc-intrinsic

    // Initialize it to 2.2 as bf16, since that's the original data type for A
    call @fillResource1DBF16(%memref_A_bf16, %cst_2) : (memref<*xbf16>, f32) -> ()

    // Calling the reference function/CPU version
    call @dpas_ref(%arg_sys_dpth, %arg_rpt_cnt,  %arg_N, %memref_C, %memref_B_bf16, %memref_A_bf16) : (index, index, index, memref<?xf32>, memref<?xbf16>, memref<?xbf16>) -> ()

    // Calling the GPU version, using f16 view of B and A vector
    call @dpas_gpu(%arg_sys_dpth, %arg_rpt_cnt,  %arg_N, %memref_C_i8, %memref_B, %memref_A) : (index, index, index, memref<?xi8>, memref<?xi8>, memref<?xi8>) -> ()

    // Print the result
    %result = memref.cast %memref_C : memref<?xf32> to memref<*xf32>
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
