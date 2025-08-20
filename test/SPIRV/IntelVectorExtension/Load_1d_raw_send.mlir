// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%mlir_runner_utils,%irunner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%mlir_runner_utils,%irunner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

// A simple test case showing how to use raw_send2 VC intrinsics for doing a load1d

module attributes {gpu.container_module}  {

  // function to setup the launch and launch the kernel
  func.func @load_1d_raw_send_gpu(%arg_A : memref<256xi8>, %arg_B : memref<256xi8>) {
    %c1 = arith.constant 1 : index

    // 1 workgroup and, 1 thread per workgroup
    gpu.launch_func @load_1d_raw_send_module::@load_1d_raw_send_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg_A : memref<256xi8>, %arg_B : memref<256xi8>)
    return
  }

  // SPIR-V load_1d_raw_send module, it holds the load_1d_raw_send kernel
  spirv.module @__spv__load_1d_raw_send_module Physical64 OpenCL requires #spirv.vce<v1.0, [Int8, Int16, Int64, Float16, Kernel, Addresses, Linkage, Vector16, VectorAnyINTEL, Float16Buffer, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, FunctionFloatControlINTEL], [SPV_INTEL_float_controls2, SPV_INTEL_vector_compute]> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorAnyINTEL, VectorComputeINTEL, FunctionFloatControlINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    // load_1d_raw_send kernel
    spirv.func @load_1d_raw_send_kernel(%arg1: !spirv.ptr<i8, CrossWorkgroup>, %arg2: !spirv.ptr<i8, CrossWorkgroup>)  "DontInline"  attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>, workgroup_attributions = 0 : i64, VectorComputeFunctionINTEL}  {
        %ushort_1 = spirv.Constant 1 : i16
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
        %uint_255 =  spirv.Constant 255 :  i32
        %uint_256 =  spirv.Constant 256 :  i32
        %uint_3855 = spirv.Constant 3855 :  i32


        // You can refer to :
        // https://gfxspecs.intel.com/Predator/Home/Index/53523
        // bit[0: 5] : opcode for instruction:
        // https://gfxspecs.intel.com/Predator/Home/Index/68015
        // bit[7: 8] Address Size
        // https://gfxspecs.intel.com/Predator/Home/Index/53558
        // bit[9: 11] Data Size
        // https://gfxspecs.intel.com/Predator/Home/Index/53563
        // bit[12: 14] Vector Size
        // https://gfxspecs.intel.com/Predator/Home/Index/53566
        // bit[15] Transpose
        // https://gfxspecs.intel.com/Predator/Home/Index/53565
        // https://github.com/intel-innersource/drivers.gpu.compute.vc-intrinsics/blob/cmc_experimental/GenXIntrinsics/include/llvm/GenXIntrinsics/GenXIntrinsics.h#L85
        // bit[17: 19] cacheHint
        // https://gfxspecs.intel.com/Predator/Home/Index/53560
        // https://github.com/intel-innersource/drivers.gpu.compute.vc-intrinsics/blob/cmc_experimental/GenXIntrinsics/include/llvm/GenXIntrinsics/Intrinsic_definitions.py#L1953
        // bit[20: 24] Dest Length
        // bit[25: 28] Src Length
        // https://gfxspecs.intel.com/Predator/Home/Index/53680
        %uint_38335872 = spirv.Constant 38335872 : i32

        %ulong_32 =  spirv.Constant 32 :  i64
        %zero_vector = spirv.Constant dense<0.0> : vector<64xf32>
        %addr_payload_vector = spirv.Constant dense<[0,0,0,0]> : vector<4xi64>

        // Cast the uchar pointers (i8 ptr) to ulongs (i64)
        %arg_1 = spirv.FunctionCall @llvm_genx_address_convert_i64_p1i8(%arg1) : (!spirv.ptr<i8, CrossWorkgroup>) -> i64
        %arg_2 = spirv.FunctionCall @llvm_genx_address_convert_i64_p1i8(%arg2) : (!spirv.ptr<i8, CrossWorkgroup>) -> i64


        // For load_1d we only need to put addr into the payload
        // In the generated assembly, a mov instruction will be generated to create the payload.
        %0 = spirv.VectorInsertDynamic %arg_1, %addr_payload_vector[%uint_0] : vector<4xi64>, i32

        // Load A from global using raw_send2
        %load_from_global = spirv.FunctionCall @llvm_genx_raw_send2_v64f32_i1_v4i64(%uchar_0, %uchar_0, %true, %uchar_1, %uchar_16, %uchar_15, %uint_0,%uint_38335872, %0, %zero_vector) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<4xi64>, vector<64xf32>) -> vector<64xf32>

       // store A in B (global)
        spirv.FunctionCall @llvm_genx_lsc_store_stateless_i1_i64_v64f32(%true, %uchar_4, %uchar_0, %uchar_0, %ushort_1, %uint_0, %uchar_3, %uchar_8, %uchar_2, %uchar_0, %arg_2, %load_from_global, %uint_0) : (i1, i8, i8, i8, i16, i32, i8, i8, i8, i8, i64, vector<64 x f32>, i32) -> () // -> mlir::NoneType
        spirv.Return
    }

    spirv.EntryPoint "Kernel" @load_1d_raw_send_kernel
    spirv.ExecutionMode @load_1d_raw_send_kernel "ContractionOff"
    spirv.ExecutionMode @load_1d_raw_send_kernel "SharedLocalMemorySizeINTEL", 0
    // Utility function declarations (Intel vc-intrinsics)

    spirv.func @llvm_genx_address_convert_i64_p1i8(%arg: !spirv.ptr<i8, CrossWorkgroup>) -> i64 "Pure" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.address.convert.i64.p1i8",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_lsc_store_stateless_i1_i64_v64f32(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i16, %arg5 : i32, %arg6 : i8, %arg7 : i8, %arg8 : i8, %arg9 : i8, %arg10 : i64, %arg11 : vector<64 x f32>, %arg12 : i32)  "None" attributes{
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.lsc.store.stateless.i1.i64.v64f32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}


    spirv.func @llvm_genx_raw_send2_v64f32_i1_v4i64(%arg0 : i8, %arg1 : i8, %arg2 : i1, %arg3 : i8, %arg4 : i8, %arg5 : i8, %arg6 : i32, %arg7 : i32, %arg8 : vector<4xi64>, %arg9 : vector<64xf32>) -> vector<64xf32> "Pure" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.raw.send2.v64f32.i1.v4i64",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}
  }

  // GPU module, almost same as the SPIR-V module but without 'spirv' dialect specific properties
  gpu.module @load_1d_raw_send_module {
    gpu.func @load_1d_raw_send_kernel(%arg1: memref<256xi8>, %arg2: memref<256xi8>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      gpu.return
    }
  }

  func.func @load_1d_raw_send_test(%arg_sys_dpth: index, %arg_rpt_cnt: index, %arg_N: index){
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

    call @load_1d_raw_send_gpu(%memref_A, %memref_B_i8) : (memref<256xi8>, memref<256xi8>) -> ()

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

    call @load_1d_raw_send_test(%cst_sys_dpth, %cst_rpt_cnt, %cst_N) : (index, index, index) -> ()
    return
  }

  // Helper functions
  func.func private @fillResource1DF16(memref<*xf16>, f32) attributes {llvm.emit_c_interface}
  func.func private @fillResource1DF32(memref<*xf32>, f32) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}

}
