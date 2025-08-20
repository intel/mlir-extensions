// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%mlir_runner_utils,%irunner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/spirv-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%mlir_runner_utils,%irunner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck

/// A simple load2d/store2d example
/// This example loads and stores 16x16xf32 elements using raw_send2/store2d

module attributes {gpu.container_module}  {

  func.func @load_store_2d_gpu(%arg_In : memref<1024xi8>, %arg_Out : memref<1024xi8>) {
    %c1 = arith.constant 1 : index

    // 1 workgroup and, 1 thread per workgroup
    gpu.launch_func @load_store_2d_module::@load_store_2d_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg_In : memref<1024xi8>, %arg_Out : memref<1024xi8>)
    return
  }

  // SPIR-V module, it holds the kernel
  spirv.module @__spv__load_store_2d_module Physical64 OpenCL requires #spirv.vce<v1.0, [Int8, Int16, Int64, Float16, Kernel, Addresses, Linkage, Vector16, VectorAnyINTEL, Float16Buffer, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, FunctionFloatControlINTEL], [SPV_INTEL_float_controls2, SPV_INTEL_vector_compute]> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorAnyINTEL, VectorComputeINTEL, FunctionFloatControlINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    // load_store_2d kernel
    spirv.func @load_store_2d_kernel(%arg0: !spirv.ptr<i8, CrossWorkgroup>, %arg1: !spirv.ptr<i8, CrossWorkgroup>)  "DontInline"  attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>, workgroup_attributions = 0 : i64, VectorComputeFunctionINTEL}  {

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

        // 0xFFFFFFFF
        %uint_4294967295 = spirv.Constant 4294967295 :  i64


        // Load Message descriptor : 0 00 0001 10000 000 0 0 000 010 0 0 0 000011
        // https://gfxspecs.intel.com/Predator/Home/Index/53680
        %uint_50332675 = spirv.Constant 50332675 : i32

        %ulong_32 =  spirv.Constant 32 :  i64
        %zero_vector = spirv.Constant dense<0.0> : vector<256xf32>
        %addr_payload_vector = spirv.Constant dense<[0,0,0,0,0,0,0,0]> : vector<8xi32>


        // Cast the uchar pointers (i8 ptr) to ulongs (i64)
        %arg_0 = spirv.FunctionCall @llvm_genx_address_convert_i64_p1i8(%arg0) : (!spirv.ptr<i8, CrossWorkgroup>) -> i64
        %arg_1 = spirv.FunctionCall @llvm_genx_address_convert_i64_p1i8(%arg1) : (!spirv.ptr<i8, CrossWorkgroup>) -> i64


        // --------------- LOAD USING RAW SEND -------------------
        %addr_payload_msb_32_or = spirv.ShiftRightLogical %arg_0, %ulong_32 : i64, i64
        %addr_payload_msb_32 = spirv.UConvert %addr_payload_msb_32_or : i64 to i32
        %addr_payload_lsb_and = spirv.BitwiseAnd %arg_0, %uint_4294967295 : i64
        %addr_payload_lsb_32 = spirv.UConvert %addr_payload_lsb_and : i64 to i32


        %0 = spirv.VectorInsertDynamic %addr_payload_lsb_32, %addr_payload_vector[%uint_0] : vector<8xi32>, i32
        %1 = spirv.VectorInsertDynamic %addr_payload_msb_32, %0[%uint_1] : vector<8xi32>, i32
        %2 = spirv.VectorInsertDynamic %uint_63, %1[%uint_2] : vector<8xi32>, i32
        %3 = spirv.VectorInsertDynamic %uint_15, %2[%uint_3] : vector<8xi32>, i32
        %4 = spirv.VectorInsertDynamic %uint_63, %3[%uint_4] : vector<8xi32>, i32
        %5 = spirv.VectorInsertDynamic %uint_0, %4[%uint_5] : vector<8xi32>, i32
        %6 = spirv.VectorInsertDynamic %uint_0, %5[%uint_6] : vector<8xi32>, i32
        %7 = spirv.VectorInsertDynamic %uint_3855, %6[%uint_7] : vector<8xi32>, i32


        // https://gfxspecs.intel.com/Predator/Home/Index/53567
        // %addr_payload = vector<8xi32> : spirv.vector<8xi32>
        // vector[0] = %addr_payload_lsb_32
        // vector[1] = %addr_payload_msb_32
        // vector[2] =  63 (bytes) ..width is 16*32 = 512/8 = 64 - 1 = 63
        // vector[3] = 16 (height in number of elements) - 1 = 15
        // vector[4] = pitch = 63. distance between two rows in number of bytes.
        // vector[5] = block start X = 0;
        // vector[6] = block start Y = 0;
        // vector[7] = block width 224:231 (15), block height 232:239 (15), array_length 240:243 (0) = 0000 00001111 00001111 = 3855


        // Load data from the input using raw_send2
        %input = spirv.FunctionCall @llvm_genx_raw_send2_v256f32_i1_v8i32(%uchar_0, %uchar_1, %true, %uchar_1, %uchar_16, %uchar_15, %uint_0, %uint_50332675, %7, %zero_vector) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<256xf32>) -> vector<256xf32>

        // Store the result
        spirv.FunctionCall @llvm_genx_lsc_store2d_stateless_i1_i64_v256f32(%true, %uchar_0, %uchar_0, %uchar_3, %uchar_1, %uchar_1, %uint_16, %uint_16, %uchar_0, %arg_1, %uint_63, %uint_15, %uint_63, %uint_0, %uint_0, %input) : (i1, i8, i8, i8, i8, i8, i32, i32, i8, i64, i32, i32, i32, i32, i32, vector<256xf32>) -> ()

        spirv.Return
    }
    spirv.EntryPoint "Kernel" @load_store_2d_kernel
    spirv.ExecutionMode @load_store_2d_kernel "ContractionOff"
    spirv.ExecutionMode @load_store_2d_kernel "SharedLocalMemorySizeINTEL", 0
    // Utility function declarations (Intel vc-intrinsics)
    spirv.func @llvm_genx_address_convert_i64_p1i8(%arg: !spirv.ptr<i8, CrossWorkgroup>) -> i64 "Pure" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.address.convert.i64.p1i8",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_raw_send2_v256f32_i1_v8i32(%arg0 : i8, %arg1 : i8, %arg2 : i1, %arg3 : i8, %arg4 : i8, %arg5 : i8, %arg6 : i32, %arg7 : i32, %arg8 : vector<8xi32>, %arg9 : vector<256xf32>) -> vector<256xf32> "Pure" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.raw.send2.v256f32.i1.v8i32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

    spirv.func @llvm_genx_lsc_store2d_stateless_i1_i64_v256f32(%arg0 : i1, %arg1 : i8, %arg2 : i8, %arg3 : i8, %arg4 : i8, %arg5 : i8, %arg6 : i32, %arg7 : i32, %arg8 : i8, %arg9 : i64, %arg10 : i32, %arg11 : i32, %arg12 : i32, %arg13 : i32, %arg14 : i32, %arg15 : vector<256xf32>) "None" attributes {
        linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.lsc.store2d.stateless.i1.i64.v256f32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}

  }

  // GPU module, almost same as the SPIR-V module but without 'spirv' dialect specific properties
  gpu.module @load_store_2d_module {
    gpu.func @load_store_2d_kernel(%arg0: memref<1024xi8>, %arg1: memref<1024xi8>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      gpu.return
    }
  }

  func.func @load_store_2d_test(){
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.100000e+00 : f32
    %cst_2 = arith.constant 2.200000e+00 : f32

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index

    // Allocate Inputs and Outputs to be passed to function

    %memref_In_i8 = gpu.alloc host_shared () : memref<1024xi8>
    %memref_In = memref.view %memref_In_i8[%c0][] : memref<1024xi8> to memref<256xf32>
    // Initialize Input
    %memref_In_1D = memref.cast %memref_In : memref<256xf32> to memref<?xf32>
    call @fillResource1DF32(%memref_In_1D, %cst_1) : (memref<?xf32>, f32) -> ()

    // Output
    %memref_Out_i8 = gpu.alloc host_shared () : memref<1024xi8>
    %memref_Out = memref.view %memref_Out_i8[%c0][] : memref<1024xi8> to memref<256xf32>
    // Initialize Out to 0
    %memref_Out_1D = memref.cast %memref_Out : memref<256xf32> to memref<?xf32>
    call @fillResource1DF32(%memref_Out_1D, %cst_0) : (memref<?xf32>, f32) -> ()

    // Calling the GPU version of load and store
    call @load_store_2d_gpu(%memref_In_i8, %memref_Out_i8) : (memref<1024xi8>, memref<1024xi8>) -> ()

    // Print the result
    %result = memref.cast %memref_Out_1D : memref<?xf32> to memref<*xf32>
    call @printMemrefF32(%result) : (memref<*xf32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-COUNT-256: 1.1
    return
  }

  // main function
  func.func @main() {
    call @load_store_2d_test() : () -> ()
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
