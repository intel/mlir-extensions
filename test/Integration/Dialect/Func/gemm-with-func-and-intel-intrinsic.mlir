// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/func-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/func-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=opencl-runtime -i %s --pass-pipeline-file=%p/func-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%opencl_runtime --filecheck

module @gemm attributes {gpu.container_module,
spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  memref.global "private" constant @__constant_8x16xf16 : memref<8x16xf16> = dense<5.000000e-01>
  memref.global "private" constant @__constant_16x16xf16 : memref<16x16xf16> = dense<1.099610e+00>
  func.func @test(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<8x16xf16>
    memref.copy %arg0, %memref : memref<8x16xf16> to memref<8x16xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<16x16xf16>
    memref.copy %arg1, %memref_0 : memref<16x16xf16> to memref<16x16xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<8x16xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8x16xf16>, %memref_0 : memref<16x16xf16>, %memref_1 : memref<8x16xf32>)
    gpu.dealloc  %memref : memref<8x16xf16>
    gpu.dealloc  %memref_0 : memref<16x16xf16>
    return %memref_1 : memref<8x16xf32>
  }

gpu.module @test_kernel {
  func.func private @llvm.genx.raw.sends2.noresult.i1.v8i32.v64i64(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<64xi64>) attributes{
    linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.raw.sends2.noresult.i1.v8i32.v64i64",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}
  func.func private @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32> attributes{
    linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}
  func.func private @llvm.genx.raw.send2.v128i32.i1.v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32> attributes{
    linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.raw.send2.v128i32.i1.v8i32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}
  func.func private @llvm.genx.raw.send2.v32i64.i1.v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<32xi64>) -> vector<32xi64> attributes{
    linkage_attributes=#spirv.linkage_attributes<
            linkage_name="llvm.genx.raw.send2.v32i64.i1.v8i32",
            linkage_type=<Import>
        >,
        VectorComputeFunctionINTEL}
  gpu.func @test_kernel(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL,spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128], strides: [1] : memref<8x16xf16> to memref<128xf16>
    %cst = arith.constant dense<0> : vector<4xi64>
    %intptr = memref.extract_aligned_pointer_as_index %reinterpret_cast : memref<128xf16> -> index
    %0 = arith.index_castui %intptr : index to i64
    %1 = vector.insert %0, %cst[0] : i64 into vector<4xi64>
    %2 = vector.bitcast %1 : vector<4xi64> to vector<8xi32>
    %cst_0 = arith.constant dense<0> : vector<4xi64>
    %cst_to_non_cst = arith.addi %cst_0, %cst_0 : vector<4xi64>
    %intptr_1 = memref.extract_aligned_pointer_as_index %arg1 : memref<16x16xf16> -> index
    %3 = arith.index_castui %intptr_1 : index to i64
    %4 = vector.insert %3, %cst_to_non_cst[0] : i64 into vector<4xi64>
    %5 = vector.bitcast %4 : vector<4xi64> to vector<8xi32>
    %c31_i32 = arith.constant 31 : i32
    %c15_i32 = arith.constant 15 : i32
    %c31_i32_2 = arith.constant 31 : i32
    %6 = vector.insert %c31_i32, %5 [2] : i32 into vector<8xi32>
    %7 = vector.insert %c15_i32, %6 [3] : i32 into vector<8xi32>
    %8 = vector.insert %c31_i32_2, %7 [4] : i32 into vector<8xi32>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_3 = arith.constant 0 : i32
    %9 = vector.insert %c0_i32, %8 [5] : i32 into vector<8xi32>
    %10 = vector.insert %c0_i32_3, %9 [6] : i32 into vector<8xi32>
    %c3855_i32 = arith.constant 3855 : i32
    %11 = vector.insert %c3855_i32, %10 [7] : i32 into vector<8xi32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [128], strides: [1] : memref<8x16xf32> to memref<128xf32>
    %cst_5_t = arith.constant dense<0> : vector<4xi64>
    %cst_5 = arith.addi %cst_5_t, %cst_5_t : vector<4xi64>
    %intptr_6 = memref.extract_aligned_pointer_as_index %reinterpret_cast_4 : memref<128xf32> -> index
    %12 = arith.index_castui %intptr_6 : index to i64
    %13 = vector.insert %12, %cst_5 [0] : i64 into vector<4xi64>
    %14 = vector.bitcast %13 : vector<4xi64> to vector<8xi32>
    %c0_i8 = arith.constant 0 : i8
    %c0_i8_7 = arith.constant 0 : i8
    %true = arith.constant true
    %c1_i8 = arith.constant 1 : i8
    %c4_i8 = arith.constant 4 : i8
    %c15_i8 = arith.constant 15 : i8
    %c0_i32_8 = arith.constant 0 : i32
    %c42133376_i32 = arith.constant 42133376 : i32
    %cst_9 = arith.constant dense<0> : vector<32xi64>
    %15 = func.call @llvm.genx.raw.send2.v32i64.i1.v8i32(%c0_i8, %c0_i8_7, %true, %c1_i8, %c4_i8, %c15_i8, %c0_i32_8, %c42133376_i32, %2, %cst_9) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<32xi64>) -> vector<32xi64>
    %16 = vector.bitcast %15 : vector<32xi64> to vector<128xf16>
    %c0_i8_10 = arith.constant 0 : i8
    %c0_i8_11 = arith.constant 0 : i8
    %true_12 = arith.constant true
    %c1_i8_13 = arith.constant 1 : i8
    %c8_i8 = arith.constant 8 : i8
    %c15_i8_14 = arith.constant 15 : i8
    %c0_i32_15 = arith.constant 0 : i32
    %c42074755_i32 = arith.constant 42074755 : i32
    %cst_16 = arith.constant dense<0> : vector<128xi32>
    %17 = func.call @llvm.genx.raw.send2.v128i32.i1.v8i32(%c0_i8_10, %c0_i8_11, %true_12, %c1_i8_13, %c8_i8, %c15_i8_14, %c0_i32_15, %c42074755_i32, %11, %cst_16) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32>
    %18 = vector.bitcast %16 : vector<128xf16> to vector<64xi32>
    %c134744586_i32 = arith.constant 134744586 : i32
    %19 = func.call @llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32(%17, %18, %c134744586_i32) : (vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32>
    %c0_i8_17 = arith.constant 0 : i8
    %c0_i8_18 = arith.constant 0 : i8
    %true_19 = arith.constant true
    %c1_i8_20 = arith.constant 1 : i8
    %c8_i8_21 = arith.constant 8 : i8
    %c15_i8_22 = arith.constant 15 : i8
    %c0_i32_23 = arith.constant 0 : i32
    %c33748868_i32 = arith.constant 33748868 : i32
    %20 = vector.bitcast %19 : vector<128xf32> to vector<64xi64>
    func.call @llvm.genx.raw.sends2.noresult.i1.v8i32.v64i64(%c0_i8_17, %c0_i8_18, %true_19, %c1_i8_20, %c8_i8_21, %c15_i8_22, %c0_i32_23, %c33748868_i32, %14, %20) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<64xi64>) -> ()
    gpu.return
  }
}
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_8x16xf16 : memref<8x16xf16>
    %1 = memref.get_global @__constant_16x16xf16 : memref<16x16xf16>
    %2 = call @test(%0, %1) : (memref<8x16xf16>, memref<16x16xf16>) -> memref<8x16xf32>
    %cast = memref.cast %2 : memref<8x16xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-COUNT-128: 8.79688
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
