
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

module @gemm attributes {gpu.container_module} {
  memref.global "private" @__constant_8x16xf16 : memref<8x16xf16> = dense<1.000000e+00>
  memref.global "private" @__constant_16x16xf16 : memref<16x16xf16> = dense<1.000000e+00>
  memref.global "private" @__constant_16x16xf32 : memref<16x16xf32> = dense<0.000000e+00>
  func.func @test(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<8x16xf16>
    memref.copy %arg0, %memref : memref<8x16xf16> to memref<8x16xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<16x16xf16>
    memref.copy %arg1, %memref_0 : memref<16x16xf16> to memref<16x16xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<8x16xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<8x16xf16>, %memref_0 : memref<16x16xf16>, %memref_1 : memref<8x16xf32>)
    gpu.dealloc  %memref : memref<8x16xf16>
    gpu.dealloc  %memref_0 : memref<16x16xf16>
    return %memref_1 : memref<8x16xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    func.func private @llvm.genx.raw.sends2.noresult.i1.v8i32.v128f32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.sends2.noresult.i1.v8i32.v128f32", linkage_type = <Import>>}
    func.func private @llvm.genx.dpas2.v128f32.v128i32.v64i32(vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas2.v128f32.v128i32.v64i32", linkage_type = <Import>>}
    func.func private @llvm.genx.raw.send2.v128f32.i1.v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf32>) -> vector<128xf32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v128f32.i1.v8i32", linkage_type = <Import>>}
    func.func private @llvm.genx.raw.send2.v128i32.i1.v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v128i32.i1.v8i32", linkage_type = <Import>>}
    func.func private @llvm.genx.raw.send2.v64i32.i1.v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<64xi32>) -> vector<64xi32> attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v64i32.i1.v8i32", linkage_type = <Import>>}
    gpu.func @test_kernel(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<0> : vector<4xi64>
      %intptr = memref.extract_aligned_pointer_as_index %arg0 : memref<8x16xf16> -> index
      %0 = arith.index_castui %intptr : index to i64
      %1 = vector.insert %0, %cst [0] : i64 into vector<4xi64>
      %2 = vector.bitcast %1 : vector<4xi64> to vector<8xi32>
      %c31_i32 = arith.constant 31 : i32
      %c7_i32 = arith.constant 7 : i32
      %c31_i32_0 = arith.constant 31 : i32
      %3 = vector.insert %c31_i32, %2 [2] : i32 into vector<8xi32>
      %4 = vector.insert %c7_i32, %3 [3] : i32 into vector<8xi32>
      %5 = vector.insert %c31_i32_0, %4 [4] : i32 into vector<8xi32>
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_1 = arith.constant 0 : i32
      %6 = vector.insert %c0_i32, %5 [5] : i32 into vector<8xi32>
      %7 = vector.insert %c0_i32_1, %6 [6] : i32 into vector<8xi32>
      %c1807_i32 = arith.constant 1807 : i32
      %8 = vector.insert %c1807_i32, %7 [7] : i32 into vector<8xi32>
      %cst_2 = arith.constant dense<0> : vector<4xi64>
      %intptr_3 = memref.extract_aligned_pointer_as_index %arg1 : memref<16x16xf16> -> index
      %9 = arith.index_castui %intptr_3 : index to i64
      %10 = vector.insert %9, %cst_2 [0] : i64 into vector<4xi64>
      %11 = vector.bitcast %10 : vector<4xi64> to vector<8xi32>
      %c31_i32_4 = arith.constant 31 : i32
      %c15_i32 = arith.constant 15 : i32
      %c31_i32_5 = arith.constant 31 : i32
      %12 = vector.insert %c31_i32_4, %11 [2] : i32 into vector<8xi32>
      %13 = vector.insert %c15_i32, %12 [3] : i32 into vector<8xi32>
      %14 = vector.insert %c31_i32_5, %13 [4] : i32 into vector<8xi32>
      %c0_i32_6 = arith.constant 0 : i32
      %c0_i32_7 = arith.constant 0 : i32
      %15 = vector.insert %c0_i32_6, %14 [5] : i32 into vector<8xi32>
      %16 = vector.insert %c0_i32_7, %15 [6] : i32 into vector<8xi32>
      %c3855_i32 = arith.constant 3855 : i32
      %17 = vector.insert %c3855_i32, %16 [7] : i32 into vector<8xi32>
      %cst_8 = arith.constant dense<0> : vector<4xi64>
      %intptr_9 = memref.extract_aligned_pointer_as_index %arg2 : memref<8x16xf32> -> index
      %18 = arith.index_castui %intptr_9 : index to i64
      %19 = vector.insert %18, %cst_8 [0] : i64 into vector<4xi64>
      %20 = vector.bitcast %19 : vector<4xi64> to vector<8xi32>
      %c63_i32 = arith.constant 63 : i32
      %c7_i32_10 = arith.constant 7 : i32
      %c63_i32_11 = arith.constant 63 : i32
      %21 = vector.insert %c63_i32, %20 [2] : i32 into vector<8xi32>
      %22 = vector.insert %c7_i32_10, %21 [3] : i32 into vector<8xi32>
      %23 = vector.insert %c63_i32_11, %22 [4] : i32 into vector<8xi32>
      %c0_i32_12 = arith.constant 0 : i32
      %c0_i32_13 = arith.constant 0 : i32
      %24 = vector.insert %c0_i32_12, %23 [5] : i32 into vector<8xi32>
      %25 = vector.insert %c0_i32_13, %24 [6] : i32 into vector<8xi32>
      %c1807_i32_14 = arith.constant 1807 : i32
      %26 = vector.insert %c1807_i32_14, %25 [7] : i32 into vector<8xi32>
      %c0_i8 = arith.constant 0 : i8
      %c0_i8_15 = arith.constant 0 : i8
      %true = arith.constant true
      %c1_i8 = arith.constant 1 : i8
      %c4_i8 = arith.constant 4 : i8
      %c15_i8 = arith.constant 15 : i8
      %c0_i32_16 = arith.constant 0 : i32
      %c37880323_i32 = arith.constant 37880323 : i32
      %cst_17 = arith.constant dense<0> : vector<64xi32>
      %27 = func.call @llvm.genx.raw.send2.v64i32.i1.v8i32(%c0_i8, %c0_i8_15, %true, %c1_i8, %c4_i8, %c15_i8, %c0_i32_16, %c37880323_i32, %8, %cst_17) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<64xi32>) -> vector<64xi32>
      %c0_i8_18 = arith.constant 0 : i8
      %c0_i8_19 = arith.constant 0 : i8
      %true_20 = arith.constant true
      %c1_i8_21 = arith.constant 1 : i8
      %c8_i8 = arith.constant 8 : i8
      %c15_i8_22 = arith.constant 15 : i8
      %c0_i32_23 = arith.constant 0 : i32
      %c42074755_i32 = arith.constant 42074755 : i32
      %cst_24 = arith.constant dense<0> : vector<128xi32>
      %28 = func.call @llvm.genx.raw.send2.v128i32.i1.v8i32(%c0_i8_18, %c0_i8_19, %true_20, %c1_i8_21, %c8_i8, %c15_i8_22, %c0_i32_23, %c42074755_i32, %17, %cst_24) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32>
      %c0_i8_25 = arith.constant 0 : i8
      %c0_i8_26 = arith.constant 0 : i8
      %true_27 = arith.constant true
      %c1_i8_28 = arith.constant 1 : i8
      %c8_i8_29 = arith.constant 8 : i8
      %c15_i8_30 = arith.constant 15 : i8
      %c0_i32_31 = arith.constant 0 : i32
      %c42075139_i32 = arith.constant 42075139 : i32
      %cst_32 = arith.constant dense<0.000000e+00> : vector<128xf32>
      %29 = func.call @llvm.genx.raw.send2.v128f32.i1.v8i32(%c0_i8_25, %c0_i8_26, %true_27, %c1_i8_28, %c8_i8_29, %c15_i8_30, %c0_i32_31, %c42075139_i32, %26, %cst_32) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf32>) -> vector<128xf32>
      %c134744586_i32 = arith.constant 134744586 : i32
      %c10_i32 = arith.constant 10 : i32
      %c10_i32_33 = arith.constant 10 : i32
      %c8_i32 = arith.constant 8 : i32
      %c8_i32_34 = arith.constant 8 : i32
      %c0_i32_35 = arith.constant 0 : i32
      %30 = func.call @llvm.genx.dpas2.v128f32.v128i32.v64i32(%29, %28, %27, %c10_i32, %c10_i32_33, %c8_i32, %c8_i32_34, %c0_i32_35, %c0_i32_35) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %c0_i8_36 = arith.constant 0 : i8
      %c0_i8_37 = arith.constant 0 : i8
      %true_38 = arith.constant true
      %c1_i8_39 = arith.constant 1 : i8
      %c8_i8_40 = arith.constant 8 : i8
      %c15_i8_41 = arith.constant 15 : i8
      %c0_i32_42 = arith.constant 0 : i32
      %c33686535_i32 = arith.constant 33686535 : i32
      func.call @llvm.genx.raw.sends2.noresult.i1.v8i32.v128f32(%c0_i8_36, %c0_i8_37, %true_38, %c1_i8_39, %c8_i8_40, %c15_i8_41, %c0_i32_42, %c33686535_i32, %26, %30) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf32>) -> ()
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_8x16xf16 : memref<8x16xf16>
    %1 = memref.get_global @__constant_16x16xf16 : memref<16x16xf16>
    %2 = memref.get_global @__constant_16x16xf32 : memref<16x16xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %4 = arith.index_cast %arg0 : index to i16
        %5 = arith.index_cast %arg1 : index to i16
        %c16_i16 = arith.constant 16 : i16
        %6 = arith.muli %4, %c16_i16 : i16
        %7 = arith.addi %5, %6 : i16
        %8 = arith.uitofp %7 : i16 to f16
        %cst = arith.constant 1.000000e+00 : f16
        %9 = arith.divf %8, %cst : f16
        %cst_1 = arith.constant 1.000000e+00 : f16
        %10 = arith.addf %9, %cst_1 : f16
        memref.store %9, %0[%arg0, %arg1] : memref<8x16xf16>
        memref.store %10, %1[%arg0, %arg1] : memref<16x16xf16>
      }
    }
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %4 = memref.load %2[%arg0, %arg1] : memref<16x16xf32>
        %5 = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %4) -> (f32) {
          %6 = memref.load %0[%arg0, %arg2] : memref<8x16xf16>
          %7 = memref.load %1[%arg2, %arg1] : memref<16x16xf16>
          %8 = arith.mulf %6, %7 : f16
          %9 = arith.extf %8 : f16 to f32
          %10 = arith.addf %9, %arg3 : f32
          scf.yield %10 : f32
        }
        memref.store %5, %2[%arg0, %arg1] : memref<16x16xf32>
      }
    }
    %cast = memref.cast %2 : memref<16x16xf32> to memref<*xf32>
    %3 = call @test(%0, %1) : (memref<8x16xf16>, memref<16x16xf16>) -> memref<8x16xf32>
    %cast_0 = memref.cast %3 : memref<8x16xf32> to memref<*xf32>
    call @printMemrefF32(%cast_0) : (memref<*xf32>) -> ()
    call @printAllcloseF32(%cast_0, %cast) : (memref<*xf32>, memref<*xf32>) -> ()
    // CHECK:   [ALLCLOSE: TRUE]
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
}
