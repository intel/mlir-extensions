// RUN: imex-opt -imex-convert-gpu-to-spirv='enable-vc-intrinsic=true'  %s | FileCheck %s
// RUN: IMEX_NOT_PREFER_RAWSEND=1 imex-opt -imex-convert-gpu-to-spirv='enable-vc-intrinsic=true'  %s | FileCheck %s --check-prefix=LSC
module @gemm attributes {gpu.container_module} {
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
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // LSC: spirv.FunctionCall @llvm_genx_lsc_prefetch2d_stateless_i1_i64
      // LSC: spirv.FunctionCall @llvm_genx_lsc_prefetch2d_stateless_i1_i64
      // LSC: spirv.FunctionCall @llvm_genx_lsc_load2d_stateless_v64i32_i1_i64
      // LSC: spirv.FunctionCall @llvm_genx_lsc_load2d_stateless_v128i32_i1_i64
      // LSC: spirv.FunctionCall @llvm_genx_dpas_nosrc0_v128f32_v128i32_v64i32
      // LSC: spirv.FunctionCall @llvm_genx_lsc_store2d_stateless_i1_i64_v128f32
      // CHECK: %[[BASE:.*]] = spirv.ConvertPtrToU %arg0 : !spirv.ptr<!spirv.array<128 x f16>, CrossWorkgroup> to i64
      // CHECK: %[[BASE1:.*]] = spirv.VectorInsertDynamic %[[BASE]]
      // CHECK: %[[BASE2:.*]] = spirv.Bitcast %[[BASE1]]
      // CHECK: spirv.VectorInsertDynamic
      // CHECK: spirv.VectorInsertDynamic
      // CHECK: spirv.FunctionCall @llvm_genx_raw_send2_noresult_i1_v8i32
      // CHECK: spirv.FunctionCall @llvm_genx_raw_send2_noresult_i1_v8i32
      // CHECK: spirv.FunctionCall @llvm_genx_raw_send2_v64i32_i1_v8i32
      // CHECK: spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32
      // CHECK: spirv.FunctionCall @llvm_genx_dpas_nosrc0_v128f32_v128i32_v64i32
      // CHECK: spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f32
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %1 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
      %2 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.prefetch_nd %0 : !xegpu.tensor_desc<8x16xf16>
      xegpu.prefetch_nd %1 : !xegpu.tensor_desc<16x16xf16>

      %3 = xegpu.load_nd %0 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %4 = xegpu.load_nd %1 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
      %5 = xegpu.dpas %3, %4 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
      xegpu.store_nd %5, %2 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_8x16xf16 : memref<8x16xf16>
    %1 = memref.get_global @__constant_16x16xf16 : memref<16x16xf16>
    %2 = call @test(%0, %1) : (memref<8x16xf16>, memref<16x16xf16>) -> memref<8x16xf32>
    %cast = memref.cast %2 : memref<8x16xf32> to memref<*xf32>
    //call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
