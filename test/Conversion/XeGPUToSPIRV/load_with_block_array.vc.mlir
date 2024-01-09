// RUN: imex-opt -imex-convert-gpu-to-spirv='enable-vc-intrinsic=true'  %s | FileCheck %s
// RUN: IMEX_NOT_PREFER_RAWSEND=1 imex-opt -imex-convert-gpu-to-spirv='enable-vc-intrinsic=true'  %s | FileCheck %s --check-prefix=LSC
module @gemm attributes {gpu.container_module} {
  memref.global "private" constant @__constant_8x32xf16 : memref<8x32xf16> = dense<5.000000e-01>
  func.func @test(%arg0: memref<8x32xf16>) -> memref<8x32xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<8x32xf16>
    memref.copy %arg0, %memref : memref<8x32xf16> to memref<8x32xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<8x32xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8x32xf16>, %memref_1 : memref<8x32xf32>)
    gpu.dealloc  %memref : memref<8x32xf16>
    return %memref_1 : memref<8x32xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<8x32xf16>, %arg1: memref<8x32xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      //CHECK: spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32
      //CHECK: spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f32
      //CHECK: spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f32
      //LSC: spirv.FunctionCall @llvm_genx_lsc_load2d_stateless_v128i32_i1_i64
      //LSC: spirv.FunctionCall @llvm_genx_lsc_store2d_stateless_i1_i64_v128f32
      //LSC: spirv.FunctionCall @llvm_genx_lsc_store2d_stateless_i1_i64_v128f32
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] {mode = vc} : memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<array_length = 2>>
      %1 = xegpu.create_nd_tdesc %arg1[0, 0] {mode = vc} : memref<8x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %2 = xegpu.create_nd_tdesc %arg1[0, 16] {mode = vc} : memref<8x32xf32> -> !xegpu.tensor_desc<8x16xf32>
      %3 = xegpu.load_nd %0  {vnni_axis = 1, l1_hint = cached, l2_hint = cached, mode = vc} : !xegpu.tensor_desc<8x16xf16, #xegpu.tdesc_attr<array_length = 2>> -> vector<2x8x8x2xf16>
      %4 = vector.extract %3[0]: vector<8x8x2xf16> from vector<2x8x8x2xf16>
      %5 = vector.extract %3[1]: vector<8x8x2xf16> from vector<2x8x8x2xf16>
      %6 = vector.shape_cast %4: vector<8x8x2xf16> to vector<8x16xf16>
      %7 = vector.shape_cast %5: vector<8x8x2xf16> to vector<8x16xf16>
      %8 = arith.extf %6: vector<8x16xf16> to vector<8x16xf32>
      %9 = arith.extf %7: vector<8x16xf16> to vector<8x16xf32>
      xegpu.store_nd %8, %1 {mode = vc} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %9, %2 {mode = vc} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_8x32xf16 : memref<8x32xf16>
    %2 = call @test(%0) : (memref<8x32xf16>) -> memref<8x32xf32>
    %cast = memref.cast %2 : memref<8x32xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
