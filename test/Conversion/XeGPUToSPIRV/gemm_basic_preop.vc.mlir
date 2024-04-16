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
    gpu.func @test_kernel(%A: memref<8x16xf16>, %B: memref<16x16xf16>, %C: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // LSC: spirv.FunctionCall @llvm_genx_lsc_prefetch2d_stateless_i1_i64
      // LSC: spirv.FunctionCall @llvm_genx_lsc_prefetch2d_stateless_i1_i64
      // LSC: spirv.FunctionCall @llvm_genx_lsc_load2d_stateless_v64i32_i1_i64
      // LSC: spirv.FunctionCall @llvm_genx_lsc_load2d_stateless_v128i32_i1_i64
      // LSC: spirv.FunctionCall @llvm_genx_dpas_nosrc0_v128f32_v128i32_v64i32
      // LSC: spirv.FunctionCall @llvm_genx_lsc_store2d_stateless_i1_i64_v128f32

      // CHECK: %[[A_tile_desc_base:.*]] = spirv.ConvertPtrToU %arg0 : !spirv.ptr<!spirv.array<128 x f16>, CrossWorkgroup> to i64
      // CHECK: %[[A_tile_payload_idx0:.*]] = spirv.VectorInsertDynamic %[[A_tile_desc_base]]
      // CHECK: %[[A_tile_payload_idx0_i32:.*]] = spirv.Bitcast %[[A_tile_payload_idx0]] : vector<4xi64> to vector<8xi32>
      // CHECK: %[[A_tile_payload_idx2:.*]] = spirv.VectorInsertDynamic
      // CHECK: %[[A_tile_payload_idx3:.*]] = spirv.VectorInsertDynamic
      // CHECK: %[[A_tile_payload_idx4:.*]] = spirv.VectorInsertDynamic
      // CHECK: %[[A_tile_payload_idx5:.*]] = spirv.VectorInsertDynamic
      // CHECK: %[[A_tile_payload_idx6:.*]] = spirv.VectorInsertDynamic
      // CHECK: %[[A_tile_payload_idx7:.*]] = spirv.VectorInsertDynamic

      // CHECK: %[[B_tile_desc_base:.*]] = spirv.ConvertPtrToU %arg1 : !spirv.ptr<!spirv.array<256 x f16>, CrossWorkgroup> to i64
      // CHECK: %[[B_tile_payload_idx0:.*]] = spirv.VectorInsertDynamic %[[B_tile_desc_base]]
      // CHECK: %[[B_tile_payload_idx0_i32:.*]] = spirv.Bitcast %[[B_tile_payload_idx0]] : vector<4xi64> to vector<8xi32>
      // CHECK: %[[B_tile_payload_idx2:.*]] = spirv.VectorInsertDynamic
      // CHECK: %[[B_tile_payload_idx3:.*]] = spirv.VectorInsertDynamic
      // CHECK: %[[B_tile_payload_idx4:.*]] = spirv.VectorInsertDynamic
      // CHECK: %[[B_tile_payload_idx5:.*]] = spirv.VectorInsertDynamic
      // CHECK: %[[B_tile_payload_idx6:.*]] = spirv.VectorInsertDynamic
      // CHECK: %[[B_tile_payload_idx7:.*]] = spirv.VectorInsertDynamic

      // CHECK: %[[C_tile_desc_base:.*]] = spirv.ConvertPtrToU %arg2 : !spirv.ptr<!spirv.array<128 x f32>, CrossWorkgroup> to i64
      // CHECK: %[[C_tile_payload_idx0:.*]] = spirv.VectorInsertDynamic %[[C_tile_desc_base]]
      // CHECK: %[[C_tile_payload_idx0_i32:.*]] = spirv.Bitcast %[[C_tile_payload_idx0]] : vector<4xi64> to vector<8xi32>
      // CHECK: %[[C_tile_payload_idx2:.*]] = spirv.VectorInsertDynamic
      // CHECK: %[[C_tile_payload_idx3:.*]] = spirv.VectorInsertDynamic
      // CHECK: %[[C_tile_payload_idx4:.*]] = spirv.VectorInsertDynamic
      // CHECK: %[[C_tile_payload_idx5:.*]] = spirv.VectorInsertDynamic
      // CHECK: %[[C_tile_payload_idx6:.*]] = spirv.VectorInsertDynamic
      // CHECK: %[[C_tile_payload_idx7:.*]] = spirv.VectorInsertDynamic

      // CHECK: spirv.FunctionCall @llvm_genx_raw_send2_noresult_i1_v8i32(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[A_tile_payload_idx7]])

      // CHECK: spirv.FunctionCall @llvm_genx_raw_send2_noresult_i1_v8i32(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[B_tile_payload_idx7]])

      // CHECK: %[[A_increment:.*]] = spirv.Constant dense<1.000000e+00> : vector<128xf16>

      // CHECK: %[[A_i32:.*]] = spirv.FunctionCall @llvm_genx_raw_send2_v64i32_i1_v8i32(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[A_tile_payload_idx7]], %{{.*}})
      // CHECK: %[[A_f16:.*]] = spirv.Bitcast %[[A_i32]] : vector<64xi32> to vector<128xf16>
      // CHECK: %[[A_f16_inc:.*]] = spirv.FAdd %[[A_f16]], %[[A_increment]] : vector<128xf16>

      // CHECK: %[[B_i32:.*]] = spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[B_tile_payload_idx7]], %{{.*}})
      // CHECK: %[[B_f16:.*]] = spirv.Bitcast %[[B_i32]] : vector<128xi32> to vector<256xf16>

      // CHECK: %[[A_back_i32:.*]] = spirv.Bitcast %[[A_f16_inc]] : vector<128xf16> to vector<64xi32>
      // CHECK: %[[B_back_i32:.*]] = spirv.Bitcast %[[B_f16]] : vector<256xf16> to vector<128xi32>
      // CHECK: %[[DPAS_RES:.*]] = spirv.FunctionCall @llvm_genx_dpas_nosrc0_v128f32_v128i32_v64i32(%[[B_back_i32]], %[[A_back_i32]], %{{.*}})

      // CHECK: spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f32(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[C_tile_payload_idx7]], %[[DPAS_RES]])
      %A_tdesc = xegpu.create_nd_tdesc %A[0, 0] {mode = vc} : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      %B_tdesc = xegpu.create_nd_tdesc %B[0, 0] {mode = vc} : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
      %C_tdesc = xegpu.create_nd_tdesc %C[0, 0] {mode = vc} : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
      xegpu.prefetch_nd %A_tdesc {mode = vc} : !xegpu.tensor_desc<8x16xf16>
      xegpu.prefetch_nd %B_tdesc {mode = vc} : !xegpu.tensor_desc<16x16xf16>
      %A_increment = arith.constant dense<1.0> : vector<128xf16>
      %A_increment_ = vector.shape_cast %A_increment : vector<128xf16> to vector<8x8x2xf16>

      %A_tensor = xegpu.load_nd %A_tdesc  {mode = vc, vnni_axis = 1} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
      %A_tensor_incremented = arith.addf %A_tensor, %A_increment_ : vector<8x8x2xf16>
      %B_tensor = xegpu.load_nd %B_tdesc  {mode = vc, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
      %dpas_result = xegpu.dpas %A_tensor_incremented, %B_tensor {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
      xegpu.store_nd %dpas_result, %C_tdesc {mode = vc} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
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
