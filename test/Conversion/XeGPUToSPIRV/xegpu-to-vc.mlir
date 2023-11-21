// RUN: imex-opt -imex-convert-gpu-to-spirv='enable-vc-intrinsic=true'  %s | FileCheck %s

gpu.module @test attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  // CHECK: spirv.ConvertPtrToU
  // CHECK: spirv.VectorInsertDynamic
  gpu.func @create_nd_tdesc(%src: memref<64x64xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c32 = arith.constant 16 : index
    %0 = xegpu.create_nd_tdesc %src[%c32, 0] {mode = vc} : memref<64x64xf16> -> !xegpu.tensor_desc<8x16xf16>
    gpu.return
  }


  // CHECK-LABEL: spirv.func @llvm_genx_raw_send2_v128i32_i1_v8i32
  // CHECK (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>)
  // CHECK: -> vector<128xi32> "None" attributes
  // CHECK: {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name
  // CHECK: = "llvm.genx.raw.send2.v128i32.i1.v8i32", linkage_type = <Import>>}
  // CHECK-LABEL: spirv.func @load_nd
  // CHECK: %[[ptr:.*]]: !spirv.ptr<!spirv.array<4096 x f16>, CrossWorkgroup>
  // CHECK:  %[[ptr_i64:.*]] = spirv.ConvertPtrToU %[[ptr]] : !spirv.ptr<!spirv.array<4096 x f16>, CrossWorkgroup> to i64
  // CHECK:  %{{.*}} = spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32

  gpu.func @load_nd(%src : memref<64x64xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %1 = xegpu.create_nd_tdesc %src[0, 0] { mode = vc} : memref<64x64xf16> ->  !xegpu.tensor_desc<16x16xf16>
    %3 = xegpu.load_nd %1  {vnni_axis = 0, mode = vc} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    gpu.return
  }

  // CHECK-LABEL: spirv.func @llvm_genx_dpas_nosrc0_v128f32_v128i32_v64i32(vector<128xi32>, vector<64xi32>, i32)
  // CHECK: -> vector<128xf32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes =
  // CHECK:  #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32", linkage_type = <Import>>}
  // CHECK-LABEL: spirv.func @dpas
  // CHECK: (%[[A:.*]]: vector<64xi32>, %[[B:.*]]: vector<128xi32>)
  // CHECK-NEXT: %[[cst134744586_i32:.*]] = spirv.Constant 134744586 : i32
  // CHECK-NEXT: %{{.*}} = spirv.FunctionCall @llvm_genx_dpas_nosrc0_v128f32_v128i32_v64i32(%[[B]], %[[A]], %[[cst134744586_i32]])
  // CHECK: (vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32>
  gpu.func @dpas(%A : vector<8x8x2xf16>, %B : vector<8x16x2xf16>)
    kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %C = xegpu.dpas %A, %B { mode = vc }: vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
    gpu.return
  }


  // CHECK: (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf32>)
  // CHECK: "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name
  // CHECK: = "llvm.genx.raw.sends2.noresult.i1.v8i32.v128f32", linkage_type = <Import>>}
  // CHECK: (%[[value:.*]]: vector<128xf32>, %[[ptr:.*]]: !spirv.ptr<!spirv.array<4096 x f32>, CrossWorkgroup>)
  // CHECK: %[[ptr_i64]] = spirv.ConvertPtrToU %[[ptr]] : !spirv.ptr<!spirv.array<4096 x f32>, CrossWorkgroup> to i64
  // CHECK: spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128f32
  gpu.func @store_nd(%value : vector<8x16xf32>, %dest : memref<64x64xf32>)
    kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %1 = xegpu.create_nd_tdesc %dest[0, 0] { mode = vc } : memref<64x64xf32> -> !xegpu.tensor_desc<8x16xf32>
    xegpu.store_nd %value, %1 { mode = vc } : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
    gpu.return
  }
  // CHECK: (i8, i8, i1, i8, i8, i32, i32, vector<8xi32>)
  // CHECK: "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name =
  // CHECK:  "llvm.genx.raw.send2.noresult.i1.v8i32", linkage_type = <Import>>}
  // CHECK: (%[[ptr:.*]]: !spirv.ptr<!spirv.array<4096 x f16>, CrossWorkgroup>)
  // CHECK: spirv.ConvertPtrToU %[[ptr]] : !spirv.ptr<!spirv.array<4096 x f16>, CrossWorkgroup> to i64
  // CHECK: spirv.VectorInsertDynamic
  // CHECK: spirv.FunctionCall @llvm_genx_raw_send2_noresult_i1_v8i32
  gpu.func @prefetch(%src : memref<64x64xf16>)
    kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %src[0, 0] { mode = vc } : memref<64x64xf16> -> !xegpu.tensor_desc<8x16xf16>
      xegpu.prefetch_nd %0 { mode = vc } : !xegpu.tensor_desc<8x16xf16>
      gpu.return
  }

}
