// RUN: imex-opt -imex-convert-gpu-to-spirv='enable-vc-intrinsic=true'  %s | FileCheck %s

gpu.module @test attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  // CHECK: spirv.ConvertPtrToU
  // CHECK: spirv.VectorInsertDynamic
  gpu.func @create_nd_tdesc(%src: memref<64x64xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %c32 = arith.constant 16 : index
    %0 = xegpu.create_nd_tdesc %src[%c32, 0] : memref<64x64xf16> -> !xegpu.tensor_desc<8x16xf16>
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
    %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<64x64xf16> ->  !xegpu.tensor_desc<16x16xf16>
    %3 = xegpu.load_nd %1 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
    gpu.return
  }

  // CHECK-LABEL: spirv.func @llvm_genx_dpas_nosrc0_v128f32_v128i32_v64i32(vector<128xi32>, vector<64xi32>, i32)
  // CHECK: -> vector<128xf32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes =
  // CHECK:  #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas.nosrc0.v128f32.v128i32.v64i32", linkage_type = <Import>>}
  // CHECK-LABEL: spirv.func @dpas
  // CHECK: (%[[A:.*]]: vector<128xf16>, %[[B:.*]]: vector<256xf16>)
  // CHECK-NEXT: %[[cst134744586_i32:.*]] = spirv.Constant 134744586 : i32
  // CHECK-NEXT: %[[A_cast:.*]] = spirv.Bitcast %[[A]] : vector<128xf16> to vector<64xi32>
  // CHECK-NEXT: %[[B_cast:.*]] = spirv.Bitcast %[[B]] : vector<256xf16> to vector<128xi32>
  // CHECK-NEXT: %{{.*}} = spirv.FunctionCall @llvm_genx_dpas_nosrc0_v128f32_v128i32_v64i32(%[[B_cast]], %[[A_cast]], %[[cst134744586_i32]])
  // CHECK: (vector<128xi32>, vector<64xi32>, i32) -> vector<128xf32>
  gpu.func @dpas(%A : vector<8x8x2xf16>, %B : vector<8x16x2xf16>)
    kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %C = xegpu.dpas %A, %B : vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
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
    %1 = xegpu.create_nd_tdesc %dest[0, 0]  : memref<64x64xf32> -> !xegpu.tensor_desc<8x16xf32>
    xegpu.store_nd %value, %1 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
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
      %0 = xegpu.create_nd_tdesc %src[0, 0] : memref<64x64xf16> -> !xegpu.tensor_desc<8x16xf16>
      xegpu.prefetch_nd %0 : !xegpu.tensor_desc<8x16xf16>
      gpu.return
  }

  gpu.func @vector_extract_strided_slice(%src_1d : vector<128xf32>, %src_2d : vector<8x16xf32>, %src_nd : vector<2x32x8xf32>)
    kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: spirv.VectorShuffle [32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32,
      // CHECK: 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32]
      // CHECK: %[[vec_1d:.*]], %[[vec_1d]] : vector<128xf32>, vector<128xf32> -> vector<16xf32>
      %0 = vector.extract_strided_slice %src_1d {sizes = [16], strides = [1], offsets = [32]}
        : vector<128xf32> to vector<16xf32>

      // CHECK: spirv.VectorShuffle [8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 24 : i32,
      // CHECK: 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 40 : i32, 41 : i32, 42 : i32,
      // CHECK: 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32,
      // CHECK: 61 : i32, 62 : i32, 63 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32,
      // CHECK: 79 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 104 : i32,
      // CHECK: 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 120 : i32, 121 : i32,
      // CHECK: 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32]
      // CHECK: %[[vec_2d:.*]], %[[vec_2d]] : vector<128xf32>, vector<128xf32> -> vector<64xf32>
      %1 = vector.extract_strided_slice %src_2d {sizes = [8, 8], strides = [1, 1], offsets = [0, 8]}
        : vector<8x16xf32> to vector<8x8xf32>

      // CHECK: spirv.VectorShuffle [192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32,
      // CHECK: 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32,
      // CHECK: 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32,
      // CHECK: 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32,
      // CHECK: 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32,
      // CHECK: 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32,
      // CHECK: 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32,
      // CHECK: 253 : i32, 254 : i32, 255 : i32] %[[vec_nd:.*]], %[[vec_nd]] : vector<512xf32>, vector<512xf32> -> vector<64xf32>
      %2 = vector.extract_strided_slice %src_nd { offsets = [0, 24], strides = [1, 1], sizes = [1, 8] } : vector<2x32x8xf32> to vector<1x8x8xf32>
      gpu.return

  }

}
