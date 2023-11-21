// RUN: imex-opt -imex-convert-gpu-to-spirv='enable-vc-intrinsic=true'  %s | FileCheck %s --check-prefix=CHECK-RAW
// RUN: IMEX_NOT_PREFER_RAWSEND=1 imex-opt -imex-convert-gpu-to-spirv='enable-vc-intrinsic=true'  %s | FileCheck %s --check-prefix=CHECK-LSC
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
      // CHECK-RAW: %[[BASE:.*]] = spirv.ConvertPtrToU %arg0 : !spirv.ptr<!spirv.array<128 x f16>, CrossWorkgroup> to i64
      // CHECK-RAW: spirv.VectorInsertDynamic %[[BASE]]
      // CHECK-RAW: spirv.FunctionCall @llvm_genx_raw_send2_v32i64_i1_v8i32
      // CHECK-RAW: spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v64i64
      // CHECK-LSC: spirv.FunctionCall @llvm_genx_lsc_load_stateless_v32i64_i1_i64
      // CHECK-LSC: spirv.FunctionCall @llvm_genx_lsc_store_stateless_i1_i64_v64i64
      %arg00 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128], strides: [1] : memref<8x16xf16> to memref<128xf16>
      %0 = xegpu.create_nd_tdesc %arg00[0] {mode = vc} : memref<128xf16> -> !xegpu.tensor_desc<128xf16>
      %1 = xegpu.create_nd_tdesc %arg1[0, 0] {mode = vc, boundary_check = true} : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
      %arg02 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [128], strides: [1] : memref<8x16xf32> to memref<128xf32>
      %2 = xegpu.create_nd_tdesc %arg02[0] {mode = vc} : memref<128xf32> -> !xegpu.tensor_desc<128xf32>
      %3 = xegpu.load_nd %0 {mode = vc}: !xegpu.tensor_desc<128xf16> -> vector<128xf16>
      %4 = xegpu.load_nd %1 {mode = vc, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
      %6 = vector.shape_cast %3: vector<128xf16> to vector<8x8x2xf16>
      %5 = xegpu.dpas %6, %4 {mode = vc}: vector<8x8x2xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
      %7 = vector.shape_cast %5: vector<8x16xf32> to vector<128xf32>
      xegpu.store_nd %7, %2 {mode = vc}: vector<128xf32>, !xegpu.tensor_desc<128xf32>
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
