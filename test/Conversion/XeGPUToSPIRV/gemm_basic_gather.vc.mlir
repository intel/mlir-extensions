// RUN: imex-opt -imex-convert-gpu-to-spirv='enable-vc-intrinsic=true'  %s | FileCheck %s --check-prefix=CHECK-RAW
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
      // CHECK-RAW: spirv.FunctionCall @llvm_genx_raw_send2_v64i32_i1_v16i64
      // CHECK-RAW: spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32
      %offsets = arith.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : vector<16xindex>
      %arg00 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128], strides: [1] : memref<8x16xf16> to memref<128xf16>
      %0 = xegpu.create_tdesc %arg00, %offsets {chunk_size = 8} : memref<128xf16>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf16, #xegpu.tdesc_attr<scattered = true>>
      %cst = arith.constant dense<true> : vector<128xi1>
      %mask = vector.shape_cast %cst : vector<128xi1> to vector<16x8xi1>
      %3 = xegpu.load %0, %mask : !xegpu.tensor_desc<16x8xf16, #xegpu.tdesc_attr<scattered = true>>, vector<16x8xi1> -> vector<16x8xf16>
      %66 = vector.shape_cast %3: vector<16x8xf16> to vector<128xf16>
      %6 = vector.shape_cast %66: vector<128xf16> to vector<8x16xf16>

      %1 = xegpu.create_nd_tdesc %arg1[0, 0] {boundary_check = true} : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
      %4 = xegpu.load_nd %1 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>

      %5 = xegpu.dpas %6, %4 : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>

      %offsets2 = arith.constant dense<[0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480]> : vector<16xindex>
      %arg02 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [128], strides: [1] : memref<8x16xf32> to memref<128xf32>
      %2 = xegpu.create_tdesc %arg02, %offsets2 {chunk_size = 8} : memref<128xf32>, vector<16xindex> -> !xegpu.tensor_desc<16x8xf32, #xegpu.tdesc_attr<scattered = true>>
      %7 = vector.shape_cast %5: vector<8x16xf32> to vector<128xf32>
      %8 = vector.shape_cast %7: vector<128xf32> to vector<16x8xf32>
      xegpu.store %8, %2, %mask : vector<16x8xf32>, !xegpu.tensor_desc<16x8xf32, #xegpu.tdesc_attr<scattered = true>>, vector<16x8xi1>
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
