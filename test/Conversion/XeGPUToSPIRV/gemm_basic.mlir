// RUN: imex-opt -imex-convert-gpu-to-spirv='enable-vc-intrinsic=false'  %s | FileCheck %s

#sg_map_fp16_a = #xegpu.sg_map<{mma_block_size = [8, 16], wi_layout = [2, 8], wi_data = [1, 2]}>
#sg_map_fp16_b = #xegpu.sg_map<{mma_block_size = [16, 16], wi_layout = [1, 16], wi_data = [1, 1]}>
#sg_map_fp16_c = #xegpu.sg_map<{mma_block_size = [8, 16], wi_layout = [1, 16], wi_data = [1, 1]}>
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
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: %[[a:.*]] = spirv.FunctionCall @llvm.genx.GenISA.LSC2DBlockRead.v8i16
      // CHECK: %[[a0:.*]] = spirv.Bitcast %[[a]]
      // CHECK: %[[b:.*]] = spirv.FunctionCall @llvm.genx.GenISA.LSC2DBlockRead.v16i16
      // CHECK: %[[b0:.*]] = spirv.Bitcast %[[b]]
      // CHECK: %[[A:.*]] = spirv.Bitcast %[[a0]]
      // CHECK: %[[B:.*]] = spirv.Bitcast %[[b0]]
      // CHECK: %[[C:.*]] = spirv.FunctionCall @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v16i16
      // CHECK-SAME: %[[A]], %[[B]]
      // CHECK: spirv.FunctionCall @llvm.genx.GenISA.LSC2DBlockWrite.isVoid
      // CHECK-SAME: %[[C]]
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16, #sg_map_fp16_a>
      %1 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #sg_map_fp16_b>
      %2 = xegpu.create_nd_tdesc %arg2[0, 0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #sg_map_fp16_c>
      %3 = xegpu.load_nd %0  {vnni_axis = 1} : !xegpu.tensor_desc<8x16xf16, #sg_map_fp16_a> -> vector<4x1x2xf16>
      %4 = xegpu.load_nd %1  {vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16, #sg_map_fp16_b> -> vector<8x1x2xf16>
      %5 = xegpu.dpas %3, %4 : vector<4x1x2xf16>, vector<8x1x2xf16> -> vector<8x1xf32>
      xegpu.store_nd %5, %2 : vector<8x1xf32>, !xegpu.tensor_desc<8x16xf32, #sg_map_fp16_c>
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
