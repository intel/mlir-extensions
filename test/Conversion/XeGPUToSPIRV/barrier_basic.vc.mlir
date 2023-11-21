// RUN: imex-opt -imex-convert-gpu-to-spirv='enable-vc-intrinsic=true'  %s | FileCheck %s
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
      // CHECK: spirv.FunctionCall @llvm_genx_raw_send2_noresult_i1_v8i32
      // CHECK: spirv.FunctionCall @llvm.genx.lsc.fence.i1
      // CHECK: spirv.FunctionCall @llvm_genx_fence
      // CHECK: spirv.FunctionCall @llvm_genx_nbarrier
      // CHECK: spirv.ExecutionMode @test_kernel "NamedBarrierCountINTEL", 16
      xegpu.alloc_nbarrier 16
      %nbarrier_id = arith.constant 1 : i8
      %nbarrier_role = arith.constant 0 : i8
      %payload = xegpu.create_nbarrier %nbarrier_id, %nbarrier_role {num_producers = 32 : i8, num_consumers = 32 : i8} : (i8, i8) -> !xegpu.nbarrier
      xegpu.nbarrier_arrive %payload : !xegpu.nbarrier
      xegpu.mfence {memory_kind = "ugm" , fence_op = "none", fence_scope = "local"}
      xegpu.compile_hint
      xegpu.nbarrier_wait %payload : !xegpu.nbarrier
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
