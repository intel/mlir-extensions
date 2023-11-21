// RUN: imex-opt -imex-convert-gpu-to-spirv='enable-vc-intrinsic=true'  %s | FileCheck %s --check-prefix=CHECK
module @gemm attributes {gpu.container_module} {
  memref.global "private" @__constant_8x16xf32 : memref<8x16xf32> = dense<4.000000e-01>
  func.func @test(%arg0: memref<8x16xf32>) -> memref<8x16xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<8x16xf32>
    memref.copy %arg0, %memref : memref<8x16xf32> to memref<8x16xf32>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8x16xf32>)
    return %memref : memref<8x16xf32>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: llvm_genx_lsc_xatomic_stateless_v16i32_v16i1_v16i64
      %mask = arith.constant dense<true> : vector<16xi1>
      %offsets = arith.constant dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xindex>
      %1 = arith.constant dense<0.5> : vector<16xf32>
      %2 = xegpu.create_tdesc %arg0, %offsets {chunk_size_per_lane = 1} : memref<8x16xf32>, vector<16xindex> -> !xegpu.tensor_desc<16xf32, #xegpu.scattered>
      %3 = xegpu.atomic_rmw "addf" %2, %mask, %1 : !xegpu.tensor_desc<16xf32, #xegpu.scattered>, vector<16xi1>, vector<16xf32> -> vector<16xf32>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_8x16xf32 : memref<8x16xf32>
    %2 = call @test(%0) : (memref<8x16xf32>) -> (memref<8x16xf32>)
    %cast = memref.cast %2 : memref<8x16xf32> to memref<*xf32>
    // call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}
