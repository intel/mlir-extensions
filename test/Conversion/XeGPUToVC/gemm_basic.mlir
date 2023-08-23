// RUN: imex-opt -imex-convert-gpu-to-spirv  %s | FileCheck %s
module @gemm attributes {gpu.container_module} {
  //memref.global "private" constant @__constant_8x16xf16 : memref<8x16xf16> = dense<5.000000e-01>
  //memref.global "private" constant @__constant_16x16xf16 : memref<16x16xf16> = dense<1.099610e+00>
  // 0~12.8 step 0.1
  memref.global "private" constant @__constant_8x16xf16 : memref<8x16xf16> = dense<"0x0000662E6632CD3466360038CD389A39663A333B003C663CCD3C333D9A3D003E663ECD3E333F9A3F0040334066409A40CD400041334166419A41CD410042334266429A42CD420043334366439A43CD4300441A4433444D44664480449A44B344CD44E64400451A4533454D45664580459A45B345CD45E64500461A4633464D46664680469A46B346CD46E64600471A4733474D47664780479A47B347CD47E64700480D481A482648334840484D485A486648734880488D489A48A648B348C048CD48DA48E648F34800490D491A492649334940494D495A496649734980498D499A49A649B349C049CD49DA49E649F349004A0D4A1A4A264A334A404A4D4A5A4A">
  // 0~25.5 step 0.1
  memref.global "private" constant @__constant_16x16xf16 : memref<16x16xf16> = dense<"0x0000662E6632CD3466360038CD389A39663A333B003C663CCD3C333D9A3D003E663ECD3E333F9A3F0040334066409A40CD400041334166419A41CD410042334266429A42CD420043334366439A43CD4300441A4433444D44664480449A44B344CD44E64400451A4533454D45664580459A45B345CD45E64500461A4633464D46664680469A46B346CD46E64600471A4733474D47664780479A47B347CD47E64700480D481A482648334840484D485A486648734880488D489A48A648B348C048CD48DA48E648F34800490D491A492649334940494D495A496649734980498D499A49A649B349C049CD49DA49E649F349004A0D4A1A4A264A334A404A4D4A5A4A664A734A804A8D4A9A4AA64AB34AC04ACD4ADA4AE64AF34A004B0D4B1A4B264B334B404B4D4B5A4B664B734B804B8D4B9A4BA64BB34BC04BCD4BDA4BE64BF34B004C064C0D4C134C1A4C204C264C2D4C334C3A4C404C464C4D4C534C5A4C604C664C6D4C734C7A4C804C864C8D4C934C9A4CA04CA64CAD4CB34CBA4CC04CC64CCD4CD34CDA4CE04CE64CED4CF34CFA4C004D064D0D4D134D1A4D204D264D2D4D334D3A4D404D464D4D4D534D5A4D604D664D6D4D734D7A4D804D864D8D4D934D9A4DA04DA64DAD4DB34DBA4DC04DC64DCD4DD34DDA4DE04DE64DED4DF34DFA4D004E064E0D4E134E1A4E204E264E2D4E334E3A4E404E464E4D4E534E5A4E604E">
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
      // CHECK: spirv.FunctionCall @llvm_genx_lsc_load2d_stateless_v64i32_i1_i64
      // CHECK: spirv.FunctionCall @llvm_genx_lsc_load2d_stateless_v128i32_i1_i64
      // CHECK: spirv.FunctionCall @llvm_genx_dpas_nosrc0_v128f32_v64i32_v128i32
      // CHECK: spirv.FunctionCall @llvm_genx_lsc_store2d_stateless_i1_i64_v128f32
      %0 = xegpu.init_tile %arg0[0, 0] [8, 16] [1, 1] : memref<8x16xf16> -> !xegpu.tile<8x16xf16>
      %1 = xegpu.init_tile %arg1[0, 0] [16, 16] [1, 1] : memref<16x16xf16> -> !xegpu.tile<16x16xf16>
      %2 = xegpu.init_tile %arg2[0, 0] [8, 16] [1, 1] : memref<8x16xf32> -> !xegpu.tile<8x16xf32>
      %3 = xegpu.load_2d %0  : !xegpu.tile<8x16xf16> -> vector<8x8x2xf16>
      %4 = xegpu.load_2d %1  {VNNI_AXIS = 1 : i32} : !xegpu.tile<16x16xf16> -> vector<8x16x2xf16>
      %5 = xegpu.dpas %3, %4 : (vector<8x8x2xf16>, vector<8x16x2xf16>) -> vector<8x16xf32>
      xegpu.store_2d %2,  %5 : (!xegpu.tile<8x16xf32>, vector<8x16xf32>)
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
