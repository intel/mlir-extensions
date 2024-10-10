// RUN: imex-opt -convert-xegpu-to-vc -cse %s | FileCheck %s
module @gemm attributes {gpu.container_module} {
  gpu.module @test_kernel {
    // CHECK: func.func private @llvm.genx.nbarrier(i8, i8, i8) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.nbarrier", linkage_type = <Import>>}
    // CHECK: func.func private @llvm.genx.lsc.fence.i1(i1, i8, i8, i8) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.fence.i1", linkage_type = <Import>>}
    // CHECK: func.func private @llvm.genx.nbarrier.arrive(i8, i8, i8, i8) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.nbarrier.arrive", linkage_type = <Import>>}

    gpu.func @test_kernel(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // TODO: spirv.ExecutionMode @test_kernel "NamedBarrierCountINTEL", 16

      //CHECK: %[[c1_i8:.*]] = arith.constant 1 : i8
      //CHECK: %[[c0_i8:.*]] = arith.constant 0 : i8
      //CHECK: func.call @llvm.genx.nbarrier.arrive(%[[c1_i8]], %[[c0_i8]], %[[c0_i8]], %[[c0_i8]]) : (i8, i8, i8, i8) -> ()
      xegpu.alloc_nbarrier 16
      %nbarrier_id = arith.constant 1 : i8
      %nbarrier_role = arith.constant 0 : i8
      %payload = xegpu.init_nbarrier %nbarrier_id, %nbarrier_role : i8, i8 -> !xegpu.nbarrier
      xegpu.nbarrier_arrive %payload : !xegpu.nbarrier

      //CHECK: %[[true:.*]] = arith.constant true
      //CHECK: func.call @llvm.genx.lsc.fence.i1(%[[true]], %[[c0_i8]], %[[c0_i8]], %[[c0_i8]]) : (i1, i8, i8, i8) -> ()
      xegpu.fence memory_kind = global , fence_scope = workgroup

      //CHECK: func.call @llvm.genx.nbarrier(%[[c0_i8]], %[[c1_i8]], %[[c0_i8]]) : (i8, i8, i8) -> ()
      xegpu.nbarrier_wait %payload : !xegpu.nbarrier

      //CHECK: gpu.return
      gpu.return
    }
  }
  }
