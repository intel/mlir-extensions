// RUN: imex-opt -convert-xegpu-to-vc -cse %s | FileCheck %s
module @gemm attributes {gpu.container_module} {
  gpu.module @test_kernel {
    // CHECK: func.func private @llvm.genx.nbarrier(i8, i8, i8) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.nbarrier", linkage_type = <Import>>}
    // CHECK: func.func private @llvm.genx.lsc.fence.i1(i1, i8, i8, i8) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.fence.i1", linkage_type = <Import>>}
    // CHECK: func.func private @llvm.genx.raw.send2.noresult.i1.v8i32(i8, i8, i1, i8, i8, i32, i32, vector<8xi32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.noresult.i1.v8i32", linkage_type = <Import>>}

    gpu.func @test_kernel(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // TODO: spirv.ExecutionMode @test_kernel "NamedBarrierCountINTEL", 16

      //CHECK: %[[cst:.*]] = arith.constant dense<0> : vector<8xi32>
      //CHECK: %[[c1_i32:.*]] = arith.constant 1 : i32
      //CHECK: %[[r0:.*]] = vector.insert %[[c1_i32]], %[[cst]] [2] : i32 into vector<8xi32>
      //CHECK: %[[c0_i8:.*]] = arith.constant 0 : i8
      //CHECK: %[[true:.*]] = arith.constant true
      //CHECK: %[[c1_i8:.*]] = arith.constant 1 : i8
      //CHECK: %[[c3_i8:.*]] = arith.constant 3 : i8
      //CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
      //CHECK: %[[c33554436_i32:.*]] = arith.constant 33554436 : i32
      //CHECK: func.call @llvm.genx.raw.send2.noresult.i1.v8i32(%[[c0_i8]], %[[c0_i8]], %[[true]], %[[c1_i8]], %[[c3_i8]], %[[c0_i32]], %[[c33554436_i32]], %[[r0]]) : (i8, i8, i1, i8, i8, i32, i32, vector<8xi32>) -> ()
      xegpu.alloc_nbarrier 16
      %nbarrier_id = arith.constant 1 : i8
      %nbarrier_role = arith.constant 0 : i8
      %payload = xegpu.init_nbarrier %nbarrier_id, %nbarrier_role : i8, i8 -> !xegpu.nbarrier
      xegpu.nbarrier_arrive %payload : !xegpu.nbarrier

      //CHECK: func.call @llvm.genx.lsc.fence.i1(%[[true]], %[[c0_i8]], %[[c0_i8]], %[[c0_i8]]) : (i1, i8, i8, i8) -> ()
      xegpu.fence memory_kind = global , fence_scope = workgroup

      //CHECK: %[[r1:.*]] = vector.extract %[[r0]][2] : i32 from vector<8xi32>
      //CHECK: %[[c255_i32:.*]] = arith.constant 255 : i32
      //CHECK: %[[r2:.*]] = arith.andi %[[r1]], %[[c255_i32]] : i32
      //CHECK: %[[r3:.*]] = arith.trunci %[[r2]] : i32 to i8
      //CHECK: func.call @llvm.genx.nbarrier(%[[c0_i8]], %[[r3]], %[[c0_i8]]) : (i8, i8, i8) -> ()
      xegpu.nbarrier_wait %payload : !xegpu.nbarrier

      //CHECK: gpu.return
      gpu.return
    }
  }
  }
