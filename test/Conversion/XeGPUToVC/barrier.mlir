// RUN: imex-opt -convert-xegpu-to-vc='enable-vc-intrinsic=true useRawSend=false' %s | FileCheck %s
module @gemm attributes {gpu.container_module} {
  gpu.module @test_kernel {
    // CHECK: func.func private @llvm.genx.nbarrier(i8, i8, i8) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.nbarrier", linkage_type = <Import>>}
    // CHECK: func.func private @llvm.genx.lsc.fence.i1(i1, i8, i8, i8) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.lsc.fence.i1", linkage_type = <Import>>}
    // CHECK: func.func private @llvm.genx.raw.send2.noresult.i1.v8i32(i8, i8, i1, i8, i8, i32, i32, vector<8xi32>) attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.noresult.i1.v8i32", linkage_type = <Import>>}

    gpu.func @test_kernel(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // TODO: spirv.ExecutionMode @test_kernel "NamedBarrierCountINTEL", 16
     xegpu.alloc_nbarrier 16
      // CHECK: %[[c1:.*]] = arith.constant 1 : i8
      // CHECK: %[[c0:.*]] = arith.constant 0 : i8
      // CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
      %nbarrier_id = arith.constant 1 : i8
      %nbarrier_role = arith.constant 0 : i8
      // CHECK: %[[ROLE_i32:.*]] = arith.extui %[[c0]] : i8 to i32
      // CHECK: %[[NBARRIER_SRC:.*]] = arith.constant dense<0> : vector<8xi32>
      // CHECK: %[[ID:.*]] = arith.extui %[[c1]] : i8 to i32
      // CHECK: %[[c14:.*]] = arith.constant 14 : i32
      // CHECK: %[[ROLE:.*]] = arith.shli %[[c0_i32]], %[[c14]] : i32
      // CHECK: %[[P0:.*]] = arith.ori %[[ID]], %[[ROLE]] : i32
      // CHECK: %[[c16:.*]] = arith.constant 16 : i32
      // CHECK: %[[PRODUCERS:.*]] = arith.shli %[[ROLE_i32]], %[[c16]] : i32
      // CHECK: %[[P1:.*]] = arith.ori %[[P0]], %[[PRODUCERS]] : i32
      // CHECK: %[[c24:.*]] = arith.constant 24 : i32
      // CHECK: %[[P2:.*]] = arith.shli %[[ROLE_i32]], %[[c24]] : i32
      // CHECK: %[[PAYLOAD:.*]] = arith.ori %[[P1]], %[[P2]] : i32
      // CHECK: %[[NBARRIER:.*]] = vector.insert %[[PAYLOAD]], %[[NBARRIER_SRC]] [2] : i32 into vector<8xi32>
      %payload = xegpu.init_nbarrier %nbarrier_id, %nbarrier_role : i8, i8 -> !xegpu.nbarrier

      // CHECK: func.call @llvm.genx.raw.send2.noresult.i1.v8i32({{.*}}, %[[NBARRIER]]) : (i8, i8, i1, i8, i8, i32, i32, vector<8xi32>) -> ()
      xegpu.nbarrier_arrive %payload : !xegpu.nbarrier

      // CHECK: %[[TRUE:.*]] = arith.constant true
      // CHECK: func.call @llvm.genx.lsc.fence.i1(%[[TRUE]], {{.*}}) : (i1, i8, i8, i8) -> ()
      xegpu.fence memory_kind = global , fence_scope = workgroup

      // ADD CHECK %[[c128:.*]] = arith.constant -128 : i8
      // ADD CHECK func.call @llvm.genx.fence(%[[c128]]) : (i8) -> ()
      // xegpu.compile_hint

      // CHECK: %[[NBARRIER_VAL:.*]] = vector.extract %[[NBARRIER]][2] : i32 from vector<8xi32>
      // CHECK: %[[c255:.*]] = arith.constant 255 : i32
      // CHECK: %[[NBARRIER_ID:.*]] = arith.andi %[[NBARRIER_VAL]], %[[c255]] : i32
      // CHECK: %[[NBARRIER_PAYLOAD:.*]] = arith.trunci %[[NBARRIER_ID]] : i32 to i8
      // CHECK: func.call @llvm.genx.nbarrier(%{{.*}}, %[[NBARRIER_PAYLOAD]], %{{.*}}) : (i8, i8, i8) -> ()
      xegpu.nbarrier_wait %payload : !xegpu.nbarrier
      gpu.return
    }
  }
  }
