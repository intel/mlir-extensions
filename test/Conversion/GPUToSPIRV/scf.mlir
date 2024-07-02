// RUN: imex-opt -allow-unregistered-dialect -imex-convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
    gpu.container_module,
    spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    // CHECK:       spirv.module @{{.*}} Physical64 OpenCL
    // CHECK-LABEL: spirv.func @loop_kernel
    // CHECK-SAME: spirv.entry_point_abi = #spirv.entry_point_abi<>, workgroup_attributions = 0 : i64
    gpu.func @loop_kernel(%arg2 : memref<10xf32>, %arg3 : memref<10xf32>) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      // CHECK: spirv.Branch ^bb1
      // CHECK: ^bb1:  // pred: ^bb0
       cf.br ^bb1
      ^bb1:  // pred: ^bb0
      // CHECK: %[[LB:.*]] = spirv.Constant 4 : i64
        %lb = arith.constant 4 : index
        // CHECK: %[[UB:.*]] = spirv.Constant 42 : i64
        %ub = arith.constant 42 : index
        // CHECK: %[[STEP:.*]] = spirv.Constant 2 : i64
        %step = arith.constant 2 : index
        // CHECK: spirv.mlir.loop {
        // CHECK-NEXT: spirv.Branch ^[[HEADER:.*]](%[[LB]] : i64)
        // CHECK:      ^[[HEADER]](%[[INDVAR:.*]]: i64):
        // CHECK:        %[[CMP:.*]] = spirv.SLessThan %[[INDVAR]], %[[UB]] : i64
        // CHECK:        spirv.BranchConditional %[[CMP]], ^[[BODY:.*]], ^[[MERGE:.*]]
        // CHECK:      ^[[BODY]]:
        // CHECK:        %[[OFFSET1:.*]] = spirv.Constant 0 : i64
        // CHECK:        %[[STRIDE1:.*]] = spirv.Constant 1 : i64
        // CHECK:        spirv.AccessChain {{%.*}}
        // CHECK:        %[[OFFSET2:.*]] = spirv.Constant 0 : i64
        // CHECK:        %[[STRIDE2:.*]] = spirv.Constant 1 : i64
        // CHECK:        spirv.AccessChain {{%.*}}
        // CHECK:        %[[INCREMENT:.*]] = spirv.IAdd %[[INDVAR]], %[[STEP]] : i64
        // CHECK:        spirv.Branch ^[[HEADER]](%[[INCREMENT]] : i64)
        // CHECK:      ^[[MERGE]]
        // CHECK:        spirv.mlir.merge
        // CHECK:      }
        scf.for %arg4 = %lb to %ub step %step {
             %1 = memref.load %arg2[%arg4] : memref<10xf32>
            memref.store %1, %arg3[%arg4] : memref<10xf32>
        }
      // CHECK: spirv.Return
      gpu.return
    }
  }

  func.func @main() {
    %0 = "op"() : () -> (memref<10xf32>)
    %1 = "op"() : () -> (memref<10xf32>)
    %cst = arith.constant 1 : index
    gpu.launch_func @kernels::@loop_kernel
        blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)
        args(%0 : memref<10xf32>, %1 : memref<10xf32>)
    return
  }
}
