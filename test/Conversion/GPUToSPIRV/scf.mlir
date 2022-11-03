// RUN: imex-opt -allow-unregistered-dialect -imex-convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
    gpu.container_module,
    spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    // CHECK:       spirv.module @{{.*}} Logical GLSL450 {
    // CHECK-LABEL: spirv.func @loop_kernel
    // CHECK-SAME: {{%.*}}: !spirv.ptr<!spirv.struct<(!spirv.array<10 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}
    // CHECK-SAME: {{%.*}}: !spirv.ptr<!spirv.struct<(!spirv.array<10 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}
    // CHECK-SAME: spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[32, 4, 1]> : vector<3xi32>>
    gpu.func @loop_kernel(%arg2 : memref<10xf32, #spirv.storage_class<StorageBuffer>>, %arg3 : memref<10xf32, #spirv.storage_class<StorageBuffer>>) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[32, 4, 1]>: vector<3xi32>>} {
      // CHECK: spirv.Branch ^bb1
      // CHECK: ^bb1:  // pred: ^bb0
       cf.br ^bb1
      ^bb1:  // pred: ^bb0
      // CHECK: %[[LB:.*]] = spirv.Constant 4 : i32
        %lb = arith.constant 4 : index
        // CHECK: %[[UB:.*]] = spirv.Constant 42 : i32
        %ub = arith.constant 42 : index
        // CHECK: %[[STEP:.*]] = spirv.Constant 2 : i32
        %step = arith.constant 2 : index
        // CHECK: spirv.mlir.loop {
        // CHECK-NEXT: spirv.Branch ^[[HEADER:.*]](%[[LB]] : i32)
        // CHECK:      ^[[HEADER]](%[[INDVAR:.*]]: i32):
        // CHECK:        %[[CMP:.*]] = spirv.SLessThan %[[INDVAR]], %[[UB]] : i32
        // CHECK:        spirv.BranchConditional %[[CMP]], ^[[BODY:.*]], ^[[MERGE:.*]]
        // CHECK:      ^[[BODY]]:
        // CHECK:        %[[ZERO1:.*]] = spirv.Constant 0 : i32
        // CHECK:        %[[OFFSET1:.*]] = spirv.Constant 0 : i32
        // CHECK:        %[[STRIDE1:.*]] = spirv.Constant 1 : i32
        // CHECK:        %[[UPDATE1:.*]] = spirv.IMul %[[STRIDE1]], %[[INDVAR]] : i32
        // CHECK:        %[[INDEX1:.*]] = spirv.IAdd %[[OFFSET1]], %[[UPDATE1]] : i32
        // CHECK:        spirv.AccessChain {{%.*}}{{\[}}%[[ZERO1]], %[[INDEX1]]{{\]}}
        // CHECK:        %[[ZERO2:.*]] = spirv.Constant 0 : i32
        // CHECK:        %[[OFFSET2:.*]] = spirv.Constant 0 : i32
        // CHECK:        %[[STRIDE2:.*]] = spirv.Constant 1 : i32
        // CHECK:        %[[UPDATE2:.*]] = spirv.IMul %[[STRIDE2]], %[[INDVAR]] : i32
        // CHECK:        %[[INDEX2:.*]] = spirv.IAdd %[[OFFSET2]], %[[UPDATE2]] : i32
        // CHECK:        spirv.AccessChain {{%.*}}[%[[ZERO2]], %[[INDEX2]]]
        // CHECK:        %[[INCREMENT:.*]] = spirv.IAdd %[[INDVAR]], %[[STEP]] : i32
        // CHECK:        spirv.Branch ^[[HEADER]](%[[INCREMENT]] : i32)
        // CHECK:      ^[[MERGE]]
        // CHECK:        spirv.mlir.merge
        // CHECK:      }
        scf.for %arg4 = %lb to %ub step %step {
             %1 = memref.load %arg2[%arg4] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
            memref.store %1, %arg3[%arg4] : memref<10xf32, #spirv.storage_class<StorageBuffer>>
        }
      // CHECK: spirv.Return
      gpu.return
    }
  }

  func.func @main() {
    %0 = "op"() : () -> (memref<10xf32, #spirv.storage_class<StorageBuffer>>)
    %1 = "op"() : () -> (memref<10xf32, #spirv.storage_class<StorageBuffer>>)
    %cst = arith.constant 1 : index
    gpu.launch_func @kernels::@loop_kernel
        blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)
        args(%0 : memref<10xf32, #spirv.storage_class<StorageBuffer>>, %1 : memref<10xf32, #spirv.storage_class<StorageBuffer>>)
    return
  }
}
