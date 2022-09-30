// RUN: imex-opt -allow-unregistered-dialect -convert-gpux-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
    gpu.container_module,
    spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spv.resource_limits<>>
} {
  gpu.module @kernels {
    // CHECK:       spv.module @{{.*}} Logical GLSL450 {
    // CHECK-LABEL: spv.func @loop_kernel
    // CHECK-SAME: {{%.*}}: !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer> {spv.interface_var_abi = #spv.interface_var_abi<(0, 0)>}
    // CHECK-SAME: {{%.*}}: !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer> {spv.interface_var_abi = #spv.interface_var_abi<(0, 1)>}
    // CHECK-SAME: spv.entry_point_abi = #spv.entry_point_abi<local_size = dense<[32, 4, 1]> : vector<3xi32>>
    gpu.func @loop_kernel(%arg2 : memref<10xf32, #spv.storage_class<StorageBuffer>>, %arg3 : memref<10xf32, #spv.storage_class<StorageBuffer>>) kernel
      attributes {spv.entry_point_abi = #spv.entry_point_abi<local_size = dense<[32, 4, 1]>: vector<3xi32>>} {
      // CHECK: spv.Branch ^bb1
      // CHECK: ^bb1:  // pred: ^bb0
       cf.br ^bb1
      ^bb1:  // pred: ^bb0
      // CHECK: %[[LB:.*]] = spv.Constant 4 : i32
        %lb = arith.constant 4 : index
        // CHECK: %[[UB:.*]] = spv.Constant 42 : i32
        %ub = arith.constant 42 : index
        // CHECK: %[[STEP:.*]] = spv.Constant 2 : i32
        %step = arith.constant 2 : index
        // CHECK: spv.mlir.loop {
        // CHECK-NEXT: spv.Branch ^[[HEADER:.*]](%[[LB]] : i32)
        // CHECK:      ^[[HEADER]](%[[INDVAR:.*]]: i32):
        // CHECK:        %[[CMP:.*]] = spv.SLessThan %[[INDVAR]], %[[UB]] : i32
        // CHECK:        spv.BranchConditional %[[CMP]], ^[[BODY:.*]], ^[[MERGE:.*]]
        // CHECK:      ^[[BODY]]:
        // CHECK:        %[[ZERO1:.*]] = spv.Constant 0 : i32
        // CHECK:        %[[OFFSET1:.*]] = spv.Constant 0 : i32
        // CHECK:        %[[STRIDE1:.*]] = spv.Constant 1 : i32
        // CHECK:        %[[UPDATE1:.*]] = spv.IMul %[[STRIDE1]], %[[INDVAR]] : i32
        // CHECK:        %[[INDEX1:.*]] = spv.IAdd %[[OFFSET1]], %[[UPDATE1]] : i32
        // CHECK:        spv.AccessChain {{%.*}}{{\[}}%[[ZERO1]], %[[INDEX1]]{{\]}}
        // CHECK:        %[[ZERO2:.*]] = spv.Constant 0 : i32
        // CHECK:        %[[OFFSET2:.*]] = spv.Constant 0 : i32
        // CHECK:        %[[STRIDE2:.*]] = spv.Constant 1 : i32
        // CHECK:        %[[UPDATE2:.*]] = spv.IMul %[[STRIDE2]], %[[INDVAR]] : i32
        // CHECK:        %[[INDEX2:.*]] = spv.IAdd %[[OFFSET2]], %[[UPDATE2]] : i32
        // CHECK:        spv.AccessChain {{%.*}}[%[[ZERO2]], %[[INDEX2]]]
        // CHECK:        %[[INCREMENT:.*]] = spv.IAdd %[[INDVAR]], %[[STEP]] : i32
        // CHECK:        spv.Branch ^[[HEADER]](%[[INCREMENT]] : i32)
        // CHECK:      ^[[MERGE]]
        // CHECK:        spv.mlir.merge
        // CHECK:      }
        scf.for %arg4 = %lb to %ub step %step {
             %1 = memref.load %arg2[%arg4] : memref<10xf32, #spv.storage_class<StorageBuffer>>
            memref.store %1, %arg3[%arg4] : memref<10xf32, #spv.storage_class<StorageBuffer>>
        }
      // CHECK: spv.Return
      gpu.return
    }
  }

  func.func @main() {
    %0 = "op"() : () -> (memref<10xf32, #spv.storage_class<StorageBuffer>>)
    %1 = "op"() : () -> (memref<10xf32, #spv.storage_class<StorageBuffer>>)
    %cst = arith.constant 1 : index
    gpu.launch_func @kernels::@loop_kernel
        blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)
        args(%0 : memref<10xf32, #spv.storage_class<StorageBuffer>>, %1 : memref<10xf32, #spv.storage_class<StorageBuffer>>)
    return
  }
}
