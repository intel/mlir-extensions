// RUN: imex-opt --gpux-tile-parallel-loops --split-input-file %s | FileCheck %s

// CHECK-LABEL: check1D
// CHECK-SAME: (%[[MEM1:.*]]: memref<?xf64>, %[[MEM2:.*]]: memref<?xf64>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM1:.*]] = memref.dim %[[MEM1]], %[[C0]] : memref<?xf64>
// CHECK: imex_util.env_region #gpu_runtime.region_desc<device = "test">
// CHECK: %[[B:.*]]:3 = gpu_runtime.suggest_block_size, %[[DIM1]], %[[C1]], %[[C1]] -> index, index, index
// CHECK: %[[G1:.*]] = arith.ceildivui %[[DIM1]], %[[B]]#0 : index
// CHECK: scf.parallel
// CHECK-SAME: (%[[ARG1:.*]], %[[ARG2:.*]], %[[ARG3:.*]], %[[ARG4:.*]], %[[ARG5:.*]], %[[ARG6:.*]]) =
// CHECK-SAME: (%[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]]) to
// CHECK-SAME: (%[[G1]], %[[C1]], %[[C1]], %[[B]]#0, %[[C1]], %[[C1]]) step
// CHECK-SAME: (%[[C1]], %[[C1]], %[[C1]], %[[C1]], %[[C1]], %[[C1]])
// CHECK: %[[IDX1:.*]] = arith.muli %[[ARG1]], %[[B]]#0 : index
// CHECK: %[[IDX2:.*]] = arith.addi %[[IDX1]], %[[ARG4]] : index
// CHECK: %[[VAL:.*]] = memref.load %[[MEM1]][%[[IDX2]]] : memref<?xf64>
// CHECK: memref.store %[[VAL]], %[[MEM2]][%[[IDX2]]] : memref<?xf64>
// CHECK: {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = block_z, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = thread_z, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK: return
func.func @check1D(%arg0: memref<?xf64>, %arg1: memref<?xf64>) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?xf64>
  imex_util.env_region #gpu_runtime.region_desc<device = "test"> {
    scf.parallel (%arg4) = (%c0) to (%0) step (%c1) {
      %2 = memref.load %arg0[%arg4] : memref<?xf64>
      memref.store %2, %arg1[%arg4] : memref<?xf64>
      scf.yield
    }
  }
  return
}
