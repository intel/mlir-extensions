// RUN: imex-opt -allow-unregistered-dialect --gpux-insert-global-reduce --split-input-file %s | FileCheck %s


func.func @test(%lb: index, %ub: index, %s: index) {
  %init = "test.init"() : () -> f32
  %0 = scf.parallel (%i, %j, %k) = (%lb, %lb, %lb) to (%ub, %ub, %ub) step (%s, %s, %s) init(%init) -> f32 {
    %1 = "test.produce"() : () -> f32
    scf.reduce(%1) : f32 {
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = "test.transform"(%lhs, %rhs) : (f32, f32) -> f32
      scf.reduce.return %2 : f32
    }
    scf.yield
  } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>,
                #gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
                #gpu.loop_dim_map<processor = block_z, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
  "test.consume"(%0) : (f32) -> ()
  return
}


// CHECK-LABEL: @test
//  CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[S:.*]]: index)
//       CHECK: %[[INIT:.*]] = "test.init"() : () -> f32
//       CHECK: %[[ARRAY:.*]] = memref.alloc() : memref<f32>
//       CHECK: scf.parallel (%[[ARG1:.*]], %[[ARG2:.*]], %[[ARG3:.*]]) = (%[[LB]], %[[LB]], %[[LB]]) to (%[[UB]], %[[UB]], %[[UB]]) step (%[[S]], %[[S]], %[[S]]) {
//       CHECK: %[[COND1:.*]] = arith.cmpi eq, %[[ARG1]], %[[LB]] : index
//       CHECK: %[[COND2:.*]] = arith.cmpi eq, %[[ARG2]], %[[LB]] : index
//       CHECK: %[[COND3:.*]] = arith.andi %[[COND1]], %[[COND2]] : i1
//       CHECK: %[[COND4:.*]] = arith.cmpi eq, %[[ARG3]], %[[LB]] : index
//       CHECK: %[[COND5:.*]] = arith.andi %[[COND3]], %[[COND4]] : i1
//       CHECK: %[[PROD:.*]] = "test.produce"() : () -> f32
//       CHECK: %[[RED:.*]] = gpu_runtime.global_reduce %[[PROD]] : f32 {
//       CHECK: ^bb0(%[[ARG4:.*]]: f32, %[[ARG5:.*]]: f32):
//       CHECK: %[[VAL:.*]] = "test.transform"(%[[ARG4]], %[[ARG5]]) : (f32, f32) -> f32
//       CHECK: gpu_runtime.global_reduce_yield %[[VAL]] : f32
//       CHECK: }
//       CHECK: scf.if %[[COND5]] {
//       CHECK: memref.store %[[RED]], %[[ARRAY]][] : memref<f32>
//       CHECK: }
//       CHECK: scf.yield
//       CHECK: }
//       CHECK: %[[RES1:.*]] = memref.load %[[ARRAY]][] : memref<f32>
//       CHECK: memref.dealloc %[[ARRAY]] : memref<f32>
//       CHECK: %[[RES2:.*]] = "test.transform"(%[[RES1]], %[[INIT]]) : (f32, f32) -> f32
//       CHECK: "test.consume"(%[[RES2]]) : (f32) -> ()
//       CHECK: return
