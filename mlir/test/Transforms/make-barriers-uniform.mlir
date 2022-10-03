// RUN: imex-opt -allow-unregistered-dialect --gpux-make-barriers-uniform --split-input-file %s | FileCheck %s

func.func @test() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
    %cond = "test.test1"() : () -> i1
    scf.if %cond {
      "test.test2"() : () -> ()
      %1 = "test.test3"() : () -> i32
      gpu_runtime.barrier  local
      "test.test4"() : () -> ()
      "test.test5"(%1) : (i32) -> ()
    }
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func @test
//       CHECK: gpu.launch blocks
//  CHECK-NEXT: %[[COND:.*]] = "test.test1"() : () -> i1
//  CHECK-NEXT: %[[RES1:.*]] = scf.if %[[COND]] -> (i32) {
//  CHECK-NEXT: "test.test2"() : () -> ()
//  CHECK-NEXT: %[[V1:.*]] = "test.test3"() : () -> i32
//  CHECK-NEXT: scf.yield %[[V1]] : i32
//  CHECK-NEXT: } else {
//  CHECK-NEXT: %[[V2:.*]] = imex_util.undef : i32
//  CHECK-NEXT: scf.yield %[[V2]] : i32
//  CHECK-NEXT: }
//  CHECK-NEXT: gpu_runtime.barrier  local
//  CHECK-NEXT: scf.if %[[COND]] {
//  CHECK-NEXT: "test.test4"() : () -> ()
//  CHECK-NEXT: "test.test5"(%[[RES1]]) : (i32) -> ()
//  CHECK-NEXT: }
//  CHECK-NEXT: gpu.terminator
//        NEXT: return
