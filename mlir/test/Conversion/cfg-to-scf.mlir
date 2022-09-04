// RUN: imex-opt -allow-unregistered-dialect --cfg-to-scf -split-input-file %s | FileCheck %s

func.func @test() {
  %cond = "test.test1"() : () -> i1
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  "test.test2"() : () -> ()
  cf.br ^bb3
^bb2:
  "test.test3"() : () -> ()
  cf.br ^bb3
^bb3:
  "test.test4"() : () -> ()
  return
}

// CHECK-LABEL: func @test
// CHECK: %[[COND:.*]] = "test.test1"() : () -> i1
// CHECK: scf.if %[[COND]] {
// CHECK: "test.test2"() : () -> ()
// CHECK: } else {
// CHECK: "test.test3"() : () -> ()
// CHECK: }
// CHECK: "test.test4"() : () -> ()
// CHECK: return

// -----

func.func @test() -> index {
  %cond = "test.test1"() : () -> i1
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %1 = "test.test2"() : () -> (index)
  cf.br ^bb3(%1: index)
^bb2:
  %2 = "test.test3"() : () -> (index)
  cf.br ^bb3(%2: index)
^bb3(%3: index):
  "test.test4"() : () -> ()
  return %3 : index
}

// CHECK-LABEL: func @test
// CHECK: %[[COND:.*]] = "test.test1"() : () -> i1
// CHECK: %[[RES:.*]] = scf.if %[[COND]] -> (index) {
// CHECK: %[[VAL1:.*]]  = "test.test2"() : () -> index
// CHECK: scf.yield %[[VAL1]] : index
// CHECK: } else {
// CHECK: %[[VAL2:.*]]  = "test.test3"() : () -> index
// CHECK: scf.yield %[[VAL2]] : index
// CHECK: }
// CHECK: "test.test4"() : () -> ()
// CHECK: return %[[RES]] : index
