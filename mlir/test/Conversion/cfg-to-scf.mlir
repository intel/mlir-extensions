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

// -----

func.func @test() {
  %cond = "test.test1"() : () -> i1
  cf.cond_br %cond, ^bb1, ^bb3
^bb1:
  "test.test2"() : () -> ()
  cf.br ^bb3
^bb3:
  "test.test4"() : () -> ()
  return
}

// CHECK-LABEL: func @test
// CHECK: %[[COND:.*]] = "test.test1"() : () -> i1
// CHECK: scf.if %[[COND]] {
// CHECK: "test.test2"() : () -> ()
// CHECK: }
// CHECK: "test.test4"() : () -> ()
// CHECK: return

// -----

func.func @test() {
  %cond = "test.test1"() : () -> i1
  cf.cond_br %cond, ^bb3, ^bb2
^bb2:
  "test.test3"() : () -> ()
  cf.br ^bb3
^bb3:
  "test.test4"() : () -> ()
  return
}

// CHECK-LABEL: func @test
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[COND:.*]] = "test.test1"() : () -> i1
// CHECK: %[[NCOND:.*]] = arith.xori %[[COND]], %[[TRUE]] : i1
// CHECK: scf.if %[[NCOND]] {
// CHECK: "test.test3"() : () -> ()
// CHECK: }
// CHECK: "test.test4"() : () -> ()
// CHECK: return

// -----

func.func @test() -> index {
  %cond = "test.test1"() : () -> i1
  %2 = "test.test3"() : () -> (index)
  cf.cond_br %cond, ^bb1, ^bb3(%2: index)
^bb1:
  %1 = "test.test2"() : () -> (index)
  cf.br ^bb3(%1: index)
^bb3(%3: index):
  "test.test4"() : () -> ()
  return %3 : index
}

// CHECK-LABEL: func @test
// CHECK: %[[COND:.*]] = "test.test1"() : () -> i1
// CHECK: %[[VAL2:.*]]  = "test.test3"() : () -> index
// CHECK: %[[RES:.*]] = scf.if %[[COND]] -> (index) {
// CHECK: %[[VAL1:.*]]  = "test.test2"() : () -> index
// CHECK: scf.yield %[[VAL1]] : index
// CHECK: } else {
// CHECK: scf.yield %[[VAL2]] : index
// CHECK: }
// CHECK: "test.test4"() : () -> ()
// CHECK: return %[[RES]] : index

// -----

func.func @test() -> index {
  %cond = "test.test1"() : () -> i1
  %1 = "test.test2"() : () -> (index)
  %2 = "test.test3"() : () -> (index)
  cf.cond_br %cond, ^bb1(%1: index), ^bb1(%2: index)
^bb1(%3: index):
  "test.test4"() : () -> ()
  return %3 : index
}

// CHECK-LABEL: func @test
// CHECK: %[[COND:.*]] = "test.test1"() : () -> i1
// CHECK: %[[VAL1:.*]]  = "test.test2"() : () -> index
// CHECK: %[[VAL2:.*]]  = "test.test3"() : () -> index
// CHECK: %[[RES:.*]]  = arith.select %[[COND]], %[[VAL1]], %[[VAL2]] : index
// CHECK: "test.test4"() : () -> ()
// CHECK: return %[[RES]] : index

// -----

func.func @test() {
  "test.test1"() : () -> ()
  cf.br ^bb1
^bb1:
  %cond = "test.test2"() : () -> i1
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  "test.test3"() : () -> ()
  cf.br ^bb1
^bb3:
  "test.test4"() : () -> ()
  return
}

// CHECK-LABEL: func @test
// CHECK: "test.test1"() : () -> ()
// CHECK: scf.while : () -> () {
// CHECK: %[[COND:.*]] = "test.test2"() : () -> i1
// CHECK: scf.condition(%[[COND]])
// CHECK: } do {
// CHECK: "test.test3"() : () -> ()
// CHECK: scf.yield
// CHECK: }
// CHECK: return

// -----

func.func @test() {
  "test.test1"() : () -> ()
  cf.br ^bb1
^bb1:
  %cond = "test.test2"() : () -> i1
  cf.cond_br %cond, ^bb3, ^bb2
^bb2:
  "test.test3"() : () -> ()
  cf.br ^bb1
^bb3:
  "test.test4"() : () -> ()
  return
}

// CHECK-LABEL: func @test
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: "test.test1"() : () -> ()
// CHECK: scf.while : () -> () {
// CHECK: %[[COND:.*]] = "test.test2"() : () -> i1
// CHECK: %[[XCOND:.*]] = arith.xori %[[COND]], %[[TRUE]] : i1
// CHECK: scf.condition(%[[XCOND]])
// CHECK: } do {
// CHECK: "test.test3"() : () -> ()
// CHECK: scf.yield
// CHECK: }
// CHECK: return

// -----

func.func @test() -> i64{
  %1 = "test.test1"() : () -> index
  cf.br ^bb1(%1: index)
^bb1(%2: index):
  %cond:2 = "test.test2"() : () -> (i1, i64)
  cf.cond_br %cond#0, ^bb2(%cond#1: i64), ^bb3(%cond#1: i64)
^bb2(%3: i64):
  %4 = "test.test3"(%3) : (i64) -> index
  cf.br ^bb1(%4: index)
^bb3(%5: i64):
  "test.test4"() : () -> ()
  return %5 : i64
}

// CHECK-LABEL: func @test
// CHECK: %[[VAL1:.*]] = "test.test1"() : () -> index
// CHECK: %[[RES:.*]]:2 = scf.while (%[[ARG0:.*]] = %[[VAL1]]) : (index) -> (index, i64) {
// CHECK: %[[COND:.*]]:2 = "test.test2"() : () -> (i1, i64)
// CHECK: scf.condition(%[[COND]]#0) %[[ARG0]], %[[COND]]#1 : index, i64
// CHECK: } do {
// CHECK: ^bb{{.+}}(%[[ARG1:.*]]: index, %[[ARG2:.*]]: i64):
// CHECK: %[[VAL2:.*]] = "test.test3"(%[[ARG2]]) : (i64) -> index
// CHECK: scf.yield %[[VAL2]] : index
// CHECK: }
// CHECK: "test.test4"() : () -> ()
// CHECK: return %[[RES]]#1 : i64

// -----

func.func @test() {
  "test.test1"() : () -> ()
  cf.br ^bb1
^bb1:
  %cond = "test.test2"() : () -> i1
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  %cond2 = "test.test3"() : () -> i1
  cf.cond_br %cond2, ^bb3, ^bb1
^bb3:
  "test.test4"() : () -> ()
  return
}

// CHECK-LABEL: func @test
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: "test.test1"() : () -> ()
// CHECK: %{{.*}} = scf.while (%[[ARG0:.*]] = %[[TRUE]]) : (i1) -> i1 {
// CHECK: %[[COND:.*]] = "test.test2"() : () -> i1
// CHECK: %[[ACOND:.*]] = arith.andi %[[ARG0]], %[[COND]] : i1
// CHECK: scf.condition(%[[ACOND]]) %[[ARG0]] : i1
// CHECK: } do {
// CHECK: %[[COND2:.*]] = "test.test3"() : () -> i1
// CHECK: %[[XCOND2:.*]] = arith.xori %[[COND2]], %[[TRUE]] : i1
// CHECK: scf.yield %[[XCOND2]] : i1
// CHECK: }
// CHECK: return

// -----

func.func @test() {
  "test.test1"() : () -> ()
  cf.br ^bb1
^bb1:
  %cond = "test.test2"() : () -> i1
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  %cond2 = "test.test3"() : () -> i1
  cf.cond_br %cond2, ^bb1, ^bb3
^bb3:
  "test.test4"() : () -> ()
  return
}

// CHECK-LABEL: func @test
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: "test.test1"() : () -> ()
// CHECK: %{{.*}} = scf.while (%[[ARG0:.*]] = %[[TRUE]]) : (i1) -> i1 {
// CHECK: %[[COND:.*]] = "test.test2"() : () -> i1
// CHECK: %[[ACOND:.*]] = arith.andi %[[ARG0]], %[[COND]] : i1
// CHECK: scf.condition(%[[ACOND]]) %[[ARG0]] : i1
// CHECK: } do {
// CHECK: %[[COND2:.*]] = "test.test3"() : () -> i1
// CHECK: scf.yield %[[COND2]] : i1
// CHECK: }
// CHECK: return
