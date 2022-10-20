// RUN: imex-opt -allow-unregistered-dialect --cfg-to-scf -split-input-file %s | FileCheck %s

func.func @if_test1() {
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

// CHECK-LABEL: func @if_test1
// CHECK: %[[COND:.*]] = "test.test1"() : () -> i1
// CHECK: scf.if %[[COND]] {
// CHECK: "test.test2"() : () -> ()
// CHECK: } else {
// CHECK: "test.test3"() : () -> ()
// CHECK: }
// CHECK: "test.test4"() : () -> ()
// CHECK: return

// -----

func.func @if_test2() -> index {
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

// CHECK-LABEL: func @if_test2
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

func.func @if_test3() {
  %cond = "test.test1"() : () -> i1
  cf.cond_br %cond, ^bb1, ^bb3
^bb1:
  "test.test2"() : () -> ()
  cf.br ^bb3
^bb3:
  "test.test4"() : () -> ()
  return
}

// CHECK-LABEL: func @if_test3
// CHECK: %[[COND:.*]] = "test.test1"() : () -> i1
// CHECK: scf.if %[[COND]] {
// CHECK: "test.test2"() : () -> ()
// CHECK: }
// CHECK: "test.test4"() : () -> ()
// CHECK: return

// -----

func.func @if_test4() {
  %cond = "test.test1"() : () -> i1
  cf.cond_br %cond, ^bb3, ^bb2
^bb2:
  "test.test3"() : () -> ()
  cf.br ^bb3
^bb3:
  "test.test4"() : () -> ()
  return
}

// CHECK-LABEL: func @if_test4
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

func.func @select_test() -> index {
  %cond = "test.test1"() : () -> i1
  %1 = "test.test2"() : () -> (index)
  %2 = "test.test3"() : () -> (index)
  cf.cond_br %cond, ^bb1(%1: index), ^bb1(%2: index)
^bb1(%3: index):
  "test.test4"() : () -> ()
  return %3 : index
}

// CHECK-LABEL: func @select_test
// CHECK: %[[COND:.*]] = "test.test1"() : () -> i1
// CHECK: %[[VAL1:.*]]  = "test.test2"() : () -> index
// CHECK: %[[VAL2:.*]]  = "test.test3"() : () -> index
// CHECK: %[[RES:.*]]  = arith.select %[[COND]], %[[VAL1]], %[[VAL2]] : index
// CHECK: "test.test4"() : () -> ()
// CHECK: return %[[RES]] : index

// -----

func.func @while_test1() {
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

// CHECK-LABEL: func @while_test1
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

func.func @while_test2() {
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

// CHECK-LABEL: func @while_test2
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

func.func @while_test3() -> i64{
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

// CHECK-LABEL: func @while_test3
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

func.func @break_test1() {
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

// CHECK-LABEL: func @break_test1
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
// CHECK: "test.test4"() : () -> ()
// CHECK: return

// -----

func.func @break_test2() {
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

// CHECK-LABEL: func @break_test2
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
// CHECK: "test.test4"() : () -> ()
// CHECK: return

// -----

func.func @break_test3() {
  "test.test1"() : () -> ()
  cf.br ^bb1
^bb1:
  %cond = "test.test2"() : () -> i1
  cf.cond_br %cond, ^bb3, ^bb2
^bb2:
  %cond2 = "test.test3"() : () -> i1
  cf.cond_br %cond2, ^bb3, ^bb1
^bb3:
  "test.test4"() : () -> ()
  return
}

// CHECK-LABEL: func @break_test3
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: "test.test1"() : () -> ()
// CHECK: %{{.*}} = scf.while (%[[ARG0:.*]] = %[[TRUE]]) : (i1) -> i1 {
// CHECK: %[[COND:.*]] = "test.test2"() : () -> i1
// CHECK: %[[ACOND:.*]] = arith.andi %[[ARG0]], %[[COND]] : i1
// CHECK: %[[XCOND:.*]] = arith.xori %[[ACOND]], %[[TRUE]] : i1
// CHECK: scf.condition(%[[XCOND]]) %[[ARG0]] : i1
// CHECK: } do {
// CHECK: %[[COND2:.*]] = "test.test3"() : () -> i1
// CHECK: %[[XCOND2:.*]] = arith.xori %[[COND2]], %[[TRUE]] : i1
// CHECK: scf.yield %[[XCOND2]] : i1
// CHECK: }
// CHECK: return

// -----

func.func @if_break_test1() {
  "test.test1"() : () -> ()
  cf.br ^bb1
^bb1:
  %cond = "test.test2"() : () -> i1
  cf.cond_br %cond, ^bb2, ^bb4
^bb2:
  %cond2 = "test.test3"() : () -> i1
  cf.cond_br %cond2, ^bb3, ^bb4
^bb3:
  "test.test4"() : () -> ()
  cf.br ^bb1
^bb4:
  "test.test5"() : () -> ()
  return
}

// CHECK-LABEL: func @if_break_test1
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: "test.test1"() : () -> ()
// CHECK: scf.while : () -> () {
// CHECK: %[[COND1:.*]] = "test.test2"() : () -> i1
// CHECK: %[[COND2:.*]] = scf.if %[[COND1]] -> (i1) {
// CHECK: %[[VAL1:.*]] = "test.test3"() : () -> i1
// CHECK: scf.yield %[[VAL1]] : i1
// CHECK: } else {
// CHECK: scf.yield %[[FALSE]] : i1
// CHECK: }
// CHECK: %[[COND3:.*]] = arith.andi %[[COND1]], %[[COND2]] : i1
// CHECK: scf.condition(%[[COND3]])
// CHECK: } do {
// CHECK: "test.test4"() : () -> ()
// CHECK: scf.yield
// CHECK: }
// CHECK: "test.test5"() : () -> ()
// CHECK: return

// -----

func.func @if_break_test2() {
  "test.test1"() : () -> ()
  cf.br ^bb1
^bb1:
  %cond = "test.test2"() : () -> i1
  cf.cond_br %cond, ^bb4, ^bb2
^bb2:
  %cond2 = "test.test3"() : () -> i1
  cf.cond_br %cond2, ^bb3, ^bb4
^bb3:
  "test.test4"() : () -> ()
  cf.br ^bb1
^bb4:
  "test.test5"() : () -> ()
  return
}

// CHECK-LABEL: func @if_break_test2
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: "test.test1"() : () -> ()
// CHECK: scf.while : () -> () {
// CHECK: %[[COND1:.*]] = "test.test2"() : () -> i1
// CHECK: %[[XCOND1:.*]] = arith.xori %[[COND1]], %true : i1
// CHECK: %[[COND2:.*]] = scf.if %[[COND1]] -> (i1) {
// CHECK: scf.yield %[[FALSE]] : i1
// CHECK: } else {
// CHECK: %[[VAL1:.*]] = "test.test3"() : () -> i1
// CHECK: scf.yield %[[VAL1]] : i1
// CHECK: }
// CHECK: %[[COND3:.*]] = arith.andi %[[XCOND1]], %[[COND2]] : i1
// CHECK: scf.condition(%[[COND3]])
// CHECK: } do {
// CHECK: "test.test4"() : () -> ()
// CHECK: scf.yield
// CHECK: }
// CHECK: "test.test5"() : () -> ()
// CHECK: return

// -----

func.func @if_break_test3() {
  "test.test1"() : () -> ()
  cf.br ^bb1
^bb1:
  %cond = "test.test2"() : () -> i1
  cf.cond_br %cond, ^bb2, ^bb4
^bb2:
  %cond2 = "test.test3"() : () -> i1
  cf.cond_br %cond2, ^bb4, ^bb3
^bb3:
  "test.test4"() : () -> ()
  cf.br ^bb1
^bb4:
  "test.test5"() : () -> ()
  return
}

// CHECK-LABEL: func @if_break_test3
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: "test.test1"() : () -> ()
// CHECK: scf.while : () -> () {
// CHECK: %[[COND1:.*]] = "test.test2"() : () -> i1
// CHECK: %[[COND2:.*]] = scf.if %[[COND1]] -> (i1) {
// CHECK: %[[VAL1:.*]] = "test.test3"() : () -> i1
// CHECK: %[[VAL2:.*]]  = arith.xori %[[VAL1]], %true : i1
// CHECK: scf.yield %[[VAL2]] : i1
// CHECK: } else {
// CHECK: scf.yield %[[FALSE]] : i1
// CHECK: }
// CHECK: %[[COND3:.*]] = arith.andi %[[COND1]], %[[COND2]] : i1
// CHECK: scf.condition(%[[COND3]])
// CHECK: } do {
// CHECK: "test.test4"() : () -> ()
// CHECK: scf.yield
// CHECK: }
// CHECK: "test.test5"() : () -> ()
// CHECK: return

// -----

func.func @if_break_test4() {
  "test.test1"() : () -> ()
  cf.br ^bb1
^bb1:
  %cond:2 = "test.test2"() : () -> (i1, index)
  cf.cond_br %cond#0, ^bb2, ^bb4(%cond#1: index)
^bb2:
  %cond2:2 = "test.test3"() : () -> (i1, index)
  cf.cond_br %cond2#0, ^bb3, ^bb4(%cond2#1: index)
^bb3:
  "test.test4"() : () -> ()
  cf.br ^bb1
^bb4(%1: index):
  "test.test5"(%1) : (index) -> ()
  return
}

// CHECK-LABEL: func @if_break_test4
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: "test.test1"() : () -> ()
// CHECK: %[[RES:.*]] = scf.while : () -> index {
// CHECK: %[[COND1:.*]]:2 = "test.test2"() : () -> (i1, index)
// CHECK: %[[COND2:.*]]:2 = scf.if %[[COND1]]#0 -> (i1, index) {
// CHECK: %[[VAL1:.*]]:2 = "test.test3"() : () -> (i1, index)
// CHECK: scf.yield %[[VAL1]]#0, %[[VAL1]]#1 : i1, index
// CHECK: } else {
// CHECK: scf.yield %[[FALSE]], %[[COND1]]#1 : i1, index
// CHECK: }
// CHECK: %[[COND3:.*]] = arith.andi %[[COND1]]#0, %[[COND2]]#0 : i1
// CHECK: scf.condition(%[[COND3]])
// CHECK: } do {
// CHECK: "test.test4"() : () -> ()
// CHECK: scf.yield
// CHECK: }
// CHECK: "test.test5"(%[[RES]]) : (index) -> ()
// CHECK: return

// -----

func.func @if_break_test5() {
  "test.test1"() : () -> ()
  cf.br ^bb1
^bb1:
  %cond = "test.test2"() : () -> i1
  cf.cond_br %cond, ^bb2, ^bb4
^bb2:
  %cond2:2 = "test.test3"() : () -> (i1, index)
  cf.cond_br %cond2#0, ^bb3, ^bb4
^bb3:
  "test.test4"(%cond2#1) : (index) -> ()
  cf.br ^bb1
^bb4:
  "test.test5"() : () -> ()
  return
}

// CHECK-LABEL: func @if_break_test5
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: "test.test1"() : () -> ()
// CHECK: %[[RES:.*]]:2 = scf.while : () -> (i1, index) {
// CHECK: %[[COND1:.*]] = "test.test2"() : () -> i1
// CHECK: %[[COND2:.*]]:2 = scf.if %[[COND1]] -> (i1, index) {
// CHECK: %[[VAL1:.*]]:2 = "test.test3"() : () -> (i1, index)
// CHECK: scf.yield %[[VAL1]]#0, %[[VAL1]]#1 : i1, index
// CHECK: } else {
// CHECK: %[[VAL2:.*]] = imex_util.undef : index
// CHECK: scf.yield %[[FALSE]], %[[VAL2]] : i1, index
// CHECK: }
// CHECK: %[[COND3:.*]] = arith.andi %[[COND1]], %[[COND2]]#0 : i1
// CHECK: scf.condition(%[[COND3]])
// CHECK: } do {
// CHECK: ^bb{{.+}}(%[[ARG1:.*]]: i1, %[[ARG2:.*]]: index):
// CHECK: "test.test4"(%[[ARG2]]) : (index) -> ()
// CHECK: scf.yield
// CHECK: }
// CHECK: "test.test5"() : () -> ()
// CHECK: return
