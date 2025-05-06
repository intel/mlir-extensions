// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

// CHECK-LABEL: func @alloc_nbarrier({{.*}}) {
func.func @alloc_nbarrier() {
  // CHECK: xegpu.alloc_nbarrier
  xegpu.alloc_nbarrier 8
  return
}

// CHECK-LABEL: func @init_nbarrier({{.*}}) {
func.func @init_nbarrier() {
  %nbarrier_id = arith.constant 1 : i8
  %nbarrier_role = arith.constant 0 : i8
  // CHECK: xegpu.init_nbarrier
  // CHECK-SAME: i8, i8 -> !xegpu.nbarrier
  %nbarrier = xegpu.init_nbarrier %nbarrier_id, %nbarrier_role : i8, i8 -> !xegpu.nbarrier
  return
}

// CHECK-LABEL: func @nbarrier_arrive({{.*}}) {
func.func @nbarrier_arrive(%nbarrier : !xegpu.nbarrier) {
  // CHECK:  xegpu.nbarrier_arrive
  // CHECK-SAME: !xegpu.nbarrier
  xegpu.nbarrier_arrive %nbarrier : !xegpu.nbarrier
  return
}

// CHECK-LABEL: func @nbarrier_wait({{.*}}) {
func.func @nbarrier_wait(%nbarrier : !xegpu.nbarrier) {
  // CHECK: xegpu.nbarrier_wait
  // CHECK-SAME: !xegpu.nbarrier
  xegpu.nbarrier_wait %nbarrier : !xegpu.nbarrier
  return
}

// CHECK-LABEL: func @fence({{.*}}) {
func.func @fence() {
  // CHECK: xegpu.fence memory_kind = global, fence_scope = workgroup
  xegpu.fence memory_kind = global, fence_scope = workgroup
  return
}

// CHECK-LABEL: func @fence_1({{.*}}) {
func.func @fence_1() {
  // CHECK: xegpu.fence memory_kind = global, fence_scope = tile
  xegpu.fence memory_kind = global, fence_scope = tile
  return
}

// CHECK-LABEL: func @fence_2({{.*}}) {
func.func @fence_2() {
  // CHECK: xegpu.fence memory_kind = global, fence_scope = tile, fence_op_flush = evict
  xegpu.fence memory_kind = global, fence_scope = tile, fence_op_flush = evict
  return
}