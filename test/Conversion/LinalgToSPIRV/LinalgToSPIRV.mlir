// RUN: imex-opt -allow-unregistered-dialect -imex-convert-linalg-to-spirv %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: func.func @simple_constant
func.func @simple_constant() -> (i32) {
  // CHECK-NEXT: %[[RESULT:.*]] = arith.constant 1
  // CHECK-NEXT: return %[[RESULT]]

  %0 = arith.constant 1 : i32
  return %0 : i32
}
