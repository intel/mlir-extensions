// RUN: imex-opt --ntensor-propagate-env --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//       CHECK: %[[RES:.*]] = ntensor.primitive "foo" (%[[ARG]]) : !ntensor.ntensor<?xf32> -> !ntensor.ntensor<?xf32>
//       CHECK: return %[[RES]]
func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.primitive "foo" (%arg1) : !ntensor.ntensor<?xf32> -> !ntensor.ntensor<?xf32>
  return %0 : !ntensor.ntensor<?xf32>
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG:.*]]: !ntensor.ntensor<?xf32, "test">)
//       CHECK: %[[RES:.*]] = ntensor.primitive "foo" (%[[ARG]]) : !ntensor.ntensor<?xf32> -> !ntensor.ntensor<?xf32>
//       CHECK: return %[[RES]]
func.func @test(%arg1: !ntensor.ntensor<?xf32, "test">) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.cast %arg1 : !ntensor.ntensor<?xf32, "test"> to !ntensor.ntensor<?xf32>
  %1 = ntensor.primitive "foo" (%0) : !ntensor.ntensor<?xf32> -> !ntensor.ntensor<?xf32>
  return %1 : !ntensor.ntensor<?xf32>
}
