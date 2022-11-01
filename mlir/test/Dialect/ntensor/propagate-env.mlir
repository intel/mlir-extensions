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
//       CHECK: %[[RES1:.*]] = ntensor.primitive "foo" (%[[ARG]]) : !ntensor.ntensor<?xf32, "test"> -> !ntensor.ntensor<?xf32, "test">
//       CHECK: %[[RES2:.*]] = ntensor.cast %[[RES1]] : !ntensor.ntensor<?xf32, "test"> to !ntensor.ntensor<?xf32>
//       CHECK: return %[[RES2]]
func.func @test(%arg1: !ntensor.ntensor<?xf32, "test">) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.cast %arg1 : !ntensor.ntensor<?xf32, "test"> to !ntensor.ntensor<?xf32>
  %1 = ntensor.primitive "foo" (%0) : !ntensor.ntensor<?xf32> -> !ntensor.ntensor<?xf32>
  return %1 : !ntensor.ntensor<?xf32>
}

// -----

// CHECK-LABEL: func @test
//       CHECK: %[[RES1:.*]] = scf.if %{{.*}} -> (!ntensor.ntensor<?xf32>)
//       CHECK: %[[RES2:.*]] = ntensor.cast %[[RES1]] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32, "test1">
//       CHECK: %[[RES3:.*]] = ntensor.primitive "foo" (%[[RES2]]) : !ntensor.ntensor<?xf32, "test1"> -> !ntensor.ntensor<?xf32, "test1">
//       CHECK: %[[RES4:.*]] = ntensor.cast %[[RES3]] : !ntensor.ntensor<?xf32, "test1"> to !ntensor.ntensor<?xf32>
//       CHECK: return %[[RES4]]
func.func @test(%arg1: !ntensor.ntensor<?xf32, "test1">, %arg2: !ntensor.ntensor<?xf32, "test1">, %arg3: i1) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.cast %arg1 : !ntensor.ntensor<?xf32, "test1"> to !ntensor.ntensor<?xf32>
  %1 = ntensor.cast %arg2 : !ntensor.ntensor<?xf32, "test1"> to !ntensor.ntensor<?xf32>
  %3 = scf.if %arg3 -> !ntensor.ntensor<?xf32> {
    scf.yield %0 : !ntensor.ntensor<?xf32>
  } else {
    scf.yield %1 : !ntensor.ntensor<?xf32>
  }
  %4 = ntensor.primitive "foo" (%3) : !ntensor.ntensor<?xf32> -> !ntensor.ntensor<?xf32>
  return %4 : !ntensor.ntensor<?xf32>
}

// -----

// CHECK-LABEL: func @test
//       CHECK: %[[RES1:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : !ntensor.ntensor<?xf32>
//       CHECK: %[[RES2:.*]] = ntensor.cast %[[RES1]] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32, "test1">
//       CHECK: %[[RES3:.*]] = ntensor.primitive "foo" (%[[RES2]]) : !ntensor.ntensor<?xf32, "test1"> -> !ntensor.ntensor<?xf32, "test1">
//       CHECK: %[[RES4:.*]] = ntensor.cast %[[RES3]] : !ntensor.ntensor<?xf32, "test1"> to !ntensor.ntensor<?xf32>
//       CHECK: return %[[RES4]]
func.func @test(%arg1: !ntensor.ntensor<?xf32, "test1">, %arg2: !ntensor.ntensor<?xf32, "test1">, %arg3: i1) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.cast %arg1 : !ntensor.ntensor<?xf32, "test1"> to !ntensor.ntensor<?xf32>
  %1 = ntensor.cast %arg2 : !ntensor.ntensor<?xf32, "test1"> to !ntensor.ntensor<?xf32>
  %3 = arith.select %arg3, %0, %1 : !ntensor.ntensor<?xf32>
  %4 = ntensor.primitive "foo" (%3) : !ntensor.ntensor<?xf32> -> !ntensor.ntensor<?xf32>
  return %4 : !ntensor.ntensor<?xf32>
}
