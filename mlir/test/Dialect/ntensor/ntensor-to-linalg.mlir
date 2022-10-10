// RUN: imex-opt %s -ntensor-to-linalg --split-input-file | FileCheck %s

func.func @test() -> !ntensor.ntensor<?x?xf32> {
  %0 = arith.constant 2 : index
  %1 = arith.constant 3 : index
  %3 = ntensor.create(%0, %1) : !ntensor.ntensor<?x?xf32>
  return %3 : !ntensor.ntensor<?x?xf32>
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   %[[D1:.*]] = arith.constant 2 : index
//  CHECK-NEXT:   %[[D2:.*]] = arith.constant 3 : index
//  CHECK-NEXT:   %[[RES:.*]] = tensor.empty(%[[D1]], %[[D2]]) : tensor<?x?xf32>
//  CHECK-NEXT:   %[[RES1:.*]] = ntensor.from_tensor %[[RES]] : tensor<?x?xf32> to !ntensor.ntensor<?x?xf32>
//  CHECK-NEXT:   return %[[RES1]]

// -----

func.func @test() -> !ntensor.ntensor<?x?xf32, "test"> {
  %0 = arith.constant 2 : index
  %1 = arith.constant 3 : index
  %3 = ntensor.create(%0, %1) : !ntensor.ntensor<?x?xf32, "test">
  return %3 : !ntensor.ntensor<?x?xf32, "test">
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   %[[D1:.*]] = arith.constant 2 : index
//  CHECK-NEXT:   %[[D2:.*]] = arith.constant 3 : index
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.env_region "test" -> !ntensor.ntensor<?x?xf32, "test"> {
//  CHECK-NEXT:   %[[RES1:.*]] = tensor.empty(%[[D1]], %[[D2]]) : tensor<?x?xf32>
//  CHECK-NEXT:   %[[RES2:.*]] = ntensor.from_tensor %[[RES1]] : tensor<?x?xf32> to !ntensor.ntensor<?x?xf32, "test">
//  CHECK-NEXT:   imex_util.env_region_yield %[[RES2]] : !ntensor.ntensor<?x?xf32, "test">
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]]

// -----

func.func @test() -> !ntensor.ntensor<?x?xi32> {
  %0 = arith.constant 2 : index
  %1 = arith.constant 3 : index
  %2 = arith.constant 5 : i32
  %3 = ntensor.create(%0, %1) = (%2 : i32) : !ntensor.ntensor<?x?xi32>
  return %3 : !ntensor.ntensor<?x?xi32>
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   %[[D1:.*]] = arith.constant 2 : index
//  CHECK-NEXT:   %[[D2:.*]] = arith.constant 3 : index
//  CHECK-NEXT:   %[[VAL:.*]] = arith.constant 5 : i32
//  CHECK-NEXT:   %[[RES:.*]] = tensor.empty(%[[D1]], %[[D2]]) : tensor<?x?xi32>
//  CHECK-NEXT:   %[[RES1:.*]] = linalg.fill ins(%[[VAL]] : i32) outs(%[[RES]] : tensor<?x?xi32>) -> tensor<?x?xi32>
//  CHECK-NEXT:   %[[RES2:.*]] = ntensor.from_tensor %[[RES1]] : tensor<?x?xi32> to !ntensor.ntensor<?x?xi32>
//  CHECK-NEXT:   return %[[RES2]]

// -----

func.func @test() -> !ntensor.ntensor<?x?xi32, "test"> {
  %0 = arith.constant 2 : index
  %1 = arith.constant 3 : index
  %2 = arith.constant 5 : i32
  %3 = ntensor.create(%0, %1) = (%2 : i32) : !ntensor.ntensor<?x?xi32, "test">
  return %3 : !ntensor.ntensor<?x?xi32, "test">
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   %[[D1:.*]] = arith.constant 2 : index
//  CHECK-NEXT:   %[[D2:.*]] = arith.constant 3 : index
//  CHECK-NEXT:   %[[VAL:.*]] = arith.constant 5 : i32
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.env_region "test" -> !ntensor.ntensor<?x?xi32, "test"> {
//  CHECK-NEXT:   %[[RES1:.*]] = tensor.empty(%[[D1]], %[[D2]]) : tensor<?x?xi32>
//  CHECK-NEXT:   %[[RES2:.*]] = linalg.fill ins(%[[VAL]] : i32) outs(%[[RES1]] : tensor<?x?xi32>) -> tensor<?x?xi32>
//  CHECK-NEXT:   %[[RES3:.*]] = ntensor.from_tensor %[[RES2]] : tensor<?x?xi32> to !ntensor.ntensor<?x?xi32, "test">
//  CHECK-NEXT:   imex_util.env_region_yield %[[RES3]] : !ntensor.ntensor<?x?xi32, "test">
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]]
