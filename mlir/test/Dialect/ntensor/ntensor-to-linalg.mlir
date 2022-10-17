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

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: !ntensor.ntensor<?xf32>) {
  ntensor.copy %arg1, %arg2 : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   %[[SRC:.*]] = ntensor.to_tensor %[[ARG1]] : !ntensor.ntensor<?xf32> to tensor<?xf32>
//  CHECK-NEXT:   %[[DST:.*]] = ntensor.to_memref %[[ARG2]] : !ntensor.ntensor<?xf32> to memref<?xf32>
//  CHECK-NEXT:   linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%[[SRC]] : tensor<?xf32>) outs(%[[DST]] : memref<?xf32>) {
//  CHECK-NEXT:   ^bb0(%[[BARG1:.*]]: f32, %[[ARG2:.*]]: f32):
//  CHECK-NEXT:   linalg.yield %[[BARG1]] : f32
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32, "test">, %arg2: !ntensor.ntensor<?xf32, "test">) {
  ntensor.copy %arg1, %arg2 : !ntensor.ntensor<?xf32, "test"> to !ntensor.ntensor<?xf32, "test">
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32, "test">, %[[ARG2:.*]]: !ntensor.ntensor<?xf32, "test">)
//  CHECK-NEXT:   imex_util.env_region "test" {
//  CHECK-NEXT:   %[[SRC:.*]] = ntensor.to_tensor %[[ARG1]] : !ntensor.ntensor<?xf32, "test"> to tensor<?xf32>
//  CHECK-NEXT:   %[[DST:.*]] = ntensor.to_memref %[[ARG2]] : !ntensor.ntensor<?xf32, "test"> to memref<?xf32>
//  CHECK-NEXT:   linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%[[SRC]] : tensor<?xf32>) outs(%[[DST]] : memref<?xf32>) {
//  CHECK-NEXT:   ^bb0(%[[BARG1:.*]]: f32, %[[ARG2:.*]]: f32):
//  CHECK-NEXT:   linalg.yield %[[BARG1]] : f32
//  CHECK-NEXT:   }
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: !ntensor.ntensor<?x5xf32>) -> !ntensor.ntensor<?x5xf32> {
  %0 = ntensor.elementwise %arg1 : !ntensor.ntensor<?x5xf32> -> !ntensor.ntensor<?x5xf32> {
  ^bb0(%arg2: f32):
    ntensor.elementwise_yield %arg2 : f32
  }
  return %0 : !ntensor.ntensor<?x5xf32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?x5xf32>)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[T1:.*]] = ntensor.to_tensor %[[ARG]] : !ntensor.ntensor<?x5xf32> to tensor<?x5xf32>
//  CHECK-NEXT:   %[[D:.*]] = tensor.dim %[[T1]], %[[C0]] : tensor<?x5xf32>
//  CHECK-NEXT:   %[[E:.*]] = tensor.empty(%[[D]]) : tensor<?x5xf32>
//  CHECK-NEXT:   %[[T2:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[T1]] : tensor<?x5xf32>) outs(%[[E]] : tensor<?x5xf32>) {
//  CHECK-NEXT:   ^bb0(%[[GARG1:.*]]: f32, %[[GARG2:.*]]: f32):
//  CHECK-NEXT:   linalg.yield %[[GARG1]] : f32
//  CHECK-NEXT:   } -> tensor<?x5xf32>
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.from_tensor %[[T2]] : tensor<?x5xf32> to !ntensor.ntensor<?x5xf32>
//  CHECK-NEXT:   return %[[RES]] : !ntensor.ntensor<?x5xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?x5xf32, "test">) -> !ntensor.ntensor<?x5xf32, "test"> {
  %0 = ntensor.elementwise %arg1 : !ntensor.ntensor<?x5xf32, "test"> -> !ntensor.ntensor<?x5xf32, "test"> {
  ^bb0(%arg2: f32):
    ntensor.elementwise_yield %arg2 : f32
  }
  return %0 : !ntensor.ntensor<?x5xf32, "test">
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?x5xf32, "test">)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[T0:.*]] = imex_util.env_region "test" -> !ntensor.ntensor<?x5xf32, "test"> {
//  CHECK-NEXT:   %[[T1:.*]] = ntensor.to_tensor %[[ARG]] : !ntensor.ntensor<?x5xf32, "test"> to tensor<?x5xf32>
//  CHECK-NEXT:   %[[D:.*]] = tensor.dim %[[T1]], %[[C0]] : tensor<?x5xf32>
//  CHECK-NEXT:   %[[E:.*]] = tensor.empty(%[[D]]) : tensor<?x5xf32>
//  CHECK-NEXT:   %[[T2:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[T1]] : tensor<?x5xf32>) outs(%[[E]] : tensor<?x5xf32>) {
//  CHECK-NEXT:   ^bb0(%[[GARG1:.*]]: f32, %[[GARG2:.*]]: f32):
//  CHECK-NEXT:   linalg.yield %[[GARG1]] : f32
//  CHECK-NEXT:   } -> tensor<?x5xf32>
//  CHECK-NEXT:   %[[RES:.*]] = ntensor.from_tensor %[[T2]] : tensor<?x5xf32> to !ntensor.ntensor<?x5xf32, "test">
//  CHECK-NEXT:   imex_util.env_region_yield %[[RES]] : !ntensor.ntensor<?x5xf32, "test">
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[T0]] : !ntensor.ntensor<?x5xf32, "test">


// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<5xf32> {
  %0 = ntensor.cast %arg1 : !ntensor.ntensor<?xf32> to !ntensor.ntensor<5xf32>
  return %0 : !ntensor.ntensor<5xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   %[[VAL1:.*]] = ntensor.to_tensor %[[ARG]] : !ntensor.ntensor<?xf32> to tensor<?xf32>
//  CHECK-NEXT:   %[[VAL2:.*]] = tensor.cast %[[VAL1]] : tensor<?xf32> to tensor<5xf32>
//  CHECK-NEXT:   %[[VAL3:.*]] = ntensor.from_tensor %[[VAL2]] : tensor<5xf32> to !ntensor.ntensor<5xf32>
//  CHECK-NEXT:   return %[[VAL3]] : !ntensor.ntensor<5xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32, "test">) -> !ntensor.ntensor<5xf32, "test"> {
  %0 = ntensor.cast %arg1 : !ntensor.ntensor<?xf32, "test"> to !ntensor.ntensor<5xf32, "test">
  return %0 : !ntensor.ntensor<5xf32, "test">
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?xf32, "test">)
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.env_region "test" -> !ntensor.ntensor<5xf32, "test"> {
//  CHECK-NEXT:   %[[VAL1:.*]] = ntensor.to_tensor %[[ARG]] : !ntensor.ntensor<?xf32, "test"> to tensor<?xf32>
//  CHECK-NEXT:   %[[VAL2:.*]] = tensor.cast %[[VAL1]] : tensor<?xf32> to tensor<5xf32>
//  CHECK-NEXT:   %[[VAL3:.*]] = ntensor.from_tensor %[[VAL2]] : tensor<5xf32> to !ntensor.ntensor<5xf32, "test">
//  CHECK-NEXT:   imex_util.env_region_yield %[[VAL3]] : !ntensor.ntensor<5xf32, "test">
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : !ntensor.ntensor<5xf32, "test">

// -----

func.func @test(%arg1: f32, %arg2: f32, %arg3: f32) -> !ntensor.ntensor<3xf32> {
  %0 = ntensor.from_elements %arg1, %arg2, %arg3 : !ntensor.ntensor<3xf32>
  return %0 : !ntensor.ntensor<3xf32>
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32)
//  CHECK-NEXT:   %[[RES:.*]] = tensor.from_elements %[[ARG1]], %[[ARG2]], %[[ARG3]] : tensor<3xf32>
//  CHECK-NEXT:   %[[RES1:.*]] = ntensor.from_tensor %[[RES]] : tensor<3xf32> to !ntensor.ntensor<3xf32>
//  CHECK-NEXT:   return %[[RES1]] : !ntensor.ntensor<3xf32>

// -----

func.func @test(%arg1: f32, %arg2: f32, %arg3: f32) -> !ntensor.ntensor<3xf32, "test"> {
  %0 = ntensor.from_elements %arg1, %arg2, %arg3 : !ntensor.ntensor<3xf32, "test">
  return %0 : !ntensor.ntensor<3xf32, "test">
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32)
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.env_region "test" -> !ntensor.ntensor<3xf32, "test"> {
//  CHECK-NEXT:   %[[RES1:.*]] = tensor.from_elements %[[ARG1]], %[[ARG2]], %[[ARG3]] : tensor<3xf32>
//  CHECK-NEXT:   %[[RES2:.*]] = ntensor.from_tensor %[[RES1]] : tensor<3xf32> to !ntensor.ntensor<3xf32, "test">
//  CHECK-NEXT:   imex_util.env_region_yield %[[RES2]] : !ntensor.ntensor<3xf32, "test">
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : !ntensor.ntensor<3xf32, "test">

// -----

func.func @test(%arg1: !ntensor.ntensor<?x?xf32>, %arg2: index, %arg3: index, %arg4: index) -> !ntensor.ntensor<?x?xf32> {
  %1 = ntensor.subview %arg1[1, %arg2][2, %arg3][3, %arg4] : !ntensor.ntensor<?x?xf32> to !ntensor.ntensor<?x?xf32>
  return %1 : !ntensor.ntensor<?x?xf32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?x?xf32>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index)
//  CHECK-NEXT:   %[[T1:.*]] = ntensor.to_tensor %[[ARG1]] : !ntensor.ntensor<?x?xf32> to tensor<?x?xf32>
//  CHECK-NEXT:   %[[T2:.*]] = tensor.extract_slice %[[T1]][1, %[[ARG2]]] [2, %[[ARG3]]] [3, %[[ARG4]]] : tensor<?x?xf32> to tensor<2x?xf32>
//  CHECK-NEXT:   %[[T3:.*]] = ntensor.from_tensor %[[T2]] : tensor<2x?xf32> to !ntensor.ntensor<?x?xf32>
//  CHECK-NEXT:   return %[[T3]] : !ntensor.ntensor<?x?xf32>

// -----

func.func @test(%arg1: !ntensor.ntensor<?x?xf32, "test">, %arg2: index, %arg3: index, %arg4: index) -> !ntensor.ntensor<?x?xf32, "test"> {
  %1 = ntensor.subview %arg1[1, %arg2][2, %arg3][3, %arg4] : !ntensor.ntensor<?x?xf32, "test"> to !ntensor.ntensor<?x?xf32, "test">
  return %1 : !ntensor.ntensor<?x?xf32, "test">
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?x?xf32, "test">, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.env_region "test" -> !ntensor.ntensor<?x?xf32, "test"> {
//  CHECK-NEXT:   %[[T1:.*]] = ntensor.to_tensor %[[ARG1]] : !ntensor.ntensor<?x?xf32, "test"> to tensor<?x?xf32>
//  CHECK-NEXT:   %[[T2:.*]] = tensor.extract_slice %[[T1]][1, %[[ARG2]]] [2, %[[ARG3]]] [3, %[[ARG4]]] : tensor<?x?xf32> to tensor<2x?xf32>
//  CHECK-NEXT:   %[[T3:.*]] = ntensor.from_tensor %[[T2]] : tensor<2x?xf32> to !ntensor.ntensor<?x?xf32, "test">
//  CHECK-NEXT:   imex_util.env_region_yield %[[T3]] : !ntensor.ntensor<?x?xf32, "test">
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : !ntensor.ntensor<?x?xf32, "test">

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> f32 {
  %0 = arith.constant 0 : index
  %1 = ntensor.load %arg1[%0] : !ntensor.ntensor<?xf32>
  return %1 : f32
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT:   %[[IND:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[T1:.*]] = ntensor.to_tensor %[[ARG]] : !ntensor.ntensor<?xf32> to tensor<?xf32>
//  CHECK-NEXT:   %[[RES:.*]] = tensor.extract %[[T1]][%[[IND]]] : tensor<?xf32>
//  CHECK-NEXT:   return %[[RES]] : f32

// -----

func.func @test(%arg1: !ntensor.ntensor<?xf32, "test">) -> f32 {
  %0 = arith.constant 0 : index
  %1 = ntensor.load %arg1[%0] : !ntensor.ntensor<?xf32, "test">
  return %1 : f32
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?xf32, "test">)
//  CHECK-NEXT:   %[[IND:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[RES:.*]] = imex_util.env_region "test" -> f32 {
//  CHECK-NEXT:   %[[T1:.*]] = ntensor.to_tensor %[[ARG]] : !ntensor.ntensor<?xf32, "test"> to tensor<?xf32>
//  CHECK-NEXT:   %[[T2:.*]] = tensor.extract %[[T1]][%[[IND]]] : tensor<?xf32>
//  CHECK-NEXT:   imex_util.env_region_yield %[[T2]] : f32
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : f32

// -----

func.func @test(%arg: !ntensor.ntensor<?x?xf32>) -> index {
  %0 = arith.constant 0 : index
  %1 = ntensor.dim %arg, %0 : !ntensor.ntensor<?x?xf32>
  return %1 : index
}

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: !ntensor.ntensor<?x?xf32>)
//  CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT:   %[[T0:.*]] = ntensor.to_tensor %arg0 : !ntensor.ntensor<?x?xf32> to tensor<?x?xf32>
//  CHECK-NEXT:   %[[RES:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
//  CHECK-NEXT:   return %[[RES]] : index
