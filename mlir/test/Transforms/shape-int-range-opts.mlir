// RUN: imex-opt --imex-shape-int-range-opts --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>) -> i1 {
  %cst = arith.constant 0 : index
  %cst1 = arith.constant -1 : index
  %0 = tensor.dim %arg1, %cst : tensor<?xf32>
  %1 = arith.cmpi eq, %0, %cst1 : index
  return %1: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant true
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>) -> i1 {
  %cst = arith.constant 0 : index
  %cst1 = arith.constant -1 : index
  %0 = tensor.dim %arg1, %cst : tensor<?xf32>
  %1 = arith.cmpi ne, %0, %cst1 : index
  return %1: i1
}

// -----


// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant true
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>) -> i1 {
  %cst = arith.constant 0 : index
  %0 = tensor.dim %arg1, %cst : tensor<?xf32>
  %1 = arith.cmpi sge, %0, %cst : index
  return %1: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>) -> i1 {
  %cst = arith.constant 0 : index
  %0 = tensor.dim %arg1, %cst : tensor<?xf32>
  %1 = arith.cmpi slt, %0, %cst : index
  return %1: i1
}

// -----


// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant true
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>) -> i1 {
  %cst = arith.constant 0 : index
  %cst1 = arith.constant -1 : index
  %0 = tensor.dim %arg1, %cst : tensor<?xf32>
  %1 = arith.cmpi sgt, %0, %cst1 : index
  return %1: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>) -> i1 {
  %cst = arith.constant 0 : index
  %cst1 = arith.constant -1 : index
  %0 = tensor.dim %arg1, %cst : tensor<?xf32>
  %1 = arith.cmpi sle, %0, %cst1 : index
  return %1: i1
}
