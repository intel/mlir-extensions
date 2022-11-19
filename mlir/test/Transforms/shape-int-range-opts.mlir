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

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32> {imex.shape_range = [#imex_util.index_range<[1,10]>]}) -> i1 {
  %cst = arith.constant 0 : index
  %0 = tensor.dim %arg1, %cst : tensor<?xf32>
  %1 = arith.cmpi eq, %0, %cst : index
  return %1: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32> {imex.shape_range = [#imex_util.index_range<[1,7]>]},
                %arg2: tensor<?xf32> {imex.shape_range = [#imex_util.index_range<[2,10]>]},
                %cond: i1) -> i1 {
  %cst = arith.constant 0 : index
  %0 = arith.select %cond, %arg1, %arg2 : tensor<?xf32>
  %1 = tensor.dim %0, %cst : tensor<?xf32>
  %2 = arith.cmpi eq, %1, %cst : index
  return %2: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>) -> i1 {
  %cst0 = arith.constant 0 : index
  %cst1 = arith.constant 1 : index
  %0 = tensor.from_elements %cst0, %cst1 : tensor<2xindex>
  %1 = tensor.reshape %arg1(%0) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = tensor.dim %1, %cst0 : tensor<?x?xf32>
  %3 = arith.cmpi eq, %2, %cst1 : index
  return %3: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>) -> i1 {
  %cst0 = arith.constant 0 : index
  %cst1 = arith.constant 1 : index
  %0 = tensor.from_elements %cst0, %cst1 : tensor<2xindex>
  %1 = tensor.reshape %arg1(%0) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %2 = tensor.dim %1, %cst1 : tensor<?x?xf32>
  %3 = arith.cmpi eq, %2, %cst0 : index
  return %3: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>, %arg2: tensor<?xf32> {imex.shape_range = [#imex_util.index_range<[2,10]>]}) -> i1 {
  %cst0 = arith.constant 0 : index
  %cst1 = arith.constant 1 : index
  %0 = tensor.dim %arg2, %cst0 : tensor<?xf32>
  %1 = tensor.extract_slice %arg1[0] [%0] [1] : tensor<?xf32> to tensor<?xf32>
  %2 = tensor.dim %1, %cst0 : tensor<?xf32>
  %3 = arith.cmpi eq, %2, %cst1 : index
  return %3: i1
}
