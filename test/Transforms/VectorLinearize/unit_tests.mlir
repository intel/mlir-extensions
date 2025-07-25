// RUN: imex-opt %s -split-input-file -imex-vector-linearize | FileCheck %s

// -----
func.func @test() -> vector<4x2xf16> {
  %cst = arith.constant dense<0.0> : vector<8x1xf32>
  %1 = vector.extract_strided_slice %cst {offsets = [0, 0], sizes = [4, 1], strides = [1, 1]} : vector<8x1xf32> to vector<4x1xf32>
  //CHECK: %[[r1:.*]] = vector.bitcast %{{.*}} : vector<4xf32> to vector<8xf16>
  //CHECK: %[[r2:.*]] = vector.shape_cast %[[r1]] : vector<8xf16> to vector<4x2xf16>
  %2 = vector.bitcast %1 : vector<4x1xf32> to vector<4x2xf16>
  return %2 : vector<4x2xf16>
}

// -----
// test the insert_strided_slice
func.func @test() -> vector<2x4xf32> {
  %src = arith.constant dense<1.0> : vector<1x2xf32>
  %dst = arith.constant dense<1.0> : vector<2x4xf32>
  //CHECK: vector.shuffle %{{.*}}, %{{.*}} [8, 9, 2, 3, 4, 5, 6, 7] : vector<8xf32>, vector<2xf32>
  %1 = vector.insert_strided_slice %src, %dst {offsets = [0, 0], strides = [1, 1]} : vector<1x2xf32> into vector<2x4xf32>
  //CHECK: vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 8, 9] : vector<8xf32>, vector<2xf32>
  %2 = vector.insert_strided_slice %src, %1 {offsets = [1, 2], strides = [1, 1]} : vector<1x2xf32> into vector<2x4xf32>
  return %2 : vector<2x4xf32>
}

// -----
// test the loop
func.func @test() -> vector<2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %1 = arith.constant dense<1.0> : vector<2x4xf32>
  //CHECK: scf.for %{{.*}} iter_args(%{{.*}}) -> (vector<8xf32>)
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%arg1 = %1) -> (vector<2x4xf32>) {
    %2 = arith.addf %1, %arg1 : vector<2x4xf32>
    scf.yield %2 : vector<2x4xf32>
  }
  return %r : vector<2x4xf32>
}

// -----
//CHECK: test_vector_insert_2d_idx(%[[arg0:.*]]: vector<4x8xf32>)
func.func @test_vector_insert_2d_idx(%arg0: vector<4x8xf32>) -> vector<8x16xf32> {
  //CHECK: vector.shuffle %{{.*}}, %{{.*}} [128, 129, 130, 131, 132, 133, 134, 135,
  //CHECK-SAME: 8, 9, 10, 11, 12, 13, 14, 15, 136, 137, 138, 139, 140, 141, 142, 143,
  //CHECK-SAME: 24, 25, 26, 27, 28, 29, 30, 31, 144, 145, 146, 147, 148, 149, 150, 151,
  //CHECK-SAME: 40, 41, 42, 43, 44, 45, 46, 47, 152, 153, 154, 155, 156, 157, 158, 159, 56, 57, 58, 59,
  //CHECK-SAME: 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
  //CHECK-SAME: 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
  //CHECK-SAME: 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
  //CHECK-SAME: 120, 121, 122, 123, 124, 125, 126, 127] : vector<128xf32>, vector<32xf32>
  //CHECK: vector.shape_cast %{{.*}} : vector<128xf32> to vector<8x16xf32>
  %cst = arith.constant dense <0.0> : vector<8x16xf32>
  %0 = vector.insert_strided_slice %arg0, %cst {offsets = [0, 0], strides = [1, 1]} : vector<4x8xf32> into vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @gather_memref_2d
func.func @gather_memref_2d(%base: memref<?x?xf32>, %v: vector<2x3xindex>, %mask: vector<2x3xi1>, %pass_thru: vector<2x3xf32>) -> vector<2x3xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
// CHECK:    %[[OFF0:.+]]  = arith.addi %{{.*}}, %{{.*}} : index
// CHECK:    %{{.*}}  = scf.if %{{.*}} -> (vector<3xf32>)
// CHECK:      [[LD0:%.+]]   = vector.load %{{.*}}[%{{.*}}, %[[OFF0]]] : memref<?x?xf32>, vector<1xf32>
// CHECK:      [[ELEM0:%.+]] = vector.extract [[LD0]][0] : f32 from vector<1xf32>
// CHECK:      [[INS0:%.+]]  = vector.insert [[ELEM0]], %{{.*}} [0] : f32 into vector<3xf32>
// CHECK:      scf.yield [[INS0]] : vector<3xf32>
// CHECK:    else
// CHECK:      scf.yield %{{.*}}: vector<3xf32>
// CHECK-COUNT-5: scf.if
  %0 = vector.gather %base[%c0, %c1][%v], %mask, %pass_thru : memref<?x?xf32>, vector<2x3xindex>, vector<2x3xi1>, vector<2x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// Test if with nested ops and multiple results
func.func @test_if_nested() -> (vector<4x2xi32>, vector<2x4xi32>) {
  %cond = arith.constant 1 : i1
  %v0 = arith.constant dense<5> : vector<4x2xi32>
  %v1 = arith.constant dense<6> : vector<2x4xi32>
  // CHECK: %{{.*}}:2 = scf.if %{{.*}} -> (vector<8xi32>, vector<8xi32>)
  %r:2 = scf.if %cond -> (vector<4x2xi32>, vector<2x4xi32>) {
    %mul0 = arith.muli %v0, %v0 : vector<4x2xi32>
    %add1 = arith.addi %v1, %v1 : vector<2x4xi32>
    %result1 = arith.subi %add1, %add1 : vector<2x4xi32>
    // CHECK: scf.yield %{{.*}}, %{{.*}} : vector<8xi32>, vector<8xi32>
    scf.yield %mul0, %result1 : vector<4x2xi32>, vector<2x4xi32>
  } else {
    %sub0 = arith.subi %v0, %v0 : vector<4x2xi32>
    %mul1 = arith.muli %v1, %v1 : vector<2x4xi32>
    // CHECK: scf.yield %{{.*}}, %{{.*}} : vector<8xi32>, vector<8xi32>
    scf.yield %sub0, %mul1 : vector<4x2xi32>, vector<2x4xi32>
  }
  // CHECK: vector.shape_cast %{{.*}}#1 : vector<8xi32> to vector<2x4xi32>
  // CHECK: vector.shape_cast %{{.*}}#0 : vector<8xi32> to vector<4x2xi32>
  return %r#0, %r#1 : vector<4x2xi32>, vector<2x4xi32>
}

// Test if with single 2D vector and both branches
func.func @test_if_single_vector() -> vector<16x1xi32> {
  %cond = arith.constant 0 : i1
  %v = arith.constant dense<3> : vector<16x1xi32>
  // CHECK: %{{.*}} = scf.if %{{.*}} -> (vector<16xi32>)
  %r = scf.if %cond -> (vector<16x1xi32>) {
    %add = arith.addi %v, %v : vector<16x1xi32>
    // CHECK: scf.yield %{{.*}} : vector<16xi32>
    scf.yield %add : vector<16x1xi32>
  } else {
    %sub = arith.subi %v, %v : vector<16x1xi32>
    // CHECK: scf.yield %{{.*}} : vector<16xi32>
    scf.yield %sub : vector<16x1xi32>
  }
  // CHECK: vector.shape_cast %{{.*}} : vector<16xi32> to vector<16x1xi32>
  return %r : vector<16x1xi32>
}

func.func @test_if_basic(%arg0: vector<2x4xf32>, %arg1: vector<1x8xf32>) -> (vector<2x4xf32>, vector<1x8xf32>) {
  // CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<1x8xf32> to vector<8xf32>
  // CHECK: %{{.*}} = vector.shape_cast %{{.*}} : vector<2x4xf32> to vector<8xf32>
  %0 = vector.shape_cast %arg1 : vector<1x8xf32> to vector<8xf32>
  %1 = vector.shape_cast %arg0 : vector<2x4xf32> to vector<8xf32>
  %cond = arith.constant 1 : i1
  // CHECK: %{{.*}}:2 = scf.if %{{.*}} -> (vector<8xf32>, vector<8xf32>) {
  %r:2 = scf.if %cond -> (vector<2x4xf32>, vector<1x8xf32>) {
    %sum0 = arith.addf %arg0, %arg0 : vector<2x4xf32>
    %sum1 = arith.addf %arg1, %arg1 : vector<1x8xf32>
    // CHECK: arith.addf %{{.*}}, %{{.*}} : vector<8xf32>
    // CHECK: arith.addf %{{.*}}, %{{.*}} : vector<8xf32>
    // CHECK: scf.yield %{{.*}}, %{{.*}} : vector<8xf32>, vector<8xf32>
    scf.yield %sum0, %sum1 : vector<2x4xf32>, vector<1x8xf32>
  } else {
    %diff0 = arith.subf %arg0, %arg0 : vector<2x4xf32>
    %diff1 = arith.subf %arg1, %arg1 : vector<1x8xf32>
    // CHECK: arith.subf %{{.*}}, %{{.*}} : vector<8xf32>
    // CHECK: arith.subf %{{.*}}, %{{.*}} : vector<8xf32>
    // CHECK: scf.yield %{{.*}}, %{{.*}} : vector<8xf32>, vector<8xf32>
    scf.yield %diff0, %diff1 : vector<2x4xf32>, vector<1x8xf32>
  }
  // CHECK: vector.shape_cast %{{.*}}#1 : vector<8xf32> to vector<1x8xf32>
  // CHECK: vector.shape_cast %{{.*}}#0 : vector<8xf32> to vector<2x4xf32>
  return %r#0, %r#1 : vector<2x4xf32>, vector<1x8xf32>
}

func.func @test_while() -> vector<2x4xf32> {
  %v = arith.constant dense<1.0> : vector<2x4xf32>
  %result = scf.while (%arg0 = %v) : (vector<2x4xf32>) -> vector<2x4xf32> {
    // CHECK: scf.while (%arg0 = %{{.*}}) : (vector<8xf32>) -> vector<8xf32> {
    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi slt, %c0, %c0 : i32
    scf.condition(%cond) %arg0 : vector<2x4xf32>
  } do {
  ^bb0(%arg1: vector<2x4xf32>):
    // CHECK: ^bb0(%{{.*}}: vector<8xf32>):
    %add = arith.addf %arg1, %arg1 : vector<2x4xf32>
    scf.yield %add : vector<2x4xf32>
    // CHECK: scf.yield %{{.*}} : vector<8xf32>
  }
  // CHECK: vector.shape_cast %{{.*}} : vector<8xf32> to vector<2x4xf32>
  return %result : vector<2x4xf32>
}
