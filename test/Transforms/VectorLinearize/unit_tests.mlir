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
  //CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [0], strides = [1]} : vector<2xf32> into vector<8xf32>
  %1 = vector.insert_strided_slice %src, %dst {offsets = [0, 0], strides = [1, 1]} : vector<1x2xf32> into vector<2x4xf32>
  //CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [6], strides = [1]} : vector<2xf32> into vector<8xf32>
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
  //CHECK: %[[r0:.*]] = vector.shape_cast %arg0 : vector<4x8xf32> to vector<32xf32>
  //CHECK: %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
  //CHECK: %[[r1:.*]] = vector.extract_strided_slice %[[r0]] {offsets = [0], sizes = [8], strides = [1]} : vector<32xf32> to vector<8xf32>
  //CHECK: %[[r2:.*]] = vector.insert_strided_slice %[[r1]], %cst {offsets = [0], strides = [1]} : vector<8xf32> into vector<128xf32>
  //CHECK: %[[r3:.*]] = vector.extract_strided_slice %[[r0]] {offsets = [8], sizes = [8], strides = [1]} : vector<32xf32> to vector<8xf32>
  //CHECK: %[[r4:.*]] = vector.insert_strided_slice %[[r3]], %[[r2]] {offsets = [16], strides = [1]} : vector<8xf32> into vector<128xf32>
  //CHECK: %[[r5:.*]] = vector.extract_strided_slice %[[r0]] {offsets = [16], sizes = [8], strides = [1]} : vector<32xf32> to vector<8xf32>
  //CHECK: %[[r6:.*]] = vector.insert_strided_slice %[[r5]], %[[r4]] {offsets = [32], strides = [1]} : vector<8xf32> into vector<128xf32>
  //CHECK: %[[r7:.*]] = vector.extract_strided_slice %[[r0]] {offsets = [24], sizes = [8], strides = [1]} : vector<32xf32> to vector<8xf32>
  //CHECK: %[[r8:.*]] = vector.insert_strided_slice %[[r7]], %[[r6]] {offsets = [48], strides = [1]} : vector<8xf32> into vector<128xf32>
  //CHECK: %[[r9:.*]] = vector.shape_cast %[[r8]] : vector<128xf32> to vector<8x16xf32>
  //CHECK: return %[[r9]] : vector<8x16xf32>

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