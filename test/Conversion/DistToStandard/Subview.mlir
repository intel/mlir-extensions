// RUN: imex-opt --convert-dist-to-standard -canonicalize %s -verify-diagnostics -o -| FileCheck %s

func.func @test1(%arg0: !ndarray.ndarray<0xf32>, %arg1: !ndarray.ndarray<4xf32>, %arg2: !ndarray.ndarray<0xf32>)
      -> (!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>) {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  %3 = dist.init_dist_array l_offset %c6 parts %arg0, %arg1, %arg2 : index, !ndarray.ndarray<0xf32>, !ndarray.ndarray<4xf32>, !ndarray.ndarray<0xf32> to !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 : i64 loffs = 6 lparts = 0,4,0>>

  %4 = dist.subview %3[2] [8] [1] toffs %c4 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 6 lparts = 0,4,0>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
  %5 = dist.subview %3[2] [8] [1] toffs %c6 tsizes %c2: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 6 lparts = 0,4,0>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
  %6 = dist.subview %3[2] [8] [1] toffs %c2 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 6 lparts = 0,4,0>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>

  %20, %21, %22 = "dist.parts_of"(%4) : (!ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) -> (!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>)
  %30, %31, %32 = "dist.parts_of"(%5) : (!ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) -> (!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>)
  %40, %41, %42 = "dist.parts_of"(%6) : (!ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) -> (!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>)

  return %20, %21, %22, %30, %31, %32, %40, %41, %42 : !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>,!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>
}
// CHECK-LABEL: func.func @test1(
// CHECK-SAME: [[v0:%.*]]: !ndarray.ndarray<0xf32>, [[v1:%.*]]: !ndarray.ndarray<4xf32>, [[v2:%.*]]: !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v0]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v1]][0] [4] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<4xf32>
// CHECK: ndarray.subview [[v2]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v0]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v1]][2] [2] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<2xf32>
// CHECK: ndarray.subview [[v2]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v0]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v1]][0] [2] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<2xf32>
// CHECK: ndarray.subview [[v2]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>

// -----
func.func private @printMemrefInd(memref<*xindex>)
  func.func @test2(%arg0: !ndarray.ndarray<1xf32>, %arg1: !ndarray.ndarray<4xf32>, %arg2: !ndarray.ndarray<0xf32>)
      -> (!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>) {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %3 = dist.init_dist_array l_offset %c5 parts %arg0, %arg1, %arg2 : index, !ndarray.ndarray<1xf32>, !ndarray.ndarray<4xf32>, !ndarray.ndarray<0xf32> to !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 5 lparts = 1,4,0>>

  %4 = dist.subview %3[2] [8] [1] toffs %c4 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 5 lparts = 1,4,0>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
  %5 = dist.subview %3[2] [8] [1] toffs %c6 tsizes %c2: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 5 lparts = 1,4,0>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
  %6 = dist.subview %3[2] [8] [1] toffs %c2 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 5 lparts = 1,4,0>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>

  %20, %21, %22 = "dist.parts_of"(%4) : (!ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) -> (!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>)
  %30, %31, %32 = "dist.parts_of"(%5) : (!ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) -> (!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>)
  %40, %41, %42 = "dist.parts_of"(%6) : (!ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) -> (!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>)

  return %20, %21, %22, %30, %31, %32, %40, %41, %42 : !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>,!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>
}
// CHECK-LABEL: func.func @test2(
// CHECK-SAME: [[v0:%.*]]: !ndarray.ndarray<1xf32>, [[v1:%.*]]: !ndarray.ndarray<4xf32>, [[v2:%.*]]: !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v0]][1] [0] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v1]][0] [4] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<4xf32>
// CHECK: ndarray.subview [[v2]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v0]][1] [0] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v1]][2] [2] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<2xf32>
// CHECK: ndarray.subview [[v2]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v0]][0] [1] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<1xf32>
// CHECK: ndarray.subview [[v1]][0] [2] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<2xf32>
// CHECK: ndarray.subview [[v2]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>

// -----
func.func @test3(%arg0: !ndarray.ndarray<1xf32>, %arg1: !ndarray.ndarray<4xf32>, %arg2: !ndarray.ndarray<1xf32>)
      -> (!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  %3 = dist.init_dist_array l_offset %c2 parts %arg0, %arg1, %arg2 : index, !ndarray.ndarray<1xf32>, !ndarray.ndarray<4xf32>, !ndarray.ndarray<1xf32> to !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 2 lparts = 1,4,1>>

  %4 = dist.subview %3[2] [8] [1] toffs %c4 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 2 lparts = 1,4,1>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
  %5 = dist.subview %3[2] [8] [1] toffs %c6 tsizes %c2: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 2 lparts = 1,4,1>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
  %6 = dist.subview %3[2] [8] [1] toffs %c0 tsizes %c6: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 2 lparts = 1,4,1>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>

  %20, %21, %22 = "dist.parts_of"(%4) : (!ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) -> (!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>)
  %30, %31, %32 = "dist.parts_of"(%5) : (!ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) -> (!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>)
  %40, %41, %42 = "dist.parts_of"(%6) : (!ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) -> (!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>)

  return %20, %21, %22, %30, %31, %32, %40, %41, %42 : !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>,!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>
}
// CHECK-LABEL: func.func @test3(
// CHECK-SAME: [[v0:%.*]]: !ndarray.ndarray<1xf32>, [[v1:%.*]]: !ndarray.ndarray<4xf32>, [[v2:%.*]]: !ndarray.ndarray<1xf32>
// CHECK: ndarray.subview [[v0]][1] [0] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v1]][3] [1] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<1xf32>
// CHECK: ndarray.subview [[v2]][0] [1] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<1xf32>
// CHECK: ndarray.subview [[v0]][1] [0] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v1]][4] [0] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v2]][1] [0] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v0]][0] [1] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<1xf32>
// CHECK: ndarray.subview [[v1]][0] [4] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<4xf32>
// CHECK: ndarray.subview [[v2]][0] [1] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<1xf32>

// -----
func.func @test4(%arg0: !ndarray.ndarray<0xf32>, %arg1: !ndarray.ndarray<4xf32>, %arg2: !ndarray.ndarray<1xf32>)
      -> (!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %3 = dist.init_dist_array l_offset %c3 parts %arg0, %arg1, %arg2 : index, !ndarray.ndarray<0xf32>, !ndarray.ndarray<4xf32>, !ndarray.ndarray<1xf32> to !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 3 lparts = 0,4,1>>

  %4 = dist.subview %3[2] [8] [1] toffs %c4 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 3 lparts = 0,4,1>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
  %5 = dist.subview %3[2] [8] [1] toffs %c5 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 3 lparts = 0,4,1>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
  %6 = dist.subview %3[2] [8] [1] toffs %c2 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 3 lparts = 0,4,1>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>

  %20, %21, %22 = "dist.parts_of"(%4) : (!ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) -> (!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>)
  %30, %31, %32 = "dist.parts_of"(%5) : (!ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) -> (!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>)
  %40, %41, %42 = "dist.parts_of"(%6) : (!ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) -> (!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>)

  return %20, %21, %22, %30, %31, %32, %40, %41, %42 : !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>,!ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>, !ndarray.ndarray<?xf32>
}
// CHECK-LABEL: func.func @test4(
// CHECK-SAME: [[v0:%.*]]: !ndarray.ndarray<0xf32>, [[v1:%.*]]: !ndarray.ndarray<4xf32>, [[v2:%.*]]: !ndarray.ndarray<1xf32>
// CHECK: ndarray.subview [[v0]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v1]][3] [1] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<1xf32>
// CHECK: ndarray.subview [[v2]][0] [1] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<1xf32>
// CHECK: ndarray.subview [[v0]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v1]][4] [0] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v2]][0] [1] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<1xf32>
// CHECK: ndarray.subview [[v0]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
// CHECK: ndarray.subview [[v1]][1] [3] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<3xf32>
// CHECK: ndarray.subview [[v2]][0] [1] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<1xf32>
