// RUN: imex-opt --convert-dist-to-standard -canonicalize %s -verify-diagnostics -o -| FileCheck %s

module {
  func.func private @printMemrefInd(memref<*xindex>)
  func.func @ddpt_jit1(%arg0: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 6 lparts = 0,4,0>>)
        -> (!ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>,
            !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>,
            !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %0 = dist.subview %arg0[2] [8] [1] toffs %c4 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 6 lparts = 0,4,0>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    %1 = dist.subview %arg0[2] [8] [1] toffs %c6 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 6 lparts = 0,4,0>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    %2 = dist.subview %arg0[2] [8] [1] toffs %c2 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 6 lparts = 0,4,0>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    return %0, %1, %2: !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
  }
  // CHECK-LABEL: func.func @ddpt_jit1(
  // CHECK-SAME: [[arg0:%.*]]: !ndarray.ndarray<0xf32>, [[arg1:%.*]]: !ndarray.ndarray<4xf32>, [[arg2:%.*]]: !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg0]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg1]][0] [4] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<4xf32>
  // CHECK: ndarray.subview [[arg2]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg0]][2] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg1]][2] [2] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<2xf32>
  // CHECK: ndarray.subview [[arg2]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg0]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg1]][0] [2] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<2xf32>
  // CHECK: ndarray.subview [[arg2]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>

  // -----
  func.func @ddpt_jit2(%arg0: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 5 lparts = 1,4,0>>)
        -> (!ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>,
            !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>,
            !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %0 = dist.subview %arg0[2] [8] [1] toffs %c4 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 5 lparts = 1,4,0>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    %1 = dist.subview %arg0[2] [8] [1] toffs %c6 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 5 lparts = 1,4,0>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    %2 = dist.subview %arg0[2] [8] [1] toffs %c2 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 5 lparts = 1,4,0>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    return %0, %1, %2: !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
  }
  // CHECK-LABEL: func.func @ddpt_jit2(
  // CHECK-SAME: [[arg0:%.*]]: !ndarray.ndarray<1xf32>, [[arg1:%.*]]: !ndarray.ndarray<4xf32>, [[arg2:%.*]]: !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg0]][1] [0] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg1]][0] [4] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<4xf32>
  // CHECK: ndarray.subview [[arg2]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg0]][3] [0] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg1]][2] [2] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<2xf32>
  // CHECK: ndarray.subview [[arg2]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg0]][0] [1] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<1xf32>
  // CHECK: ndarray.subview [[arg1]][0] [2] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<2xf32>
  // CHECK: ndarray.subview [[arg2]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>

  // -----
  func.func @ddpt_jit3(%arg0: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 5 lparts = 1,4,1>>)
        -> (!ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>,
            !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>,
            !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %0 = dist.subview %arg0[2] [8] [1] toffs %c4 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 5 lparts = 1,4,1>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    %1 = dist.subview %arg0[2] [8] [1] toffs %c6 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 5 lparts = 1,4,1>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    %2 = dist.subview %arg0[2] [8] [1] toffs %c2 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 5 lparts = 1,4,1>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    return %0, %1, %2: !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
  }
  // CHECK-LABEL: func.func @ddpt_jit3(
  // CHECK-SAME: [[arg0:%.*]]: !ndarray.ndarray<1xf32>, [[arg1:%.*]]: !ndarray.ndarray<4xf32>, [[arg2:%.*]]: !ndarray.ndarray<1xf32>
  // CHECK: ndarray.subview [[arg0]][1] [0] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg1]][0] [4] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<4xf32>
  // CHECK: ndarray.subview [[arg2]][0] [0] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg0]][3] [0] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg1]][2] [2] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<2xf32>
  // CHECK: ndarray.subview [[arg2]][0] [1] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<1xf32>
  // CHECK: ndarray.subview [[arg0]][0] [1] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<1xf32>
  // CHECK: ndarray.subview [[arg1]][0] [2] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<2xf32>
  // CHECK: ndarray.subview [[arg2]][0] [0] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<0xf32>

  // -----
  func.func @ddpt_jit4(%arg0: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 6 lparts = 0,4,1>>)
        -> (!ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>,
            !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>,
            !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>) {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %0 = dist.subview %arg0[2] [8] [1] toffs %c4 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 6 lparts = 0,4,1>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    %1 = dist.subview %arg0[2] [8] [1] toffs %c6 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 6 lparts = 0,4,1>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    %2 = dist.subview %arg0[2] [8] [1] toffs %c2 tsizes %c4: !ndarray.ndarray<16xf32, #dist.dist_env<team = 22 loffs = 6 lparts = 0,4,1>> to !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
    return %0, %1, %2: !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>, !ndarray.ndarray<8xf32, #dist.dist_env<team = 22 loffs = ? lparts = ?,?,?>>
  }
  // CHECK-LABEL: func.func @ddpt_jit4(
  // CHECK-SAME: [[arg0:%.*]]: !ndarray.ndarray<0xf32>, [[arg1:%.*]]: !ndarray.ndarray<4xf32>, [[arg2:%.*]]: !ndarray.ndarray<1xf32>
  // CHECK: ndarray.subview [[arg0]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg1]][0] [4] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<4xf32>
  // CHECK: ndarray.subview [[arg2]][0] [0] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg0]][2] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg1]][2] [2] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<2xf32>
  // CHECK: ndarray.subview [[arg2]][0] [1] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<1xf32>
  // CHECK: ndarray.subview [[arg0]][0] [0] [1] : !ndarray.ndarray<0xf32> to !ndarray.ndarray<0xf32>
  // CHECK: ndarray.subview [[arg1]][0] [2] [1] : !ndarray.ndarray<4xf32> to !ndarray.ndarray<2xf32>
  // CHECK: ndarray.subview [[arg2]][0] [0] [1] : !ndarray.ndarray<1xf32> to !ndarray.ndarray<0xf32>
}
