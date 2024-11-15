// RUN: imex-opt --split-input-file -tosa-to-tensor -canonicalize %s -verify-diagnostics -o -| FileCheck %s
// tosa-to-linalg needed to get tensor dialect loaded

module {
  func.func @test_canonicalize_reshape() {
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = ndarray.create %c2, %c4 value %c1_i32 : (index, index, i32) -> tensor<?x?xi32>
    %handle, %nlArray = distruntime.copy_reshape %0 g_shape %c3, %c4 l_offs %c1, %c0 to n_g_shape %c2, %c3, %c2 n_offs %c1, %c0, %c0 n_shape %c1, %c3, %c2 {team = 22 : i64} : (tensor<?x?xi32>, index, index, index, index, index, index, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, tensor<?x?x?xi32>)
    "distruntime.wait"(%handle) : (!distruntime.asynchandle) -> ()
    return
  }
}
// CHECK-LABEL: func.func @test_canonicalize_reshape
// CHECK: ndarray.create
// CHECK-SAME: -> tensor<2x4xi32>
// CHECK: distruntime.copy_reshape
// CHECK-SAME: -> (!distruntime.asynchandle, tensor<1x3x2xi32>)
// CHECK: "distruntime.wait"
