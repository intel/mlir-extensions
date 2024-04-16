// RUN: imex-opt --split-input-file -canonicalize %s -verify-diagnostics -o -| FileCheck %s

// -----
module {
    func.func @test_canonicalize(%arg0: index) {
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %9 = ndarray.create %c3, %c3, %c3 {dtype = 0 : i8} : (index, index, index) -> !ndarray.ndarray<?x?x?xf64>
      // THe first should get all static shapes
      %handle, %lHalo, %rHalo    = "distruntime.get_halo"(%9, %c4, %c4, %c4, %c1, %c1, %c1, %c1, %c1, %c1, %c2, %c2, %c2) {team = 22} : (!ndarray.ndarray<?x?x?xf64>, index, index, index, index, index, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, !ndarray.ndarray<?x?x?xf64>, !ndarray.ndarray<?x?x?xf64>)
      // THe second has a unknown SSA values in shape so not we cannot infer full static shapes
      %handle1, %lHalo1, %rHalo1 = "distruntime.get_halo"(%9, %c4, %c4, %c4, %c1, %c1, %c1, %c1, %c1, %c1, %arg0, %c2, %c2) {team = 22} : (!ndarray.ndarray<?x?x?xf64>, index, index, index, index, index, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, !ndarray.ndarray<?x?x?xf64>, !ndarray.ndarray<?x?x?xf64>)
      return
    }
}
// CHECK-LABEL: func.func @test_canonicalize
// CHECK: "distruntime.get_halo"
// CHECK-SAME: -> (!distruntime.asynchandle, !ndarray.ndarray<0x2x2xf64>, !ndarray.ndarray<0x2x2xf64>)
// CHECK: "distruntime.get_halo"
// CHECK-SAME: -> (!distruntime.asynchandle, !ndarray.ndarray<?x2x2xf64>, !ndarray.ndarray<?x2x2xf64>)

module {
  func.func @test_canonicalize_reshape() {
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = ndarray.create %c2, %c4 value %c1_i32 : (index, index, i32) -> !ndarray.ndarray<?x?xi32>
    %handle, %nlArray = distruntime.copy_reshape %0 g_shape %c3, %c4 l_offs %c1, %c0 to n_g_shape %c2, %c3, %c2 n_offs %c1, %c0, %c0 n_shape %c1, %c3, %c2 {team = 22 : i64} : (!ndarray.ndarray<?x?xi32>, index, index, index, index, index, index, index, index, index, index, index, index, index) -> (!distruntime.asynchandle, !ndarray.ndarray<?x?x?xi32>)
    "distruntime.wait"(%handle) : (!distruntime.asynchandle) -> ()
    return
  }
}
// CHECK-LABEL: func.func @test_canonicalize_reshape
// CHECK: ndarray.create
// CHECK-SAME: -> !ndarray.ndarray<2x4xi32>
// CHECK: distruntime.copy_reshape
// CHECK-SAME: -> (!distruntime.asynchandle, !ndarray.ndarray<1x3x2xi32>)
// CHECK: "distruntime.wait"
