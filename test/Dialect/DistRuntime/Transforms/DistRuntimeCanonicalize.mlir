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
