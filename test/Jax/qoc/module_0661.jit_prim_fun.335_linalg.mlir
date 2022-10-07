#map = affine_map<(d0) -> (d0)>
module @jit_prim_fun.335 {
  func @main(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<2xi32> {
    %c0 = arith.constant 0 : index
    %0 = linalg.init_tensor [2] : tensor<2xi32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<2xi32>) {
    ^bb0(%arg2: i32):
      %2 = linalg.index 0 : index
      %3 = linalg.index 0 : index
      %c0_0 = arith.constant 0 : index
      %4 = tensor.dim %arg0, %c0_0 : tensor<1xi32>
      %5 = arith.addi %c0, %4 : index
      %6 = arith.cmpi ult, %3, %5 : index
      %7 = scf.if %6 -> (i32) {
        %8 = arith.subi %3, %c0 : index
        %9 = tensor.extract %arg0[%8] : tensor<1xi32>
        scf.yield %9 : i32
      } else {
        %8 = arith.subi %3, %5 : index
        %9 = tensor.extract %arg1[%8] : tensor<1xi32>
        scf.yield %9 : i32
      }
      linalg.yield %7 : i32
    } -> tensor<2xi32>
    return %1 : tensor<2xi32>
  }
}

