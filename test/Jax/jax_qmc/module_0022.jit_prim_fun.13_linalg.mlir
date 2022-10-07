#map = affine_map<(d0) -> (d0)>
module @jit_prim_fun.13 {
  func @main(%arg0: tensor<1xui32>, %arg1: tensor<1xui32>) -> tensor<2xui32> {
    %0 = builtin.unrealized_conversion_cast %arg0 : tensor<1xui32> to tensor<1xi32>
    %1 = builtin.unrealized_conversion_cast %arg1 : tensor<1xui32> to tensor<1xi32>
    %c0 = arith.constant 0 : index
    %2 = linalg.init_tensor [2] : tensor<2xi32>
    %3 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%2 : tensor<2xi32>) {
    ^bb0(%arg2: i32):
      %5 = linalg.index 0 : index
      %6 = linalg.index 0 : index
      %c0_0 = arith.constant 0 : index
      %7 = tensor.dim %0, %c0_0 : tensor<1xi32>
      %8 = arith.addi %c0, %7 : index
      %9 = arith.cmpi ult, %6, %8 : index
      %10 = scf.if %9 -> (i32) {
        %11 = arith.subi %6, %c0 : index
        %12 = tensor.extract %0[%11] : tensor<1xi32>
        scf.yield %12 : i32
      } else {
        %11 = arith.subi %6, %8 : index
        %12 = tensor.extract %1[%11] : tensor<1xi32>
        scf.yield %12 : i32
      }
      linalg.yield %10 : i32
    } -> tensor<2xi32>
    %4 = builtin.unrealized_conversion_cast %3 : tensor<2xi32> to tensor<2xui32>
    return %4 : tensor<2xui32>
  }
}

