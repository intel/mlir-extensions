#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module @jit__reduce_sum.357 {
  func @main(%arg0: tensor<1xi32>) -> tensor<i32> {
    %cst = arith.constant dense<0> : tensor<i32>
    %c0_i32 = arith.constant 0 : i32
    %0 = linalg.init_tensor [] : tensor<i32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<i32>) -> tensor<i32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%arg0 : tensor<1xi32>) outs(%1 : tensor<i32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      %3 = arith.addi %arg1, %arg2 : i32
      linalg.yield %3 : i32
    } -> tensor<i32>
    return %2 : tensor<i32>
  }
}
