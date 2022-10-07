#map = affine_map<() -> ()>
module @jit_absolute.341 {
  func @main(%arg0: tensor<i32>) -> tensor<i32> {
    %0 = linalg.init_tensor [] : tensor<i32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%arg0 : tensor<i32>) outs(%0 : tensor<i32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      %c0_i32 = arith.constant 0 : i32
      %2 = arith.cmpi sge, %arg1, %c0_i32 : i32
      %3 = arith.subi %c0_i32, %arg1 : i32
      %4 = arith.select %2, %arg1, %3 : i32
      linalg.yield %4 : i32
    } -> tensor<i32>
    return %1 : tensor<i32>
  }
}

