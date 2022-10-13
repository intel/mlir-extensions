#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 + d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
module @conv_1d {
  func @main(%arg0: tensor<1x244x3xi64>, %arg1: tensor<3x3x1xi64>) -> tensor<1x242x1xi64> {
    %c0_i64 = arith.constant 0 : i64
    %0 = linalg.init_tensor [1, 242, 1] : tensor<1x242x1xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1x242x1xi64>) -> tensor<1x242x1xi64>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x244x3xi64>, tensor<3x3x1xi64>) outs(%1 : tensor<1x242x1xi64>) attrs =  {iterator_ranges = [1, 242, 1, 3, 3]} {
    ^bb0(%arg2: i64, %arg3: i64, %arg4: i64):
      %3 = arith.muli %arg2, %arg3 : i64
      %4 = arith.addi %arg4, %3 : i64
      linalg.yield %4 : i64
    } -> tensor<1x242x1xi64>
    return %2 : tensor<1x242x1xi64>
  }
}
