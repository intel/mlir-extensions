#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @dot {
  func @main(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [8, 32] : tensor<8x32xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8x32xf32>) -> tensor<8x32xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<16x32xf32>) outs(%1 : tensor<8x32xf32>) attrs =  {iterator_ranges = [8, 32, 16]} {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %3 = arith.mulf %arg2, %arg3 : f32
      %4 = arith.addf %arg4, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<8x32xf32>
    return %2 : tensor<8x32xf32>
  }
}
