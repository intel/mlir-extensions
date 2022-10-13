#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0 + 2, d1 + 1)>
module @explicit_padding {
  func @main(%arg0: tensor<2x3xf32>) -> tensor<6x5xf32> {
    %cst = arith.constant 0x7F800000 : f32
    %0 = linalg.init_tensor [6, 5] : tensor<6x5xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<6x5xf32>) -> tensor<6x5xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["window", "window"]} ins(%arg0 : tensor<2x3xf32>) outs(%1 : tensor<6x5xf32>) attrs =  {iterator_ranges = [2, 3], name = "explicit_padding"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<6x5xf32>
    return %2 : tensor<6x5xf32>
  }
}
