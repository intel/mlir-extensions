#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module @gemv {
  func @main(%arg0: tensor<3x3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>) -> tensor<3xf32> {
    %0 = linalg.init_tensor [3] : tensor<3xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x3xf32>, tensor<3xf32>) outs(%arg2 : tensor<3xf32>) attrs =  {iterator_ranges = [3, 3]} {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %2 = arith.mulf %arg3, %arg4 : f32
      %3 = arith.addf %arg5, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<3xf32>
    return %1 : tensor<3xf32>
  }
}
