#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module @quantize {
  func @main(%arg0: tensor<3xf32>) -> tensor<3xi8> {
    %cst = arith.constant 2.560000e+02 : f32
    %0 = linalg.init_tensor [3] : tensor<3xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel"]} ins(%arg0, %cst : tensor<3xf32>, f32) outs(%0 : tensor<3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %4 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %4 : f32
    } -> tensor<3xf32>
    %2 = linalg.init_tensor [3] : tensor<3xi8>
    %3 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%1 : tensor<3xf32>) outs(%2 : tensor<3xi8>) {
    ^bb0(%arg1: f32, %arg2: i8):
      %4 = arith.fptosi %arg1 : f32 to i8
      linalg.yield %4 : i8
    } -> tensor<3xi8>
    return %3 : tensor<3xi8>
  }
}
