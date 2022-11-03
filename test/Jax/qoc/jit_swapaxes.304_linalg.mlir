#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module @jit_swapaxes.304 {
  func @main(%arg0: tensor<2x1xi32>) -> tensor<1x2xi32> {
    %0 = linalg.init_tensor [1, 2] : tensor<1x2xi32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<2x1xi32>) outs(%0 : tensor<1x2xi32>) attrs =  {xla_shape = "s32[1,2]{0,1}"} {
    ^bb0(%arg1: i32, %arg2: i32):
      linalg.yield %arg1 : i32
    } -> tensor<1x2xi32>
    return %1 : tensor<1x2xi32>
  }
}
