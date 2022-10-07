#map = affine_map<(d0) -> (d0)>
module @jit_real.364 {
  func @main(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
    %0 = linalg.init_tensor [2] : tensor<2xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<2xcomplex<f32>>) outs(%0 : tensor<2xf32>) {
    ^bb0(%arg1: complex<f32>, %arg2: f32):
      %2 = complex.re %arg1 : complex<f32>
      linalg.yield %2 : f32
    } -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }
}

