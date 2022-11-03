#map = affine_map<(d0, d1) -> (d0, d1)>
module @jit_conjugate.307 {
  func @main(%arg0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xcomplex<f32>> {
    %0 = linalg.init_tensor [1, 2] : tensor<1x2xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x2xcomplex<f32>>) outs(%0 : tensor<1x2xf32>) {
    ^bb0(%arg1: complex<f32>, %arg2: f32):
      %8 = complex.re %arg1 : complex<f32>
      linalg.yield %8 : f32
    } -> tensor<1x2xf32>
    %2 = linalg.init_tensor [1, 2] : tensor<1x2xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x2xcomplex<f32>>) outs(%2 : tensor<1x2xf32>) {
    ^bb0(%arg1: complex<f32>, %arg2: f32):
      %8 = complex.im %arg1 : complex<f32>
      linalg.yield %8 : f32
    } -> tensor<1x2xf32>
    %4 = linalg.init_tensor [1, 2] : tensor<1x2xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<1x2xf32>) outs(%4 : tensor<1x2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %8 = arith.negf %arg1 : f32
      linalg.yield %8 : f32
    } -> tensor<1x2xf32>
    %6 = linalg.init_tensor [1, 2] : tensor<1x2xcomplex<f32>>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %5 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%6 : tensor<1x2xcomplex<f32>>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: complex<f32>):
      %8 = complex.create %arg1, %arg2 : complex<f32>
      linalg.yield %8 : complex<f32>
    } -> tensor<1x2xcomplex<f32>>
    return %7 : tensor<1x2xcomplex<f32>>
  }
}
