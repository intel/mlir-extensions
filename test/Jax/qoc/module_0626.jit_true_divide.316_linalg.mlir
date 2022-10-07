#map = affine_map<() -> ()>
module @jit_true_divide.316 {
  func @main(%arg0: tensor<complex<f32>>, %arg1: tensor<i32>) -> tensor<complex<f32>> {
    %0 = linalg.init_tensor [] : tensor<complex<f32>>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%arg1 : tensor<i32>) outs(%0 : tensor<complex<f32>>) {
    ^bb0(%arg2: i32, %arg3: complex<f32>):
      %4 = arith.sitofp %arg2 : i32 to f32
      %cst = arith.constant 0.000000e+00 : f32
      %5 = complex.create %4, %cst : complex<f32>
      linalg.yield %5 : complex<f32>
    } -> tensor<complex<f32>>
    %2 = linalg.init_tensor [] : tensor<complex<f32>>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg0, %1 : tensor<complex<f32>>, tensor<complex<f32>>) outs(%2 : tensor<complex<f32>>) {
    ^bb0(%arg2: complex<f32>, %arg3: complex<f32>, %arg4: complex<f32>):
      %4 = complex.div %arg2, %arg3 : complex<f32>
      linalg.yield %4 : complex<f32>
    } -> tensor<complex<f32>>
    return %3 : tensor<complex<f32>>
  }
}

