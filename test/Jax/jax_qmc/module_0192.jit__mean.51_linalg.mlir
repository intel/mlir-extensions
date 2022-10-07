#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
#map2 = affine_map<() -> ()>
module @jit__mean.51 {
  func @main(%arg0: tensor<1000x16xf64>) -> tensor<f64> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = linalg.init_tensor [] : tensor<f64>
    %1 = linalg.fill ins(%cst_0 : f64) outs(%0 : tensor<f64>) -> tensor<f64>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction", "reduction"]} ins(%arg0 : tensor<1000x16xf64>) outs(%1 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64):
      %5 = arith.addf %arg1, %arg2 : f64
      linalg.yield %5 : f64
    } -> tensor<f64>
    %cst_1 = arith.constant dense<1.600000e+04> : tensor<f64>
    %3 = linalg.init_tensor [] : tensor<f64>
    %4 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%2, %cst_1 : tensor<f64>, tensor<f64>) outs(%3 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %5 = arith.divf %arg1, %arg2 : f64
      linalg.yield %5 : f64
    } -> tensor<f64>
    return %4 : tensor<f64>
  }
}

