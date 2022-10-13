#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @dot_f16 {
  func @main(%arg0: tensor<8x16xf16>, %arg1: tensor<16x32xf16>) -> tensor<8x32xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = linalg.init_tensor [8, 32] : tensor<8x32xf16>
    %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<8x32xf16>) -> tensor<8x32xf16>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x16xf16>, tensor<16x32xf16>) outs(%1 : tensor<8x32xf16>) attrs =  {iterator_ranges = [8, 32, 16]} {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %3 = arith.mulf %arg2, %arg3 : f16
      %4 = arith.addf %arg4, %3 : f16
      linalg.yield %4 : f16
    } -> tensor<8x32xf16>
    return %2 : tensor<8x32xf16>
  }
}
