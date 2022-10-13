#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module @transpose {
  func @main(%arg0: tensor<10x20xf32>) -> tensor<20x10xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [20, 10] : tensor<20x10xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<20x10xf32>) -> tensor<20x10xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<10x20xf32>) outs(%1 : tensor<20x10xf32>) attrs =  {iterator_ranges = [20, 10], name = "transpose"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<20x10xf32>
    return %2 : tensor<20x10xf32>
  }
}
