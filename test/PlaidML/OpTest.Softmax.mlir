#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
module @softmax {
  func @main(%arg0: tensor<10x20xf32>) -> tensor<10x20xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = linalg.init_tensor [10, 1] : tensor<10x1xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<10x1xf32>) -> tensor<10x1xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<10x20xf32>) outs(%1 : tensor<10x1xf32>) attrs =  {iterator_ranges = [10, 20], name = "softmax"} {
    ^bb0(%arg1: f32, %arg2: f32):
      %12 = arith.cmpf ogt, %arg2, %arg1 : f32
      %13 = arith.select %12, %arg2, %arg1 : f32
      linalg.yield %13 : f32
    } -> tensor<10x1xf32>
    %3 = linalg.init_tensor [10, 20] : tensor<10x20xf32>
    %4 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : tensor<10x20xf32>, tensor<10x1xf32>) outs(%3 : tensor<10x20xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %12 = arith.subf %arg1, %arg2 : f32
      linalg.yield %12 : f32
    } -> tensor<10x20xf32>
    %5 = linalg.init_tensor [10, 20] : tensor<10x20xf32>
    %6 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<10x20xf32>) outs(%5 : tensor<10x20xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %12 = math.exp %arg1 : f32
      linalg.yield %12 : f32
    } -> tensor<10x20xf32>
    %7 = linalg.init_tensor [10, 1] : tensor<10x1xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<10x1xf32>) -> tensor<10x1xf32>
    %9 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%6 : tensor<10x20xf32>) outs(%8 : tensor<10x1xf32>) attrs =  {iterator_ranges = [10, 20], name = "softmax"} {
    ^bb0(%arg1: f32, %arg2: f32):
      %12 = arith.addf %arg2, %arg1 : f32
      linalg.yield %12 : f32
    } -> tensor<10x1xf32>
    %10 = linalg.init_tensor [10, 20] : tensor<10x20xf32>
    %11 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%6, %9 : tensor<10x20xf32>, tensor<10x1xf32>) outs(%10 : tensor<10x20xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %12 = arith.divf %arg1, %arg2 : f32
      linalg.yield %12 : f32
    } -> tensor<10x20xf32>
    return %11 : tensor<10x20xf32>
  }
}
