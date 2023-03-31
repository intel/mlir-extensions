#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: tensor<512x640x20x15xf32>) -> tensor<512x640x20x15xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<512x640x20x15xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x640x20x15xf32>) outs(%0 : tensor<512x640x20x15xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.cmpf ugt, %in, %cst : f32
      %3 = arith.select %2, %in, %cst : f32
      linalg.yield %3 : f32
    } -> tensor<512x640x20x15xf32>
    return %1 : tensor<512x640x20x15xf32>
  }
  func.func @main() {
    %0= arith.constant dense<1.3>:tensor<512x640x20x15xf32>
    %1 = call @forward(%0) : (tensor<512x640x20x15xf32>) -> tensor<512x640x20x15xf32>
     return
  }
}
