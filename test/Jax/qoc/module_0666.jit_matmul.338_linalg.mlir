#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module @jit_matmul.338 {
  func @main(%arg0: tensor<1x2x2xi32>, %arg1: tensor<1x2x2xi32>) -> tensor<1x2x2xi32> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x2x2xi32> into tensor<2x2xi32>
    %c0_i32 = arith.constant 0 : i32
    %1 = linalg.init_tensor [2, 1, 2] : tensor<2x1x2xi32>
    %2 = linalg.fill ins(%c0_i32 : i32) outs(%1 : tensor<2x1x2xi32>) -> tensor<2x1x2xi32>
    %3 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %arg1 : tensor<2x2xi32>, tensor<1x2x2xi32>) outs(%2 : tensor<2x1x2xi32>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):
      %6 = arith.muli %arg2, %arg3 : i32
      %7 = arith.addi %6, %arg4 : i32
      linalg.yield %7 : i32
    } -> tensor<2x1x2xi32>
    %4 = linalg.init_tensor [1, 2, 2] : tensor<1x2x2xi32>
    %5 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : tensor<2x1x2xi32>) outs(%4 : tensor<1x2x2xi32>) attrs =  {xla_shape = "s32[1,2,2]{2,0,1}"} {
    ^bb0(%arg2: i32, %arg3: i32):
      linalg.yield %arg2 : i32
    } -> tensor<1x2x2xi32>
    return %5 : tensor<1x2x2xi32>
  }
}
