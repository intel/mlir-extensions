#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> ()>
module @jit_trace.340 {
  func @main(%arg0: tensor<2x2xi32>) -> tensor<i32> {
    %0 = linalg.init_tensor [2] : tensor<2xi32>
    %1 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%0 : tensor<2xi32>) {
    ^bb0(%arg1: i32):
      %14 = linalg.index 0 : index
      %15 = arith.index_cast %14 : index to i32
      linalg.yield %15 : i32
    } -> tensor<2xi32>
    %2 = linalg.init_tensor [2, 2] : tensor<2x2xi32>
    %3 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<2xi32>) outs(%2 : tensor<2x2xi32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      linalg.yield %arg1 : i32
    } -> tensor<2x2xi32>
    %4 = linalg.init_tensor [2] : tensor<2xi32>
    %5 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%4 : tensor<2xi32>) {
    ^bb0(%arg1: i32):
      %14 = linalg.index 0 : index
      %15 = arith.index_cast %14 : index to i32
      linalg.yield %15 : i32
    } -> tensor<2xi32>
    %6 = linalg.init_tensor [2, 2] : tensor<2x2xi32>
    %7 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<2xi32>) outs(%6 : tensor<2x2xi32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      linalg.yield %arg1 : i32
    } -> tensor<2x2xi32>
    %8 = linalg.init_tensor [2, 2] : tensor<2x2xi1>
    %9 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%3, %7 : tensor<2x2xi32>, tensor<2x2xi32>) outs(%8 : tensor<2x2xi1>) {
    ^bb0(%arg1: i32, %arg2: i32, %arg3: i1):
      %14 = arith.cmpi eq, %arg1, %arg2 : i32
      linalg.yield %14 : i1
    } -> tensor<2x2xi1>
    %cst = arith.constant dense<0> : tensor<i32>
    %cst_0 = arith.constant dense<0> : tensor<2x2xi32>
    %10 = call @_where.10(%9, %arg0, %cst_0) : (tensor<2x2xi1>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    %cst_1 = arith.constant dense<0> : tensor<i32>
    %c0_i32 = arith.constant 0 : i32
    %11 = linalg.init_tensor [] : tensor<i32>
    %12 = linalg.fill ins(%c0_i32 : i32) outs(%11 : tensor<i32>) -> tensor<i32>
    %13 = linalg.generic {indexing_maps = [#map2, #map4], iterator_types = ["reduction", "reduction"]} ins(%10 : tensor<2x2xi32>) outs(%12 : tensor<i32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      %14 = arith.addi %arg1, %arg2 : i32
      linalg.yield %14 : i32
    } -> tensor<i32>
    return %13 : tensor<i32>
  }
  func private @_where.10(%arg0: tensor<2x2xi1>, %arg1: tensor<2x2xi32>, %arg2: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = linalg.init_tensor [2, 2] : tensor<2x2xi32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<2x2xi1>, tensor<2x2xi32>, tensor<2x2xi32>) outs(%0 : tensor<2x2xi32>) {
    ^bb0(%arg3: i1, %arg4: i32, %arg5: i32, %arg6: i32):
      %2 = arith.select %arg3, %arg4, %arg5 : i32
      linalg.yield %2 : i32
    } -> tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
