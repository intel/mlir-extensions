#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0)>
module @jit__linspace.39 {
  func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<100xf64> {
    %0 = linalg.init_tensor [] : tensor<f64>
    %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = []} ins(%arg0 : tensor<i64>) outs(%0 : tensor<f64>) {
    ^bb0(%arg2: i64, %arg3: f64):
      %27 = arith.sitofp %arg2 : i64 to f64
      linalg.yield %27 : f64
    } -> tensor<f64>
    %2 = tensor.expand_shape %1 [] : tensor<f64> into tensor<1xf64>
    %3 = tensor.collapse_shape %2 [] : tensor<1xf64> into tensor<f64>
    %4 = linalg.init_tensor [99] : tensor<99xf64>
    %5 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%3 : tensor<f64>) outs(%4 : tensor<99xf64>) {
    ^bb0(%arg2: f64, %arg3: f64):
      linalg.yield %arg2 : f64
    } -> tensor<99xf64>
    %cst = arith.constant dense<1.000000e+00> : tensor<f64>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<99xf64>
    %6 = linalg.init_tensor [99] : tensor<99xf64>
    %7 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%6 : tensor<99xf64>) {
    ^bb0(%arg2: f64):
      %27 = linalg.index 0 : index
      %28 = arith.index_cast %27 : index to i64
      %29 = arith.sitofp %28 : i64 to f64
      linalg.yield %29 : f64
    } -> tensor<99xf64>
    %cst_1 = arith.constant dense<9.900000e+01> : tensor<f64>
    %cst_2 = arith.constant dense<9.900000e+01> : tensor<99xf64>
    %8 = linalg.init_tensor [99] : tensor<99xf64>
    %9 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%7, %cst_2 : tensor<99xf64>, tensor<99xf64>) outs(%8 : tensor<99xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):
      %27 = arith.divf %arg2, %arg3 : f64
      linalg.yield %27 : f64
    } -> tensor<99xf64>
    %10 = linalg.init_tensor [99] : tensor<99xf64>
    %11 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%cst_0, %9 : tensor<99xf64>, tensor<99xf64>) outs(%10 : tensor<99xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):
      %27 = arith.subf %arg2, %arg3 : f64
      linalg.yield %27 : f64
    } -> tensor<99xf64>
    %12 = linalg.init_tensor [99] : tensor<99xf64>
    %13 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%5, %11 : tensor<99xf64>, tensor<99xf64>) outs(%12 : tensor<99xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):
      %27 = arith.mulf %arg2, %arg3 : f64
      linalg.yield %27 : f64
    } -> tensor<99xf64>
    %14 = linalg.init_tensor [] : tensor<f64>
    %15 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = []} ins(%arg1 : tensor<i64>) outs(%14 : tensor<f64>) {
    ^bb0(%arg2: i64, %arg3: f64):
      %27 = arith.sitofp %arg2 : i64 to f64
      linalg.yield %27 : f64
    } -> tensor<f64>
    %16 = tensor.expand_shape %15 [] : tensor<f64> into tensor<1xf64>
    %17 = tensor.collapse_shape %16 [] : tensor<1xf64> into tensor<f64>
    %18 = linalg.init_tensor [99] : tensor<99xf64>
    %19 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%17 : tensor<f64>) outs(%18 : tensor<99xf64>) {
    ^bb0(%arg2: f64, %arg3: f64):
      linalg.yield %arg2 : f64
    } -> tensor<99xf64>
    %20 = linalg.init_tensor [99] : tensor<99xf64>
    %21 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%19, %9 : tensor<99xf64>, tensor<99xf64>) outs(%20 : tensor<99xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):
      %27 = arith.mulf %arg2, %arg3 : f64
      linalg.yield %27 : f64
    } -> tensor<99xf64>
    %22 = linalg.init_tensor [99] : tensor<99xf64>
    %23 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%13, %21 : tensor<99xf64>, tensor<99xf64>) outs(%22 : tensor<99xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):
      %27 = arith.addf %arg2, %arg3 : f64
      linalg.yield %27 : f64
    } -> tensor<99xf64>
    %24 = tensor.expand_shape %15 [] : tensor<f64> into tensor<1xf64>
    %c0 = arith.constant 0 : index
    %25 = linalg.init_tensor [100] : tensor<100xf64>
    %26 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%25 : tensor<100xf64>) {
    ^bb0(%arg2: f64):
      %27 = linalg.index 0 : index
      %28 = linalg.index 0 : index
      %c0_3 = arith.constant 0 : index
      %29 = tensor.dim %23, %c0_3 : tensor<99xf64>
      %30 = arith.addi %c0, %29 : index
      %31 = arith.cmpi ult, %28, %30 : index
      %32 = scf.if %31 -> (f64) {
        %33 = arith.subi %28, %c0 : index
        %34 = tensor.extract %23[%33] : tensor<99xf64>
        scf.yield %34 : f64
      } else {
        %33 = arith.subi %28, %30 : index
        %34 = tensor.extract %24[%33] : tensor<1xf64>
        scf.yield %34 : f64
      }
      linalg.yield %32 : f64
    } -> tensor<100xf64>
    return %26 : tensor<100xf64>
  }
}

