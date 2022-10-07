#map = affine_map<() -> ()>
module @jit_v_em.42 {
  func @main(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = arith.constant dense<1.000000e+00> : tensor<f64>
    %cst_0 = arith.constant dense<1.000000e-04> : tensor<f64>
    %0 = linalg.init_tensor [] : tensor<f64>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg0, %cst_0 : tensor<f64>, tensor<f64>) outs(%0 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.maxf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %cst_1 = arith.constant dense<4.270000e+00> : tensor<f64>
    %2 = linalg.init_tensor [] : tensor<f64>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%1, %cst_1 : tensor<f64>, tensor<f64>) outs(%2 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %cst_2 = arith.constant dense<1.100000e+01> : tensor<f64>
    %4 = linalg.init_tensor [] : tensor<f64>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%3, %cst_2 : tensor<f64>, tensor<f64>) outs(%4 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %cst_3 = arith.constant dense<1.600000e+01> : tensor<f64>
    %6 = linalg.init_tensor [] : tensor<f64>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%5, %cst_3 : tensor<f64>, tensor<f64>) outs(%6 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.divf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %8 = linalg.init_tensor [] : tensor<f64>
    %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%7, %cst : tensor<f64>, tensor<f64>) outs(%8 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.addf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %10 = linalg.init_tensor [] : tensor<f64>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%3, %3 : tensor<f64>, tensor<f64>) outs(%10 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %cst_4 = arith.constant dense<3.000000e+00> : tensor<f64>
    %12 = linalg.init_tensor [] : tensor<f64>
    %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%11, %cst_4 : tensor<f64>, tensor<f64>) outs(%12 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %14 = linalg.init_tensor [] : tensor<f64>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%13, %cst_3 : tensor<f64>, tensor<f64>) outs(%14 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.divf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %16 = linalg.init_tensor [] : tensor<f64>
    %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%9, %15 : tensor<f64>, tensor<f64>) outs(%16 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.addf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %18 = linalg.init_tensor [] : tensor<f64>
    %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%3, %3 : tensor<f64>, tensor<f64>) outs(%18 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %20 = linalg.init_tensor [] : tensor<f64>
    %21 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%3, %19 : tensor<f64>, tensor<f64>) outs(%20 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %cst_5 = arith.constant dense<4.800000e+01> : tensor<f64>
    %22 = linalg.init_tensor [] : tensor<f64>
    %23 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%21, %cst_5 : tensor<f64>, tensor<f64>) outs(%22 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.divf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %24 = linalg.init_tensor [] : tensor<f64>
    %25 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%17, %23 : tensor<f64>, tensor<f64>) outs(%24 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.addf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %26 = linalg.init_tensor [] : tensor<f64>
    %27 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%3 : tensor<f64>) outs(%26 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64):
      %38 = arith.negf %arg1 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %28 = linalg.init_tensor [] : tensor<f64>
    %29 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%27 : tensor<f64>) outs(%28 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64):
      %38 = math.exp %arg1 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %30 = linalg.init_tensor [] : tensor<f64>
    %31 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%25, %29 : tensor<f64>, tensor<f64>) outs(%30 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %32 = linalg.init_tensor [] : tensor<f64>
    %33 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%cst, %31 : tensor<f64>, tensor<f64>) outs(%32 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.subf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %cst_6 = arith.constant dense<1.4399651726528193> : tensor<f64>
    %34 = linalg.init_tensor [] : tensor<f64>
    %35 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%33, %cst_6 : tensor<f64>, tensor<f64>) outs(%34 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.mulf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    %36 = linalg.init_tensor [] : tensor<f64>
    %37 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%35, %1 : tensor<f64>, tensor<f64>) outs(%36 : tensor<f64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %38 = arith.divf %arg1, %arg2 : f64
      linalg.yield %38 : f64
    } -> tensor<f64>
    return %37 : tensor<f64>
  }
}

