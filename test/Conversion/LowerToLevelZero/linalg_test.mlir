// XFAIL: *
// RUN:

#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0)>
module @jit_func.func.0 {
  func.func public @foo(%arg0: tensor<500xf64>, %arg1: tensor<5xf64>) -> tensor<500xf64> {
    %0 = tensor.extract_slice %arg1[0] [1] [1] : tensor<5xf64> to tensor<1xf64>
    %1 = tensor.collapse_shape %0 [] : tensor<1xf64> into tensor<f64>
    %2 = tensor.extract_slice %arg1[1] [1] [1] : tensor<5xf64> to tensor<1xf64>
    %3 = tensor.collapse_shape %2 [] : tensor<1xf64> into tensor<f64>
    %4 = tensor.extract_slice %arg1[2] [1] [1] : tensor<5xf64> to tensor<1xf64>
    %5 = tensor.collapse_shape %4 [] : tensor<1xf64> into tensor<f64>
    %6 = tensor.extract_slice %arg1[3] [1] [1] : tensor<5xf64> to tensor<1xf64>
    %7 = tensor.collapse_shape %6 [] : tensor<1xf64> into tensor<f64>
    %8 = tensor.extract_slice %arg1[4] [1] [1] : tensor<5xf64> to tensor<1xf64>
    %9 = tensor.collapse_shape %8 [] : tensor<1xf64> into tensor<f64>
    %10 = linalg.init_tensor [] : tensor<f64>
    %11 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%1, %3 : tensor<f64>, tensor<f64>) outs(%10 : tensor<f64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
      %46 = arith.addf %arg2, %arg3 : f64
      linalg.yield %46 : f64
    } -> tensor<f64>
    %12 = linalg.init_tensor [] : tensor<f64>
    %13 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%9, %7 : tensor<f64>, tensor<f64>) outs(%12 : tensor<f64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
      %46 = arith.subf %arg2, %arg3 : f64
      linalg.yield %46 : f64
    } -> tensor<f64>
    %14 = linalg.init_tensor [] : tensor<f64>
    %15 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = []} ins(%5 : tensor<f64>) outs(%14 : tensor<f64>) {
    ^bb0(%arg2: f64, %arg3: f64):  // no predecessors
      %46 = arith.negf %arg2 : f64
      linalg.yield %46 : f64
    } -> tensor<f64>
    %16 = linalg.init_tensor [500] : tensor<500xf64>
    %17 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%1 : tensor<f64>) outs(%16 : tensor<500xf64>) {
    ^bb0(%arg2: f64, %arg3: f64):  // no predecessors
      linalg.yield %arg2 : f64
    } -> tensor<500xf64>
    %18 = linalg.init_tensor [500] : tensor<500xf64>
    %19 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %17 : tensor<500xf64>, tensor<500xf64>) outs(%18 : tensor<500xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
      %46 = arith.subf %arg2, %arg3 : f64
      linalg.yield %46 : f64
    } -> tensor<500xf64>
    %20 = linalg.init_tensor [500] : tensor<500xf64>
    %21 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%15 : tensor<f64>) outs(%20 : tensor<500xf64>) {
    ^bb0(%arg2: f64, %arg3: f64):  // no predecessors
      linalg.yield %arg2 : f64
    } -> tensor<500xf64>
    %22 = linalg.init_tensor [500] : tensor<500xf64>
    %23 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%21, %19 : tensor<500xf64>, tensor<500xf64>) outs(%22 : tensor<500xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
      %46 = arith.mulf %arg2, %arg3 : f64
      linalg.yield %46 : f64
    } -> tensor<500xf64>
    %24 = linalg.init_tensor [500] : tensor<500xf64>
    %25 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%23 : tensor<500xf64>) outs(%24 : tensor<500xf64>) {
    ^bb0(%arg2: f64, %arg3: f64):  // no predecessors
      %46 = math.exp %arg2 : f64
      linalg.yield %46 : f64
    } -> tensor<500xf64>
    //%cst = arith.constant dense<1.000000e+00> : tensor<f64>
    //%cst_0 = arith.constant dense<1.000000e+00> : tensor<500xf64>
    %26 = linalg.init_tensor [500] : tensor<500xf64>
    %27 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%25, %arg0 : tensor<500xf64>, tensor<500xf64>) outs(%26 : tensor<500xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
      %46 = arith.addf %arg2, %arg3 : f64
      linalg.yield %46 : f64
    } -> tensor<500xf64>
    %28 = linalg.init_tensor [500] : tensor<500xf64>
    %29 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%13 : tensor<f64>) outs(%28 : tensor<500xf64>) {
    ^bb0(%arg2: f64, %arg3: f64):  // no predecessors
      linalg.yield %arg2 : f64
    } -> tensor<500xf64>
    %30 = linalg.init_tensor [500] : tensor<500xf64>
    %31 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%29, %27 : tensor<500xf64>, tensor<500xf64>) outs(%30 : tensor<500xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
      %46 = arith.divf %arg2, %arg3 : f64
      linalg.yield %46 : f64
    } -> tensor<500xf64>
    %32 = linalg.init_tensor [500] : tensor<500xf64>
    %33 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%7 : tensor<f64>) outs(%32 : tensor<500xf64>) {
    ^bb0(%arg2: f64, %arg3: f64):  // no predecessors
      linalg.yield %arg2 : f64
    } -> tensor<500xf64>
    %34 = linalg.init_tensor [500] : tensor<500xf64>
    %35 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%33, %31 : tensor<500xf64>, tensor<500xf64>) outs(%34 : tensor<500xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
      %46 = arith.addf %arg2, %arg3 : f64
      linalg.yield %46 : f64
    } -> tensor<500xf64>
    %36 = linalg.init_tensor [500] : tensor<500xf64>
    %37 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%1 : tensor<f64>) outs(%36 : tensor<500xf64>) {
    ^bb0(%arg2: f64, %arg3: f64):  // no predecessors
      linalg.yield %arg2 : f64
    } -> tensor<500xf64>
    %38 = linalg.init_tensor [500] : tensor<500xf64>
    %39 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %37 : tensor<500xf64>, tensor<500xf64>) outs(%38 : tensor<500xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
      %46 = arith.subf %arg2, %arg3 : f64
      linalg.yield %46 : f64
    } -> tensor<500xf64>
    %40 = linalg.init_tensor [500] : tensor<500xf64>
    %41 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%35, %39 : tensor<500xf64>, tensor<500xf64>) outs(%40 : tensor<500xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
      %46 = arith.mulf %arg2, %arg3 : f64
      linalg.yield %46 : f64
    } -> tensor<500xf64>
    %42 = linalg.init_tensor [500] : tensor<500xf64>
    %43 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%11 : tensor<f64>) outs(%42 : tensor<500xf64>) {
    ^bb0(%arg2: f64, %arg3: f64):  // no predecessors
      linalg.yield %arg2 : f64
    } -> tensor<500xf64>
    %44 = linalg.init_tensor [500] : tensor<500xf64>
    %45 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%43, %41 : tensor<500xf64>, tensor<500xf64>) outs(%44 : tensor<500xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):  // no predecessors
      %46 = arith.addf %arg2, %arg3 : f64
      linalg.yield %46 : f64
    } -> tensor<500xf64>
    return %45 : tensor<500xf64>
  }

  func.func @main() {
    %0 = arith.constant dense<1.000000e+00> : tensor<500xf64>
    %1 = arith.constant dense<[5.0, 4.0, 3.0, 2.0, 1.0]> : tensor<5xf64>
    %2 = call @foo(%0, %1) : (tensor<500xf64>, tensor<5xf64>) -> tensor<500xf64>
    %unranked = tensor.cast %2 : tensor<500xf64> to tensor<*xf64>
    call @printMemrefF64(%unranked) : (tensor<*xf64>) -> ()
    return
  }

  func.func private @printMemrefF64(%ptr : tensor<*xf64>)
}

