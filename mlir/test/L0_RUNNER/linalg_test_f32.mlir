// RUN: level_zero_runner %s -e main -entry-point-result=void -shared-libs=%mlir_wrappers_dir/%shlibprefixmlir_c_runner_utils%shlibext -shared-libs=%mlir_wrappers_dir/%shlibprefixmlir_runner_utils%shlibext -shared-libs=%imex_runtime_dir/%shlibprefixdpcomp-runtime%shlibext -shared-libs=%imex_igpu_runtime_dir/%shlibprefixdpcomp-gpu-runtime%shlibext | FileCheck %s

#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0)>
module @jit_func.func.0 {
  func.func public @foo(%arg0: tensor<500xf32>, %arg1: tensor<5xf32>) -> tensor<500xf32> {
    %0 = tensor.extract_slice %arg1[0] [1] [1] : tensor<5xf32> to tensor<1xf32>
    %1 = tensor.collapse_shape %0 [] : tensor<1xf32> into tensor<f32>
    %2 = tensor.extract_slice %arg1[1] [1] [1] : tensor<5xf32> to tensor<1xf32>
    %3 = tensor.collapse_shape %2 [] : tensor<1xf32> into tensor<f32>
    %4 = tensor.extract_slice %arg1[2] [1] [1] : tensor<5xf32> to tensor<1xf32>
    %5 = tensor.collapse_shape %4 [] : tensor<1xf32> into tensor<f32>
    %6 = tensor.extract_slice %arg1[3] [1] [1] : tensor<5xf32> to tensor<1xf32>
    %7 = tensor.collapse_shape %6 [] : tensor<1xf32> into tensor<f32>
    %8 = tensor.extract_slice %arg1[4] [1] [1] : tensor<5xf32> to tensor<1xf32>
    %9 = tensor.collapse_shape %8 [] : tensor<1xf32> into tensor<f32>
    %10 = tensor.empty() : tensor<f32>
    %11 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%1, %3 : tensor<f32>, tensor<f32>) outs(%10 : tensor<f32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %46 = arith.addf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<f32>
    %12 = tensor.empty() : tensor<f32>
    %13 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []} ins(%9, %7 : tensor<f32>, tensor<f32>) outs(%12 : tensor<f32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %46 = arith.subf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<f32>
    %14 = tensor.empty() : tensor<f32>
    %15 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = []} ins(%5 : tensor<f32>) outs(%14 : tensor<f32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      %46 = arith.negf %arg2 : f32
      linalg.yield %46 : f32
    } -> tensor<f32>
    %16 = tensor.empty() : tensor<500xf32>
    %17 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%1 : tensor<f32>) outs(%16 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      linalg.yield %arg2 : f32
    } -> tensor<500xf32>
    %18 = tensor.empty() : tensor<500xf32>
    %19 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %17 : tensor<500xf32>, tensor<500xf32>) outs(%18 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %46 = arith.subf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    %20 = tensor.empty() : tensor<500xf32>
    %21 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%15 : tensor<f32>) outs(%20 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      linalg.yield %arg2 : f32
    } -> tensor<500xf32>
    %22 = tensor.empty() : tensor<500xf32>
    %23 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%21, %19 : tensor<500xf32>, tensor<500xf32>) outs(%22 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %46 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    %24 = tensor.empty() : tensor<500xf32>
    %25 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%23 : tensor<500xf32>) outs(%24 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      %46 = math.exp %arg2 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    //%cst = arith.constant dense<1.000000e+00> : tensor<f32>
    //%cst_0 = arith.constant dense<1.000000e+00> : tensor<500xf32>
    %26 = tensor.empty() : tensor<500xf32>
    %27 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%25, %arg0 : tensor<500xf32>, tensor<500xf32>) outs(%26 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %46 = arith.addf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    %28 = tensor.empty() : tensor<500xf32>
    %29 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%13 : tensor<f32>) outs(%28 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      linalg.yield %arg2 : f32
    } -> tensor<500xf32>
    %30 = tensor.empty() : tensor<500xf32>
    %31 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%29, %27 : tensor<500xf32>, tensor<500xf32>) outs(%30 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %46 = arith.divf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    %32 = tensor.empty() : tensor<500xf32>
    %33 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%7 : tensor<f32>) outs(%32 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      linalg.yield %arg2 : f32
    } -> tensor<500xf32>
    %34 = tensor.empty() : tensor<500xf32>
    %35 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%33, %31 : tensor<500xf32>, tensor<500xf32>) outs(%34 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %46 = arith.addf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    %36 = tensor.empty() : tensor<500xf32>
    %37 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%1 : tensor<f32>) outs(%36 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      linalg.yield %arg2 : f32
    } -> tensor<500xf32>
    %38 = tensor.empty() : tensor<500xf32>
    %39 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %37 : tensor<500xf32>, tensor<500xf32>) outs(%38 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %46 = arith.subf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    %40 = tensor.empty() : tensor<500xf32>
    %41 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%35, %39 : tensor<500xf32>, tensor<500xf32>) outs(%40 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %46 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    %42 = tensor.empty() : tensor<500xf32>
    %43 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%11 : tensor<f32>) outs(%42 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      linalg.yield %arg2 : f32
    } -> tensor<500xf32>
    %44 = tensor.empty() : tensor<500xf32>
    %45 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%43, %41 : tensor<500xf32>, tensor<500xf32>) outs(%44 : tensor<500xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %46 = arith.addf %arg2, %arg3 : f32
      linalg.yield %46 : f32
    } -> tensor<500xf32>
    return %45 : tensor<500xf32>
  }

  func.func @main() {
    %0 = arith.constant dense<1.000000e+00> : tensor<500xf32>
    %1 = arith.constant dense<[5.0, 4.0, 3.0, 2.0, 1.0]> : tensor<5xf32>
    %2 = call @foo(%0, %1) : (tensor<500xf32>, tensor<5xf32>) -> tensor<500xf32>
    %unranked = tensor.cast %2 : tensor<500xf32> to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [500] strides = [1] data =
    // CHECK-NEXT: [1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002,
    // CHECK-SAME: 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002, 1.00002]
    return
  }

  func.func private @printMemrefF32(%ptr : tensor<*xf32>)
}
