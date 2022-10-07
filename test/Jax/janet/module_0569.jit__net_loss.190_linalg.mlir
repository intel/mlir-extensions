#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> ()>
#map3 = affine_map<() -> ()>
!tuple = type tuple<tensor<f32>, tensor<f32>, tensor<60x6xf32>, tensor<4x6xf32>, tensor<60x4xf32>, tensor<f32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>, tensor<8x4xf32>, tensor<60x8xf32>, tensor<f32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>, tensor<16x8xf32>, tensor<60x16xf32>, tensor<f32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>, tensor<32x16xf32>, tensor<60x32xf32>, tensor<f32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>, tensor<13x32xf32>, tensor<60x13xf32>, tensor<f32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>, tensor<60x13xf32>>
module @jit__net_loss.190 {
  func @main(%arg0: tensor<13x13xf32>, %arg1: tensor<13xf32>, %arg2: tensor<13x32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<32x16xf32>, %arg5: tensor<16xf32>, %arg6: tensor<16x8xf32>, %arg7: tensor<8xf32>, %arg8: tensor<8x4xf32>, %arg9: tensor<4xf32>, %arg10: tensor<4x6xf32>, %arg11: tensor<6xf32>, %arg12: tensor<60x13xf32>, %arg13: tensor<60x6xf32>) -> !tuple {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [60, 13] : tensor<60x13xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<60x13xf32>) -> tensor<60x13xf32>
    %2 = linalg.matmul ins(%arg12, %arg0 : tensor<60x13xf32>, tensor<13x13xf32>) outs(%1 : tensor<60x13xf32>) -> tensor<60x13xf32>
    %3 = tensor.expand_shape %arg1 [[0, 1]] : tensor<13xf32> into tensor<1x13xf32>
    %4 = tensor.collapse_shape %3 [[0, 1]] : tensor<1x13xf32> into tensor<13xf32>
    %5 = linalg.init_tensor [60, 13] : tensor<60x13xf32>
    %6 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<13xf32>) outs(%5 : tensor<60x13xf32>) {
    ^bb0(%arg14: f32, %arg15: f32):
      linalg.yield %arg14 : f32
    } -> tensor<60x13xf32>
    %7 = linalg.init_tensor [60, 13] : tensor<60x13xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2, %6 : tensor<60x13xf32>, tensor<60x13xf32>) outs(%7 : tensor<60x13xf32>) {
    ^bb0(%arg14: f32, %arg15: f32, %arg16: f32):
      %99 = arith.addf %arg14, %arg15 : f32
      linalg.yield %99 : f32
    } -> tensor<60x13xf32>
    %9 = call @jvp_selu_.58(%8) {xla_shape = "(f32[60,13]{1,0}, f32[], pred[60,13]{1,0}, f32[], f32[60,13]{1,0}, /*index=5*/pred[60,13]{1,0}, f32[60,13]{1,0})"} : (tensor<60x13xf32>) -> tuple<tensor<60x13xf32>, tensor<f32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>
    %10 = "mhlo.get_tuple_element"(%9) {index = 0 : i32} : (tuple<tensor<60x13xf32>, tensor<f32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>) -> tensor<60x13xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %11 = linalg.init_tensor [60, 32] : tensor<60x32xf32>
    %12 = linalg.fill ins(%cst_0 : f32) outs(%11 : tensor<60x32xf32>) -> tensor<60x32xf32>
    %13 = linalg.matmul ins(%10, %arg2 : tensor<60x13xf32>, tensor<13x32xf32>) outs(%12 : tensor<60x32xf32>) -> tensor<60x32xf32>
    %14 = tensor.expand_shape %arg3 [[0, 1]] : tensor<32xf32> into tensor<1x32xf32>
    %15 = tensor.collapse_shape %14 [[0, 1]] : tensor<1x32xf32> into tensor<32xf32>
    %16 = linalg.init_tensor [60, 32] : tensor<60x32xf32>
    %17 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%15 : tensor<32xf32>) outs(%16 : tensor<60x32xf32>) {
    ^bb0(%arg14: f32, %arg15: f32):
      linalg.yield %arg14 : f32
    } -> tensor<60x32xf32>
    %18 = linalg.init_tensor [60, 32] : tensor<60x32xf32>
    %19 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%13, %17 : tensor<60x32xf32>, tensor<60x32xf32>) outs(%18 : tensor<60x32xf32>) {
    ^bb0(%arg14: f32, %arg15: f32, %arg16: f32):
      %99 = arith.addf %arg14, %arg15 : f32
      linalg.yield %99 : f32
    } -> tensor<60x32xf32>
    %20 = call @jvp_selu__1.124(%19) {xla_shape = "(f32[60,32]{1,0}, f32[], pred[60,32]{1,0}, f32[], f32[60,32]{1,0}, /*index=5*/pred[60,32]{1,0}, f32[60,32]{1,0})"} : (tensor<60x32xf32>) -> tuple<tensor<60x32xf32>, tensor<f32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>
    %21 = "mhlo.get_tuple_element"(%20) {index = 0 : i32} : (tuple<tensor<60x32xf32>, tensor<f32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>) -> tensor<60x32xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %22 = linalg.init_tensor [60, 16] : tensor<60x16xf32>
    %23 = linalg.fill ins(%cst_1 : f32) outs(%22 : tensor<60x16xf32>) -> tensor<60x16xf32>
    %24 = linalg.matmul ins(%21, %arg4 : tensor<60x32xf32>, tensor<32x16xf32>) outs(%23 : tensor<60x16xf32>) -> tensor<60x16xf32>
    %25 = tensor.expand_shape %arg5 [[0, 1]] : tensor<16xf32> into tensor<1x16xf32>
    %26 = tensor.collapse_shape %25 [[0, 1]] : tensor<1x16xf32> into tensor<16xf32>
    %27 = linalg.init_tensor [60, 16] : tensor<60x16xf32>
    %28 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%26 : tensor<16xf32>) outs(%27 : tensor<60x16xf32>) {
    ^bb0(%arg14: f32, %arg15: f32):
      linalg.yield %arg14 : f32
    } -> tensor<60x16xf32>
    %29 = linalg.init_tensor [60, 16] : tensor<60x16xf32>
    %30 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%24, %28 : tensor<60x16xf32>, tensor<60x16xf32>) outs(%29 : tensor<60x16xf32>) {
    ^bb0(%arg14: f32, %arg15: f32, %arg16: f32):
      %99 = arith.addf %arg14, %arg15 : f32
      linalg.yield %99 : f32
    } -> tensor<60x16xf32>
    %31 = call @jvp_selu__5.190(%30) {xla_shape = "(f32[60,16]{1,0}, f32[], pred[60,16]{1,0}, f32[], f32[60,16]{1,0}, /*index=5*/pred[60,16]{1,0}, f32[60,16]{1,0})"} : (tensor<60x16xf32>) -> tuple<tensor<60x16xf32>, tensor<f32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>
    %32 = "mhlo.get_tuple_element"(%31) {index = 0 : i32} : (tuple<tensor<60x16xf32>, tensor<f32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>) -> tensor<60x16xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %33 = linalg.init_tensor [60, 8] : tensor<60x8xf32>
    %34 = linalg.fill ins(%cst_2 : f32) outs(%33 : tensor<60x8xf32>) -> tensor<60x8xf32>
    %35 = linalg.matmul ins(%32, %arg6 : tensor<60x16xf32>, tensor<16x8xf32>) outs(%34 : tensor<60x8xf32>) -> tensor<60x8xf32>
    %36 = tensor.expand_shape %arg7 [[0, 1]] : tensor<8xf32> into tensor<1x8xf32>
    %37 = tensor.collapse_shape %36 [[0, 1]] : tensor<1x8xf32> into tensor<8xf32>
    %38 = linalg.init_tensor [60, 8] : tensor<60x8xf32>
    %39 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%37 : tensor<8xf32>) outs(%38 : tensor<60x8xf32>) {
    ^bb0(%arg14: f32, %arg15: f32):
      linalg.yield %arg14 : f32
    } -> tensor<60x8xf32>
    %40 = linalg.init_tensor [60, 8] : tensor<60x8xf32>
    %41 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%35, %39 : tensor<60x8xf32>, tensor<60x8xf32>) outs(%40 : tensor<60x8xf32>) {
    ^bb0(%arg14: f32, %arg15: f32, %arg16: f32):
      %99 = arith.addf %arg14, %arg15 : f32
      linalg.yield %99 : f32
    } -> tensor<60x8xf32>
    %42 = call @jvp_selu__9.256(%41) {xla_shape = "(f32[60,8]{1,0}, f32[], pred[60,8]{1,0}, f32[], f32[60,8]{1,0}, /*index=5*/pred[60,8]{1,0}, f32[60,8]{1,0})"} : (tensor<60x8xf32>) -> tuple<tensor<60x8xf32>, tensor<f32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>
    %43 = "mhlo.get_tuple_element"(%42) {index = 0 : i32} : (tuple<tensor<60x8xf32>, tensor<f32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>) -> tensor<60x8xf32>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %44 = linalg.init_tensor [60, 4] : tensor<60x4xf32>
    %45 = linalg.fill ins(%cst_3 : f32) outs(%44 : tensor<60x4xf32>) -> tensor<60x4xf32>
    %46 = linalg.matmul ins(%43, %arg8 : tensor<60x8xf32>, tensor<8x4xf32>) outs(%45 : tensor<60x4xf32>) -> tensor<60x4xf32>
    %47 = tensor.expand_shape %arg9 [[0, 1]] : tensor<4xf32> into tensor<1x4xf32>
    %48 = tensor.collapse_shape %47 [[0, 1]] : tensor<1x4xf32> into tensor<4xf32>
    %49 = linalg.init_tensor [60, 4] : tensor<60x4xf32>
    %50 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%48 : tensor<4xf32>) outs(%49 : tensor<60x4xf32>) {
    ^bb0(%arg14: f32, %arg15: f32):
      linalg.yield %arg14 : f32
    } -> tensor<60x4xf32>
    %51 = linalg.init_tensor [60, 4] : tensor<60x4xf32>
    %52 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%46, %50 : tensor<60x4xf32>, tensor<60x4xf32>) outs(%51 : tensor<60x4xf32>) {
    ^bb0(%arg14: f32, %arg15: f32, %arg16: f32):
      %99 = arith.addf %arg14, %arg15 : f32
      linalg.yield %99 : f32
    } -> tensor<60x4xf32>
    %53 = call @jvp_selu__13.322(%52) {xla_shape = "(f32[60,4]{1,0}, f32[], pred[60,4]{1,0}, f32[], f32[60,4]{1,0}, /*index=5*/pred[60,4]{1,0}, f32[60,4]{1,0})"} : (tensor<60x4xf32>) -> tuple<tensor<60x4xf32>, tensor<f32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>
    %54 = "mhlo.get_tuple_element"(%53) {index = 0 : i32} : (tuple<tensor<60x4xf32>, tensor<f32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>) -> tensor<60x4xf32>
    %cst_4 = arith.constant 0.000000e+00 : f32
    %55 = linalg.init_tensor [60, 6] : tensor<60x6xf32>
    %56 = linalg.fill ins(%cst_4 : f32) outs(%55 : tensor<60x6xf32>) -> tensor<60x6xf32>
    %57 = linalg.matmul ins(%54, %arg10 : tensor<60x4xf32>, tensor<4x6xf32>) outs(%56 : tensor<60x6xf32>) -> tensor<60x6xf32>
    %58 = tensor.expand_shape %arg11 [[0, 1]] : tensor<6xf32> into tensor<1x6xf32>
    %59 = tensor.collapse_shape %58 [[0, 1]] : tensor<1x6xf32> into tensor<6xf32>
    %60 = linalg.init_tensor [60, 6] : tensor<60x6xf32>
    %61 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%59 : tensor<6xf32>) outs(%60 : tensor<60x6xf32>) {
    ^bb0(%arg14: f32, %arg15: f32):
      linalg.yield %arg14 : f32
    } -> tensor<60x6xf32>
    %62 = linalg.init_tensor [60, 6] : tensor<60x6xf32>
    %63 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%57, %61 : tensor<60x6xf32>, tensor<60x6xf32>) outs(%62 : tensor<60x6xf32>) {
    ^bb0(%arg14: f32, %arg15: f32, %arg16: f32):
      %99 = arith.addf %arg14, %arg15 : f32
      linalg.yield %99 : f32
    } -> tensor<60x6xf32>
    %64 = call @jvp__mse_.355(%63, %arg13) {xla_shape = "(f32[], f32[], f32[60,6]{1,0})"} : (tensor<60x6xf32>, tensor<60x6xf32>) -> tuple<tensor<f32>, tensor<f32>, tensor<60x6xf32>>
    %65 = "mhlo.get_tuple_element"(%64) {index = 0 : i32} : (tuple<tensor<f32>, tensor<f32>, tensor<60x6xf32>>) -> tensor<f32>
    %66 = "mhlo.get_tuple_element"(%64) {index = 1 : i32} : (tuple<tensor<f32>, tensor<f32>, tensor<60x6xf32>>) -> tensor<f32>
    %67 = "mhlo.get_tuple_element"(%64) {index = 2 : i32} : (tuple<tensor<f32>, tensor<f32>, tensor<60x6xf32>>) -> tensor<60x6xf32>
    %68 = "mhlo.get_tuple_element"(%53) {index = 1 : i32} : (tuple<tensor<60x4xf32>, tensor<f32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>) -> tensor<f32>
    %69 = "mhlo.get_tuple_element"(%53) {index = 2 : i32} : (tuple<tensor<60x4xf32>, tensor<f32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>) -> tensor<60x4xi1>
    %70 = "mhlo.get_tuple_element"(%53) {index = 3 : i32} : (tuple<tensor<60x4xf32>, tensor<f32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>) -> tensor<f32>
    %71 = "mhlo.get_tuple_element"(%53) {index = 4 : i32} : (tuple<tensor<60x4xf32>, tensor<f32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>) -> tensor<60x4xf32>
    %72 = "mhlo.get_tuple_element"(%53) {index = 5 : i32} : (tuple<tensor<60x4xf32>, tensor<f32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>) -> tensor<60x4xi1>
    %73 = "mhlo.get_tuple_element"(%53) {index = 6 : i32} : (tuple<tensor<60x4xf32>, tensor<f32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>) -> tensor<60x4xf32>
    %74 = "mhlo.get_tuple_element"(%42) {index = 1 : i32} : (tuple<tensor<60x8xf32>, tensor<f32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>) -> tensor<f32>
    %75 = "mhlo.get_tuple_element"(%42) {index = 2 : i32} : (tuple<tensor<60x8xf32>, tensor<f32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>) -> tensor<60x8xi1>
    %76 = "mhlo.get_tuple_element"(%42) {index = 3 : i32} : (tuple<tensor<60x8xf32>, tensor<f32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>) -> tensor<f32>
    %77 = "mhlo.get_tuple_element"(%42) {index = 4 : i32} : (tuple<tensor<60x8xf32>, tensor<f32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>) -> tensor<60x8xf32>
    %78 = "mhlo.get_tuple_element"(%42) {index = 5 : i32} : (tuple<tensor<60x8xf32>, tensor<f32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>) -> tensor<60x8xi1>
    %79 = "mhlo.get_tuple_element"(%42) {index = 6 : i32} : (tuple<tensor<60x8xf32>, tensor<f32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>) -> tensor<60x8xf32>
    %80 = "mhlo.get_tuple_element"(%31) {index = 1 : i32} : (tuple<tensor<60x16xf32>, tensor<f32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>) -> tensor<f32>
    %81 = "mhlo.get_tuple_element"(%31) {index = 2 : i32} : (tuple<tensor<60x16xf32>, tensor<f32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>) -> tensor<60x16xi1>
    %82 = "mhlo.get_tuple_element"(%31) {index = 3 : i32} : (tuple<tensor<60x16xf32>, tensor<f32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>) -> tensor<f32>
    %83 = "mhlo.get_tuple_element"(%31) {index = 4 : i32} : (tuple<tensor<60x16xf32>, tensor<f32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>) -> tensor<60x16xf32>
    %84 = "mhlo.get_tuple_element"(%31) {index = 5 : i32} : (tuple<tensor<60x16xf32>, tensor<f32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>) -> tensor<60x16xi1>
    %85 = "mhlo.get_tuple_element"(%31) {index = 6 : i32} : (tuple<tensor<60x16xf32>, tensor<f32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>) -> tensor<60x16xf32>
    %86 = "mhlo.get_tuple_element"(%20) {index = 1 : i32} : (tuple<tensor<60x32xf32>, tensor<f32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>) -> tensor<f32>
    %87 = "mhlo.get_tuple_element"(%20) {index = 2 : i32} : (tuple<tensor<60x32xf32>, tensor<f32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>) -> tensor<60x32xi1>
    %88 = "mhlo.get_tuple_element"(%20) {index = 3 : i32} : (tuple<tensor<60x32xf32>, tensor<f32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>) -> tensor<f32>
    %89 = "mhlo.get_tuple_element"(%20) {index = 4 : i32} : (tuple<tensor<60x32xf32>, tensor<f32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>) -> tensor<60x32xf32>
    %90 = "mhlo.get_tuple_element"(%20) {index = 5 : i32} : (tuple<tensor<60x32xf32>, tensor<f32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>) -> tensor<60x32xi1>
    %91 = "mhlo.get_tuple_element"(%20) {index = 6 : i32} : (tuple<tensor<60x32xf32>, tensor<f32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>) -> tensor<60x32xf32>
    %92 = "mhlo.get_tuple_element"(%9) {index = 1 : i32} : (tuple<tensor<60x13xf32>, tensor<f32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>) -> tensor<f32>
    %93 = "mhlo.get_tuple_element"(%9) {index = 2 : i32} : (tuple<tensor<60x13xf32>, tensor<f32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>) -> tensor<60x13xi1>
    %94 = "mhlo.get_tuple_element"(%9) {index = 3 : i32} : (tuple<tensor<60x13xf32>, tensor<f32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>) -> tensor<f32>
    %95 = "mhlo.get_tuple_element"(%9) {index = 4 : i32} : (tuple<tensor<60x13xf32>, tensor<f32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>) -> tensor<60x13xf32>
    %96 = "mhlo.get_tuple_element"(%9) {index = 5 : i32} : (tuple<tensor<60x13xf32>, tensor<f32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>) -> tensor<60x13xi1>
    %97 = "mhlo.get_tuple_element"(%9) {index = 6 : i32} : (tuple<tensor<60x13xf32>, tensor<f32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>) -> tensor<60x13xf32>
    %98 = "mhlo.tuple"(%65, %66, %67, %arg10, %54, %68, %69, %70, %71, %72, %73, %arg8, %43, %74, %75, %76, %77, %78, %79, %arg6, %32, %80, %81, %82, %83, %84, %85, %arg4, %21, %86, %87, %88, %89, %90, %91, %arg2, %10, %92, %93, %94, %95, %96, %97, %arg12) {xla_shape = "(f32[], f32[], f32[60,6]{1,0}, f32[4,6]{1,0}, f32[60,4]{1,0}, /*index=5*/f32[], pred[60,4]{1,0}, f32[], f32[60,4]{1,0}, pred[60,4]{1,0}, /*index=10*/f32[60,4]{1,0}, f32[8,4]{1,0}, f32[60,8]{1,0}, f32[], pred[60,8]{1,0}, /*index=15*/f32[], f32[60,8]{1,0}, pred[60,8]{1,0}, f32[60,8]{1,0}, f32[16,8]{1,0}, /*index=20*/f32[60,16]{1,0}, f32[], pred[60,16]{1,0}, f32[], f32[60,16]{1,0}, /*index=25*/pred[60,16]{1,0}, f32[60,16]{1,0}, f32[32,16]{1,0}, f32[60,32]{1,0}, f32[], /*index=30*/pred[60,32]{1,0}, f32[], f32[60,32]{1,0}, pred[60,32]{1,0}, f32[60,32]{1,0}, /*index=35*/f32[13,32]{1,0}, f32[60,13]{1,0}, f32[], pred[60,13]{1,0}, f32[], /*index=40*/f32[60,13]{1,0}, pred[60,13]{1,0}, f32[60,13]{1,0}, f32[60,13]{1,0})"} : (tensor<f32>, tensor<f32>, tensor<60x6xf32>, tensor<4x6xf32>, tensor<60x4xf32>, tensor<f32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>, tensor<8x4xf32>, tensor<60x8xf32>, tensor<f32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>, tensor<16x8xf32>, tensor<60x16xf32>, tensor<f32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>, tensor<32x16xf32>, tensor<60x32xf32>, tensor<f32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>, tensor<13x32xf32>, tensor<60x13xf32>, tensor<f32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>, tensor<60x13xf32>) -> !tuple
    return %98 : !tuple
  }
  func private @jvp_selu_.58(%arg0: tensor<60x13xf32>) -> tuple<tensor<60x13xf32>, tensor<f32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>> {
    %cst = arith.constant dense<1.67326319> : tensor<f32>
    %0 = call @jvp_elu_.36(%arg0, %cst) {xla_shape = "(f32[60,13]{1,0}, pred[60,13]{1,0}, f32[], f32[60,13]{1,0}, pred[60,13]{1,0}, /*index=5*/f32[60,13]{1,0})"} : (tensor<60x13xf32>, tensor<f32>) -> tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>
    %1 = "mhlo.get_tuple_element"(%0) {index = 0 : i32} : (tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>) -> tensor<60x13xf32>
    %cst_0 = arith.constant dense<1.05070102> : tensor<f32>
    %cst_1 = arith.constant dense<1.05070102> : tensor<60x13xf32>
    %2 = linalg.init_tensor [60, 13] : tensor<60x13xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %cst_1 : tensor<60x13xf32>, tensor<60x13xf32>) outs(%2 : tensor<60x13xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %10 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %10 : f32
    } -> tensor<60x13xf32>
    %cst_2 = arith.constant dense<1.05070102> : tensor<f32>
    %4 = "mhlo.get_tuple_element"(%0) {index = 1 : i32} : (tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>) -> tensor<60x13xi1>
    %5 = "mhlo.get_tuple_element"(%0) {index = 2 : i32} : (tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>) -> tensor<f32>
    %6 = "mhlo.get_tuple_element"(%0) {index = 3 : i32} : (tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>) -> tensor<60x13xf32>
    %7 = "mhlo.get_tuple_element"(%0) {index = 4 : i32} : (tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>) -> tensor<60x13xi1>
    %8 = "mhlo.get_tuple_element"(%0) {index = 5 : i32} : (tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>) -> tensor<60x13xf32>
    %9 = "mhlo.tuple"(%3, %cst_2, %4, %5, %6, %7, %8) {xla_shape = "(f32[60,13]{1,0}, f32[], pred[60,13]{1,0}, f32[], f32[60,13]{1,0}, /*index=5*/pred[60,13]{1,0}, f32[60,13]{1,0})"} : (tensor<60x13xf32>, tensor<f32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>) -> tuple<tensor<60x13xf32>, tensor<f32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>
    return %9 : tuple<tensor<60x13xf32>, tensor<f32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>
  }
  func private @jvp_elu_.36(%arg0: tensor<60x13xf32>, %arg1: tensor<f32>) -> tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<60x13xf32>
    %0 = linalg.init_tensor [60, 13] : tensor<60x13xi1>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<60x13xf32>, tensor<60x13xf32>) outs(%0 : tensor<60x13xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %20 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %20 : i1
    } -> tensor<60x13xi1>
    %2 = linalg.init_tensor [60, 13] : tensor<60x13xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%2 : tensor<60x13xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<60x13xf32>
    %4 = linalg.init_tensor [60, 13] : tensor<60x13xi1>
    %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<60x13xf32>, tensor<60x13xf32>) outs(%4 : tensor<60x13xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %20 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %20 : i1
    } -> tensor<60x13xi1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<f32>
    %6 = call @jvp__where_.21(%5, %cst_1, %arg0) {xla_shape = "(f32[60,13]{1,0}, pred[60,13]{1,0}, f32[60,13]{1,0})"} : (tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>) -> tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>
    %7 = "mhlo.get_tuple_element"(%6) {index = 0 : i32} : (tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>) -> tensor<60x13xf32>
    %8 = linalg.init_tensor [60, 13] : tensor<60x13xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<60x13xf32>) outs(%8 : tensor<60x13xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %20 = math.expm1 %arg2 : f32
      linalg.yield %20 : f32
    } -> tensor<60x13xf32>
    %10 = linalg.init_tensor [60, 13] : tensor<60x13xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %9 : tensor<60x13xf32>, tensor<60x13xf32>) outs(%10 : tensor<60x13xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %20 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %20 : f32
    } -> tensor<60x13xf32>
    %12 = call @jvp__where__0.30(%1, %arg0, %11) {xla_shape = "(f32[60,13]{1,0}, pred[60,13]{1,0})"} : (tensor<60x13xi1>, tensor<60x13xf32>, tensor<60x13xf32>) -> tuple<tensor<60x13xf32>, tensor<60x13xi1>>
    %13 = "mhlo.get_tuple_element"(%12) {index = 0 : i32} : (tuple<tensor<60x13xf32>, tensor<60x13xi1>>) -> tensor<60x13xf32>
    %14 = "mhlo.get_tuple_element"(%12) {index = 1 : i32} : (tuple<tensor<60x13xf32>, tensor<60x13xi1>>) -> tensor<60x13xi1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<f32>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<60x13xf32>
    %15 = linalg.init_tensor [60, 13] : tensor<60x13xf32>
    %16 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%9, %cst_3 : tensor<60x13xf32>, tensor<60x13xf32>) outs(%15 : tensor<60x13xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %20 = arith.addf %arg2, %arg3 : f32
      linalg.yield %20 : f32
    } -> tensor<60x13xf32>
    %17 = "mhlo.get_tuple_element"(%6) {index = 1 : i32} : (tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>) -> tensor<60x13xi1>
    %18 = "mhlo.get_tuple_element"(%6) {index = 2 : i32} : (tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>) -> tensor<60x13xf32>
    %19 = "mhlo.tuple"(%13, %14, %arg1, %16, %17, %18) {xla_shape = "(f32[60,13]{1,0}, pred[60,13]{1,0}, f32[], f32[60,13]{1,0}, pred[60,13]{1,0}, /*index=5*/f32[60,13]{1,0})"} : (tensor<60x13xf32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>) -> tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>
    return %19 : tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<f32>, tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>
  }
  func private @jvp__where_.21(%arg0: tensor<60x13xi1>, %arg1: tensor<f32>, %arg2: tensor<60x13xf32>) -> tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>> {
    %0 = linalg.init_tensor [60, 13] : tensor<60x13xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%0 : tensor<60x13xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<60x13xf32>
    %2 = linalg.init_tensor [60, 13] : tensor<60x13xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1, %arg2 : tensor<60x13xi1>, tensor<60x13xf32>, tensor<60x13xf32>) outs(%2 : tensor<60x13xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %5 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %5 : f32
    } -> tensor<60x13xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<60x13xf32>
    %4 = "mhlo.tuple"(%3, %arg0, %cst_0) {xla_shape = "(f32[60,13]{1,0}, pred[60,13]{1,0}, f32[60,13]{1,0})"} : (tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>) -> tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>
    return %4 : tuple<tensor<60x13xf32>, tensor<60x13xi1>, tensor<60x13xf32>>
  }
  func private @jvp__where__0.30(%arg0: tensor<60x13xi1>, %arg1: tensor<60x13xf32>, %arg2: tensor<60x13xf32>) -> tuple<tensor<60x13xf32>, tensor<60x13xi1>> {
    %0 = linalg.init_tensor [60, 13] : tensor<60x13xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<60x13xi1>, tensor<60x13xf32>, tensor<60x13xf32>) outs(%0 : tensor<60x13xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %3 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %3 : f32
    } -> tensor<60x13xf32>
    %2 = "mhlo.tuple"(%1, %arg0) {xla_shape = "(f32[60,13]{1,0}, pred[60,13]{1,0})"} : (tensor<60x13xf32>, tensor<60x13xi1>) -> tuple<tensor<60x13xf32>, tensor<60x13xi1>>
    return %2 : tuple<tensor<60x13xf32>, tensor<60x13xi1>>
  }
  func private @jvp_selu__1.124(%arg0: tensor<60x32xf32>) -> tuple<tensor<60x32xf32>, tensor<f32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>> {
    %cst = arith.constant dense<1.67326319> : tensor<f32>
    %0 = call @jvp_elu__2.102(%arg0, %cst) {xla_shape = "(f32[60,32]{1,0}, pred[60,32]{1,0}, f32[], f32[60,32]{1,0}, pred[60,32]{1,0}, /*index=5*/f32[60,32]{1,0})"} : (tensor<60x32xf32>, tensor<f32>) -> tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>
    %1 = "mhlo.get_tuple_element"(%0) {index = 0 : i32} : (tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>) -> tensor<60x32xf32>
    %cst_0 = arith.constant dense<1.05070102> : tensor<f32>
    %cst_1 = arith.constant dense<1.05070102> : tensor<60x32xf32>
    %2 = linalg.init_tensor [60, 32] : tensor<60x32xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %cst_1 : tensor<60x32xf32>, tensor<60x32xf32>) outs(%2 : tensor<60x32xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %10 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %10 : f32
    } -> tensor<60x32xf32>
    %cst_2 = arith.constant dense<1.05070102> : tensor<f32>
    %4 = "mhlo.get_tuple_element"(%0) {index = 1 : i32} : (tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>) -> tensor<60x32xi1>
    %5 = "mhlo.get_tuple_element"(%0) {index = 2 : i32} : (tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>) -> tensor<f32>
    %6 = "mhlo.get_tuple_element"(%0) {index = 3 : i32} : (tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>) -> tensor<60x32xf32>
    %7 = "mhlo.get_tuple_element"(%0) {index = 4 : i32} : (tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>) -> tensor<60x32xi1>
    %8 = "mhlo.get_tuple_element"(%0) {index = 5 : i32} : (tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>) -> tensor<60x32xf32>
    %9 = "mhlo.tuple"(%3, %cst_2, %4, %5, %6, %7, %8) {xla_shape = "(f32[60,32]{1,0}, f32[], pred[60,32]{1,0}, f32[], f32[60,32]{1,0}, /*index=5*/pred[60,32]{1,0}, f32[60,32]{1,0})"} : (tensor<60x32xf32>, tensor<f32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>) -> tuple<tensor<60x32xf32>, tensor<f32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>
    return %9 : tuple<tensor<60x32xf32>, tensor<f32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>
  }
  func private @jvp_elu__2.102(%arg0: tensor<60x32xf32>, %arg1: tensor<f32>) -> tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<60x32xf32>
    %0 = linalg.init_tensor [60, 32] : tensor<60x32xi1>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<60x32xf32>, tensor<60x32xf32>) outs(%0 : tensor<60x32xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %20 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %20 : i1
    } -> tensor<60x32xi1>
    %2 = linalg.init_tensor [60, 32] : tensor<60x32xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%2 : tensor<60x32xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<60x32xf32>
    %4 = linalg.init_tensor [60, 32] : tensor<60x32xi1>
    %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<60x32xf32>, tensor<60x32xf32>) outs(%4 : tensor<60x32xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %20 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %20 : i1
    } -> tensor<60x32xi1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<f32>
    %6 = call @jvp__where__3.87(%5, %cst_1, %arg0) {xla_shape = "(f32[60,32]{1,0}, pred[60,32]{1,0}, f32[60,32]{1,0})"} : (tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>) -> tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>
    %7 = "mhlo.get_tuple_element"(%6) {index = 0 : i32} : (tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>) -> tensor<60x32xf32>
    %8 = linalg.init_tensor [60, 32] : tensor<60x32xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<60x32xf32>) outs(%8 : tensor<60x32xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %20 = math.expm1 %arg2 : f32
      linalg.yield %20 : f32
    } -> tensor<60x32xf32>
    %10 = linalg.init_tensor [60, 32] : tensor<60x32xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %9 : tensor<60x32xf32>, tensor<60x32xf32>) outs(%10 : tensor<60x32xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %20 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %20 : f32
    } -> tensor<60x32xf32>
    %12 = call @jvp__where__4.96(%1, %arg0, %11) {xla_shape = "(f32[60,32]{1,0}, pred[60,32]{1,0})"} : (tensor<60x32xi1>, tensor<60x32xf32>, tensor<60x32xf32>) -> tuple<tensor<60x32xf32>, tensor<60x32xi1>>
    %13 = "mhlo.get_tuple_element"(%12) {index = 0 : i32} : (tuple<tensor<60x32xf32>, tensor<60x32xi1>>) -> tensor<60x32xf32>
    %14 = "mhlo.get_tuple_element"(%12) {index = 1 : i32} : (tuple<tensor<60x32xf32>, tensor<60x32xi1>>) -> tensor<60x32xi1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<f32>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<60x32xf32>
    %15 = linalg.init_tensor [60, 32] : tensor<60x32xf32>
    %16 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%9, %cst_3 : tensor<60x32xf32>, tensor<60x32xf32>) outs(%15 : tensor<60x32xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %20 = arith.addf %arg2, %arg3 : f32
      linalg.yield %20 : f32
    } -> tensor<60x32xf32>
    %17 = "mhlo.get_tuple_element"(%6) {index = 1 : i32} : (tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>) -> tensor<60x32xi1>
    %18 = "mhlo.get_tuple_element"(%6) {index = 2 : i32} : (tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>) -> tensor<60x32xf32>
    %19 = "mhlo.tuple"(%13, %14, %arg1, %16, %17, %18) {xla_shape = "(f32[60,32]{1,0}, pred[60,32]{1,0}, f32[], f32[60,32]{1,0}, pred[60,32]{1,0}, /*index=5*/f32[60,32]{1,0})"} : (tensor<60x32xf32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>) -> tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>
    return %19 : tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<f32>, tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>
  }
  func private @jvp__where__3.87(%arg0: tensor<60x32xi1>, %arg1: tensor<f32>, %arg2: tensor<60x32xf32>) -> tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>> {
    %0 = linalg.init_tensor [60, 32] : tensor<60x32xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%0 : tensor<60x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<60x32xf32>
    %2 = linalg.init_tensor [60, 32] : tensor<60x32xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1, %arg2 : tensor<60x32xi1>, tensor<60x32xf32>, tensor<60x32xf32>) outs(%2 : tensor<60x32xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %5 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %5 : f32
    } -> tensor<60x32xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<60x32xf32>
    %4 = "mhlo.tuple"(%3, %arg0, %cst_0) {xla_shape = "(f32[60,32]{1,0}, pred[60,32]{1,0}, f32[60,32]{1,0})"} : (tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>) -> tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>
    return %4 : tuple<tensor<60x32xf32>, tensor<60x32xi1>, tensor<60x32xf32>>
  }
  func private @jvp__where__4.96(%arg0: tensor<60x32xi1>, %arg1: tensor<60x32xf32>, %arg2: tensor<60x32xf32>) -> tuple<tensor<60x32xf32>, tensor<60x32xi1>> {
    %0 = linalg.init_tensor [60, 32] : tensor<60x32xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<60x32xi1>, tensor<60x32xf32>, tensor<60x32xf32>) outs(%0 : tensor<60x32xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %3 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %3 : f32
    } -> tensor<60x32xf32>
    %2 = "mhlo.tuple"(%1, %arg0) {xla_shape = "(f32[60,32]{1,0}, pred[60,32]{1,0})"} : (tensor<60x32xf32>, tensor<60x32xi1>) -> tuple<tensor<60x32xf32>, tensor<60x32xi1>>
    return %2 : tuple<tensor<60x32xf32>, tensor<60x32xi1>>
  }
  func private @jvp_selu__5.190(%arg0: tensor<60x16xf32>) -> tuple<tensor<60x16xf32>, tensor<f32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>> {
    %cst = arith.constant dense<1.67326319> : tensor<f32>
    %0 = call @jvp_elu__6.168(%arg0, %cst) {xla_shape = "(f32[60,16]{1,0}, pred[60,16]{1,0}, f32[], f32[60,16]{1,0}, pred[60,16]{1,0}, /*index=5*/f32[60,16]{1,0})"} : (tensor<60x16xf32>, tensor<f32>) -> tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>
    %1 = "mhlo.get_tuple_element"(%0) {index = 0 : i32} : (tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>) -> tensor<60x16xf32>
    %cst_0 = arith.constant dense<1.05070102> : tensor<f32>
    %cst_1 = arith.constant dense<1.05070102> : tensor<60x16xf32>
    %2 = linalg.init_tensor [60, 16] : tensor<60x16xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %cst_1 : tensor<60x16xf32>, tensor<60x16xf32>) outs(%2 : tensor<60x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %10 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %10 : f32
    } -> tensor<60x16xf32>
    %cst_2 = arith.constant dense<1.05070102> : tensor<f32>
    %4 = "mhlo.get_tuple_element"(%0) {index = 1 : i32} : (tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>) -> tensor<60x16xi1>
    %5 = "mhlo.get_tuple_element"(%0) {index = 2 : i32} : (tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>) -> tensor<f32>
    %6 = "mhlo.get_tuple_element"(%0) {index = 3 : i32} : (tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>) -> tensor<60x16xf32>
    %7 = "mhlo.get_tuple_element"(%0) {index = 4 : i32} : (tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>) -> tensor<60x16xi1>
    %8 = "mhlo.get_tuple_element"(%0) {index = 5 : i32} : (tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>) -> tensor<60x16xf32>
    %9 = "mhlo.tuple"(%3, %cst_2, %4, %5, %6, %7, %8) {xla_shape = "(f32[60,16]{1,0}, f32[], pred[60,16]{1,0}, f32[], f32[60,16]{1,0}, /*index=5*/pred[60,16]{1,0}, f32[60,16]{1,0})"} : (tensor<60x16xf32>, tensor<f32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>) -> tuple<tensor<60x16xf32>, tensor<f32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>
    return %9 : tuple<tensor<60x16xf32>, tensor<f32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>
  }
  func private @jvp_elu__6.168(%arg0: tensor<60x16xf32>, %arg1: tensor<f32>) -> tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<60x16xf32>
    %0 = linalg.init_tensor [60, 16] : tensor<60x16xi1>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<60x16xf32>, tensor<60x16xf32>) outs(%0 : tensor<60x16xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %20 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %20 : i1
    } -> tensor<60x16xi1>
    %2 = linalg.init_tensor [60, 16] : tensor<60x16xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%2 : tensor<60x16xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<60x16xf32>
    %4 = linalg.init_tensor [60, 16] : tensor<60x16xi1>
    %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<60x16xf32>, tensor<60x16xf32>) outs(%4 : tensor<60x16xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %20 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %20 : i1
    } -> tensor<60x16xi1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<f32>
    %6 = call @jvp__where__7.153(%5, %cst_1, %arg0) {xla_shape = "(f32[60,16]{1,0}, pred[60,16]{1,0}, f32[60,16]{1,0})"} : (tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>) -> tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>
    %7 = "mhlo.get_tuple_element"(%6) {index = 0 : i32} : (tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>) -> tensor<60x16xf32>
    %8 = linalg.init_tensor [60, 16] : tensor<60x16xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<60x16xf32>) outs(%8 : tensor<60x16xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %20 = math.expm1 %arg2 : f32
      linalg.yield %20 : f32
    } -> tensor<60x16xf32>
    %10 = linalg.init_tensor [60, 16] : tensor<60x16xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %9 : tensor<60x16xf32>, tensor<60x16xf32>) outs(%10 : tensor<60x16xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %20 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %20 : f32
    } -> tensor<60x16xf32>
    %12 = call @jvp__where__8.162(%1, %arg0, %11) {xla_shape = "(f32[60,16]{1,0}, pred[60,16]{1,0})"} : (tensor<60x16xi1>, tensor<60x16xf32>, tensor<60x16xf32>) -> tuple<tensor<60x16xf32>, tensor<60x16xi1>>
    %13 = "mhlo.get_tuple_element"(%12) {index = 0 : i32} : (tuple<tensor<60x16xf32>, tensor<60x16xi1>>) -> tensor<60x16xf32>
    %14 = "mhlo.get_tuple_element"(%12) {index = 1 : i32} : (tuple<tensor<60x16xf32>, tensor<60x16xi1>>) -> tensor<60x16xi1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<f32>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<60x16xf32>
    %15 = linalg.init_tensor [60, 16] : tensor<60x16xf32>
    %16 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%9, %cst_3 : tensor<60x16xf32>, tensor<60x16xf32>) outs(%15 : tensor<60x16xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %20 = arith.addf %arg2, %arg3 : f32
      linalg.yield %20 : f32
    } -> tensor<60x16xf32>
    %17 = "mhlo.get_tuple_element"(%6) {index = 1 : i32} : (tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>) -> tensor<60x16xi1>
    %18 = "mhlo.get_tuple_element"(%6) {index = 2 : i32} : (tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>) -> tensor<60x16xf32>
    %19 = "mhlo.tuple"(%13, %14, %arg1, %16, %17, %18) {xla_shape = "(f32[60,16]{1,0}, pred[60,16]{1,0}, f32[], f32[60,16]{1,0}, pred[60,16]{1,0}, /*index=5*/f32[60,16]{1,0})"} : (tensor<60x16xf32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>) -> tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>
    return %19 : tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<f32>, tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>
  }
  func private @jvp__where__7.153(%arg0: tensor<60x16xi1>, %arg1: tensor<f32>, %arg2: tensor<60x16xf32>) -> tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>> {
    %0 = linalg.init_tensor [60, 16] : tensor<60x16xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%0 : tensor<60x16xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<60x16xf32>
    %2 = linalg.init_tensor [60, 16] : tensor<60x16xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1, %arg2 : tensor<60x16xi1>, tensor<60x16xf32>, tensor<60x16xf32>) outs(%2 : tensor<60x16xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %5 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %5 : f32
    } -> tensor<60x16xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<60x16xf32>
    %4 = "mhlo.tuple"(%3, %arg0, %cst_0) {xla_shape = "(f32[60,16]{1,0}, pred[60,16]{1,0}, f32[60,16]{1,0})"} : (tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>) -> tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>
    return %4 : tuple<tensor<60x16xf32>, tensor<60x16xi1>, tensor<60x16xf32>>
  }
  func private @jvp__where__8.162(%arg0: tensor<60x16xi1>, %arg1: tensor<60x16xf32>, %arg2: tensor<60x16xf32>) -> tuple<tensor<60x16xf32>, tensor<60x16xi1>> {
    %0 = linalg.init_tensor [60, 16] : tensor<60x16xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<60x16xi1>, tensor<60x16xf32>, tensor<60x16xf32>) outs(%0 : tensor<60x16xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %3 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %3 : f32
    } -> tensor<60x16xf32>
    %2 = "mhlo.tuple"(%1, %arg0) {xla_shape = "(f32[60,16]{1,0}, pred[60,16]{1,0})"} : (tensor<60x16xf32>, tensor<60x16xi1>) -> tuple<tensor<60x16xf32>, tensor<60x16xi1>>
    return %2 : tuple<tensor<60x16xf32>, tensor<60x16xi1>>
  }
  func private @jvp_selu__9.256(%arg0: tensor<60x8xf32>) -> tuple<tensor<60x8xf32>, tensor<f32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>> {
    %cst = arith.constant dense<1.67326319> : tensor<f32>
    %0 = call @jvp_elu__10.234(%arg0, %cst) {xla_shape = "(f32[60,8]{1,0}, pred[60,8]{1,0}, f32[], f32[60,8]{1,0}, pred[60,8]{1,0}, /*index=5*/f32[60,8]{1,0})"} : (tensor<60x8xf32>, tensor<f32>) -> tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>
    %1 = "mhlo.get_tuple_element"(%0) {index = 0 : i32} : (tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>) -> tensor<60x8xf32>
    %cst_0 = arith.constant dense<1.05070102> : tensor<f32>
    %cst_1 = arith.constant dense<1.05070102> : tensor<60x8xf32>
    %2 = linalg.init_tensor [60, 8] : tensor<60x8xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %cst_1 : tensor<60x8xf32>, tensor<60x8xf32>) outs(%2 : tensor<60x8xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %10 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %10 : f32
    } -> tensor<60x8xf32>
    %cst_2 = arith.constant dense<1.05070102> : tensor<f32>
    %4 = "mhlo.get_tuple_element"(%0) {index = 1 : i32} : (tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>) -> tensor<60x8xi1>
    %5 = "mhlo.get_tuple_element"(%0) {index = 2 : i32} : (tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>) -> tensor<f32>
    %6 = "mhlo.get_tuple_element"(%0) {index = 3 : i32} : (tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>) -> tensor<60x8xf32>
    %7 = "mhlo.get_tuple_element"(%0) {index = 4 : i32} : (tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>) -> tensor<60x8xi1>
    %8 = "mhlo.get_tuple_element"(%0) {index = 5 : i32} : (tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>) -> tensor<60x8xf32>
    %9 = "mhlo.tuple"(%3, %cst_2, %4, %5, %6, %7, %8) {xla_shape = "(f32[60,8]{1,0}, f32[], pred[60,8]{1,0}, f32[], f32[60,8]{1,0}, /*index=5*/pred[60,8]{1,0}, f32[60,8]{1,0})"} : (tensor<60x8xf32>, tensor<f32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>) -> tuple<tensor<60x8xf32>, tensor<f32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>
    return %9 : tuple<tensor<60x8xf32>, tensor<f32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>
  }
  func private @jvp_elu__10.234(%arg0: tensor<60x8xf32>, %arg1: tensor<f32>) -> tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<60x8xf32>
    %0 = linalg.init_tensor [60, 8] : tensor<60x8xi1>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<60x8xf32>, tensor<60x8xf32>) outs(%0 : tensor<60x8xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %20 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %20 : i1
    } -> tensor<60x8xi1>
    %2 = linalg.init_tensor [60, 8] : tensor<60x8xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%2 : tensor<60x8xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<60x8xf32>
    %4 = linalg.init_tensor [60, 8] : tensor<60x8xi1>
    %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<60x8xf32>, tensor<60x8xf32>) outs(%4 : tensor<60x8xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %20 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %20 : i1
    } -> tensor<60x8xi1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<f32>
    %6 = call @jvp__where__11.219(%5, %cst_1, %arg0) {xla_shape = "(f32[60,8]{1,0}, pred[60,8]{1,0}, f32[60,8]{1,0})"} : (tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>) -> tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>
    %7 = "mhlo.get_tuple_element"(%6) {index = 0 : i32} : (tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>) -> tensor<60x8xf32>
    %8 = linalg.init_tensor [60, 8] : tensor<60x8xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<60x8xf32>) outs(%8 : tensor<60x8xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %20 = math.expm1 %arg2 : f32
      linalg.yield %20 : f32
    } -> tensor<60x8xf32>
    %10 = linalg.init_tensor [60, 8] : tensor<60x8xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %9 : tensor<60x8xf32>, tensor<60x8xf32>) outs(%10 : tensor<60x8xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %20 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %20 : f32
    } -> tensor<60x8xf32>
    %12 = call @jvp__where__12.228(%1, %arg0, %11) {xla_shape = "(f32[60,8]{1,0}, pred[60,8]{1,0})"} : (tensor<60x8xi1>, tensor<60x8xf32>, tensor<60x8xf32>) -> tuple<tensor<60x8xf32>, tensor<60x8xi1>>
    %13 = "mhlo.get_tuple_element"(%12) {index = 0 : i32} : (tuple<tensor<60x8xf32>, tensor<60x8xi1>>) -> tensor<60x8xf32>
    %14 = "mhlo.get_tuple_element"(%12) {index = 1 : i32} : (tuple<tensor<60x8xf32>, tensor<60x8xi1>>) -> tensor<60x8xi1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<f32>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<60x8xf32>
    %15 = linalg.init_tensor [60, 8] : tensor<60x8xf32>
    %16 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%9, %cst_3 : tensor<60x8xf32>, tensor<60x8xf32>) outs(%15 : tensor<60x8xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %20 = arith.addf %arg2, %arg3 : f32
      linalg.yield %20 : f32
    } -> tensor<60x8xf32>
    %17 = "mhlo.get_tuple_element"(%6) {index = 1 : i32} : (tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>) -> tensor<60x8xi1>
    %18 = "mhlo.get_tuple_element"(%6) {index = 2 : i32} : (tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>) -> tensor<60x8xf32>
    %19 = "mhlo.tuple"(%13, %14, %arg1, %16, %17, %18) {xla_shape = "(f32[60,8]{1,0}, pred[60,8]{1,0}, f32[], f32[60,8]{1,0}, pred[60,8]{1,0}, /*index=5*/f32[60,8]{1,0})"} : (tensor<60x8xf32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>) -> tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>
    return %19 : tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<f32>, tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>
  }
  func private @jvp__where__11.219(%arg0: tensor<60x8xi1>, %arg1: tensor<f32>, %arg2: tensor<60x8xf32>) -> tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>> {
    %0 = linalg.init_tensor [60, 8] : tensor<60x8xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%0 : tensor<60x8xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<60x8xf32>
    %2 = linalg.init_tensor [60, 8] : tensor<60x8xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1, %arg2 : tensor<60x8xi1>, tensor<60x8xf32>, tensor<60x8xf32>) outs(%2 : tensor<60x8xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %5 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %5 : f32
    } -> tensor<60x8xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<60x8xf32>
    %4 = "mhlo.tuple"(%3, %arg0, %cst_0) {xla_shape = "(f32[60,8]{1,0}, pred[60,8]{1,0}, f32[60,8]{1,0})"} : (tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>) -> tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>
    return %4 : tuple<tensor<60x8xf32>, tensor<60x8xi1>, tensor<60x8xf32>>
  }
  func private @jvp__where__12.228(%arg0: tensor<60x8xi1>, %arg1: tensor<60x8xf32>, %arg2: tensor<60x8xf32>) -> tuple<tensor<60x8xf32>, tensor<60x8xi1>> {
    %0 = linalg.init_tensor [60, 8] : tensor<60x8xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<60x8xi1>, tensor<60x8xf32>, tensor<60x8xf32>) outs(%0 : tensor<60x8xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %3 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %3 : f32
    } -> tensor<60x8xf32>
    %2 = "mhlo.tuple"(%1, %arg0) {xla_shape = "(f32[60,8]{1,0}, pred[60,8]{1,0})"} : (tensor<60x8xf32>, tensor<60x8xi1>) -> tuple<tensor<60x8xf32>, tensor<60x8xi1>>
    return %2 : tuple<tensor<60x8xf32>, tensor<60x8xi1>>
  }
  func private @jvp_selu__13.322(%arg0: tensor<60x4xf32>) -> tuple<tensor<60x4xf32>, tensor<f32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>> {
    %cst = arith.constant dense<1.67326319> : tensor<f32>
    %0 = call @jvp_elu__14.300(%arg0, %cst) {xla_shape = "(f32[60,4]{1,0}, pred[60,4]{1,0}, f32[], f32[60,4]{1,0}, pred[60,4]{1,0}, /*index=5*/f32[60,4]{1,0})"} : (tensor<60x4xf32>, tensor<f32>) -> tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>
    %1 = "mhlo.get_tuple_element"(%0) {index = 0 : i32} : (tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>) -> tensor<60x4xf32>
    %cst_0 = arith.constant dense<1.05070102> : tensor<f32>
    %cst_1 = arith.constant dense<1.05070102> : tensor<60x4xf32>
    %2 = linalg.init_tensor [60, 4] : tensor<60x4xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %cst_1 : tensor<60x4xf32>, tensor<60x4xf32>) outs(%2 : tensor<60x4xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %10 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %10 : f32
    } -> tensor<60x4xf32>
    %cst_2 = arith.constant dense<1.05070102> : tensor<f32>
    %4 = "mhlo.get_tuple_element"(%0) {index = 1 : i32} : (tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>) -> tensor<60x4xi1>
    %5 = "mhlo.get_tuple_element"(%0) {index = 2 : i32} : (tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>) -> tensor<f32>
    %6 = "mhlo.get_tuple_element"(%0) {index = 3 : i32} : (tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>) -> tensor<60x4xf32>
    %7 = "mhlo.get_tuple_element"(%0) {index = 4 : i32} : (tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>) -> tensor<60x4xi1>
    %8 = "mhlo.get_tuple_element"(%0) {index = 5 : i32} : (tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>) -> tensor<60x4xf32>
    %9 = "mhlo.tuple"(%3, %cst_2, %4, %5, %6, %7, %8) {xla_shape = "(f32[60,4]{1,0}, f32[], pred[60,4]{1,0}, f32[], f32[60,4]{1,0}, /*index=5*/pred[60,4]{1,0}, f32[60,4]{1,0})"} : (tensor<60x4xf32>, tensor<f32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>) -> tuple<tensor<60x4xf32>, tensor<f32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>
    return %9 : tuple<tensor<60x4xf32>, tensor<f32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>
  }
  func private @jvp_elu__14.300(%arg0: tensor<60x4xf32>, %arg1: tensor<f32>) -> tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<60x4xf32>
    %0 = linalg.init_tensor [60, 4] : tensor<60x4xi1>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<60x4xf32>, tensor<60x4xf32>) outs(%0 : tensor<60x4xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %20 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %20 : i1
    } -> tensor<60x4xi1>
    %2 = linalg.init_tensor [60, 4] : tensor<60x4xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%2 : tensor<60x4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<60x4xf32>
    %4 = linalg.init_tensor [60, 4] : tensor<60x4xi1>
    %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<60x4xf32>, tensor<60x4xf32>) outs(%4 : tensor<60x4xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %20 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %20 : i1
    } -> tensor<60x4xi1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<f32>
    %6 = call @jvp__where__15.285(%5, %cst_1, %arg0) {xla_shape = "(f32[60,4]{1,0}, pred[60,4]{1,0}, f32[60,4]{1,0})"} : (tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>) -> tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>
    %7 = "mhlo.get_tuple_element"(%6) {index = 0 : i32} : (tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>) -> tensor<60x4xf32>
    %8 = linalg.init_tensor [60, 4] : tensor<60x4xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<60x4xf32>) outs(%8 : tensor<60x4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %20 = math.expm1 %arg2 : f32
      linalg.yield %20 : f32
    } -> tensor<60x4xf32>
    %10 = linalg.init_tensor [60, 4] : tensor<60x4xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %9 : tensor<60x4xf32>, tensor<60x4xf32>) outs(%10 : tensor<60x4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %20 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %20 : f32
    } -> tensor<60x4xf32>
    %12 = call @jvp__where__16.294(%1, %arg0, %11) {xla_shape = "(f32[60,4]{1,0}, pred[60,4]{1,0})"} : (tensor<60x4xi1>, tensor<60x4xf32>, tensor<60x4xf32>) -> tuple<tensor<60x4xf32>, tensor<60x4xi1>>
    %13 = "mhlo.get_tuple_element"(%12) {index = 0 : i32} : (tuple<tensor<60x4xf32>, tensor<60x4xi1>>) -> tensor<60x4xf32>
    %14 = "mhlo.get_tuple_element"(%12) {index = 1 : i32} : (tuple<tensor<60x4xf32>, tensor<60x4xi1>>) -> tensor<60x4xi1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<f32>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<60x4xf32>
    %15 = linalg.init_tensor [60, 4] : tensor<60x4xf32>
    %16 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%9, %cst_3 : tensor<60x4xf32>, tensor<60x4xf32>) outs(%15 : tensor<60x4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %20 = arith.addf %arg2, %arg3 : f32
      linalg.yield %20 : f32
    } -> tensor<60x4xf32>
    %17 = "mhlo.get_tuple_element"(%6) {index = 1 : i32} : (tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>) -> tensor<60x4xi1>
    %18 = "mhlo.get_tuple_element"(%6) {index = 2 : i32} : (tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>) -> tensor<60x4xf32>
    %19 = "mhlo.tuple"(%13, %14, %arg1, %16, %17, %18) {xla_shape = "(f32[60,4]{1,0}, pred[60,4]{1,0}, f32[], f32[60,4]{1,0}, pred[60,4]{1,0}, /*index=5*/f32[60,4]{1,0})"} : (tensor<60x4xf32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>) -> tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>
    return %19 : tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<f32>, tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>
  }
  func private @jvp__where__15.285(%arg0: tensor<60x4xi1>, %arg1: tensor<f32>, %arg2: tensor<60x4xf32>) -> tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>> {
    %0 = linalg.init_tensor [60, 4] : tensor<60x4xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%0 : tensor<60x4xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<60x4xf32>
    %2 = linalg.init_tensor [60, 4] : tensor<60x4xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1, %arg2 : tensor<60x4xi1>, tensor<60x4xf32>, tensor<60x4xf32>) outs(%2 : tensor<60x4xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %5 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %5 : f32
    } -> tensor<60x4xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<60x4xf32>
    %4 = "mhlo.tuple"(%3, %arg0, %cst_0) {xla_shape = "(f32[60,4]{1,0}, pred[60,4]{1,0}, f32[60,4]{1,0})"} : (tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>) -> tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>
    return %4 : tuple<tensor<60x4xf32>, tensor<60x4xi1>, tensor<60x4xf32>>
  }
  func private @jvp__where__16.294(%arg0: tensor<60x4xi1>, %arg1: tensor<60x4xf32>, %arg2: tensor<60x4xf32>) -> tuple<tensor<60x4xf32>, tensor<60x4xi1>> {
    %0 = linalg.init_tensor [60, 4] : tensor<60x4xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<60x4xi1>, tensor<60x4xf32>, tensor<60x4xf32>) outs(%0 : tensor<60x4xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %3 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %3 : f32
    } -> tensor<60x4xf32>
    %2 = "mhlo.tuple"(%1, %arg0) {xla_shape = "(f32[60,4]{1,0}, pred[60,4]{1,0})"} : (tensor<60x4xf32>, tensor<60x4xi1>) -> tuple<tensor<60x4xf32>, tensor<60x4xi1>>
    return %2 : tuple<tensor<60x4xf32>, tensor<60x4xi1>>
  }
  func private @jvp__mse_.355(%arg0: tensor<60x6xf32>, %arg1: tensor<60x6xf32>) -> tuple<tensor<f32>, tensor<f32>, tensor<60x6xf32>> {
    %0 = linalg.init_tensor [60, 6] : tensor<60x6xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<60x6xf32>, tensor<60x6xf32>) outs(%0 : tensor<60x6xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %10 = arith.subf %arg2, %arg3 : f32
      linalg.yield %10 : f32
    } -> tensor<60x6xf32>
    %2 = linalg.init_tensor [60, 6] : tensor<60x6xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %1 : tensor<60x6xf32>, tensor<60x6xf32>) outs(%2 : tensor<60x6xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %10 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %10 : f32
    } -> tensor<60x6xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %4 = linalg.init_tensor [] : tensor<f32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<f32>) -> tensor<f32>
    %6 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction", "reduction"]} ins(%3 : tensor<60x6xf32>) outs(%5 : tensor<f32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %10 = arith.addf %arg2, %arg3 : f32
      linalg.yield %10 : f32
    } -> tensor<f32>
    %cst_1 = arith.constant dense<3.600000e+02> : tensor<f32>
    %7 = linalg.init_tensor [] : tensor<f32>
    %8 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = []} ins(%6, %cst_1 : tensor<f32>, tensor<f32>) outs(%7 : tensor<f32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %10 = arith.divf %arg2, %arg3 : f32
      linalg.yield %10 : f32
    } -> tensor<f32>
    %9 = "mhlo.tuple"(%8, %cst_1, %1) {xla_shape = "(f32[], f32[], f32[60,6]{1,0})"} : (tensor<f32>, tensor<f32>, tensor<60x6xf32>) -> tuple<tensor<f32>, tensor<f32>, tensor<60x6xf32>>
    return %9 : tuple<tensor<f32>, tensor<f32>, tensor<60x6xf32>>
  }
}
