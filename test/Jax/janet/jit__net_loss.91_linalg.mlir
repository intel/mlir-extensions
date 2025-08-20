// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN:                                       --entry-point-result=void --filecheck
// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> ()>
#map3 = affine_map<() -> ()>
module @jit__net_loss.91 {

  func.func private @printMemrefF32(tensor<*xf32>)

  func.func private @callee(%arg0: tensor<13x13xf32>, %arg1: tensor<13xf32>, %arg2: tensor<13x32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<32x16xf32>, %arg5: tensor<16xf32>, %arg6: tensor<16x8xf32>, %arg7: tensor<8xf32>, %arg8: tensor<8x4xf32>, %arg9: tensor<4xf32>, %arg10: tensor<4x6xf32>, %arg11: tensor<6xf32>, %arg12: tensor<20x13xf32>, %arg13: tensor<20x6xf32>) -> tensor<f32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<20x13xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<20x13xf32>) -> tensor<20x13xf32>
    %2 = linalg.matmul ins(%arg12, %arg0 : tensor<20x13xf32>, tensor<13x13xf32>) outs(%1 : tensor<20x13xf32>) -> tensor<20x13xf32>
    %3 = tensor.expand_shape %arg1 [[0, 1]] : tensor<13xf32> into tensor<1x13xf32>
    %4 = tensor.collapse_shape %3 [[0, 1]] : tensor<1x13xf32> into tensor<13xf32>
    %5 = tensor.empty() : tensor<20x13xf32>
    %6 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<13xf32>) outs(%5 : tensor<20x13xf32>) {
    ^bb0(%arg14: f32, %arg15: f32):
      linalg.yield %arg14 : f32
    } -> tensor<20x13xf32>
    %7 = tensor.empty() : tensor<20x13xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2, %6 : tensor<20x13xf32>, tensor<20x13xf32>) outs(%7 : tensor<20x13xf32>) {
    ^bb0(%arg14: f32, %arg15: f32, %arg16: f32):
      %60 = arith.addf %arg14, %arg15 : f32
      linalg.yield %60 : f32
    } -> tensor<20x13xf32>
    %9 = func.call @selu.45(%8) : (tensor<20x13xf32>) -> tensor<20x13xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %10 = tensor.empty() : tensor<20x32xf32>
    %11 = linalg.fill ins(%cst_0 : f32) outs(%10 : tensor<20x32xf32>) -> tensor<20x32xf32>
    %12 = linalg.matmul ins(%9, %arg2 : tensor<20x13xf32>, tensor<13x32xf32>) outs(%11 : tensor<20x32xf32>) -> tensor<20x32xf32>
    %13 = tensor.expand_shape %arg3 [[0, 1]] : tensor<32xf32> into tensor<1x32xf32>
    %14 = tensor.collapse_shape %13 [[0, 1]] : tensor<1x32xf32> into tensor<32xf32>
    %15 = tensor.empty() : tensor<20x32xf32>
    %16 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<32xf32>) outs(%15 : tensor<20x32xf32>) {
    ^bb0(%arg14: f32, %arg15: f32):
      linalg.yield %arg14 : f32
    } -> tensor<20x32xf32>
    %17 = tensor.empty() : tensor<20x32xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%12, %16 : tensor<20x32xf32>, tensor<20x32xf32>) outs(%17 : tensor<20x32xf32>) {
    ^bb0(%arg14: f32, %arg15: f32, %arg16: f32):
      %60 = arith.addf %arg14, %arg15 : f32
      linalg.yield %60 : f32
    } -> tensor<20x32xf32>
    %19 = func.call @selu_1.83(%18) : (tensor<20x32xf32>) -> tensor<20x32xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %20 = tensor.empty() : tensor<20x16xf32>
    %21 = linalg.fill ins(%cst_1 : f32) outs(%20 : tensor<20x16xf32>) -> tensor<20x16xf32>
    %22 = linalg.matmul ins(%19, %arg4 : tensor<20x32xf32>, tensor<32x16xf32>) outs(%21 : tensor<20x16xf32>) -> tensor<20x16xf32>
    %23 = tensor.expand_shape %arg5 [[0, 1]] : tensor<16xf32> into tensor<1x16xf32>
    %24 = tensor.collapse_shape %23 [[0, 1]] : tensor<1x16xf32> into tensor<16xf32>
    %25 = tensor.empty() : tensor<20x16xf32>
    %26 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%24 : tensor<16xf32>) outs(%25 : tensor<20x16xf32>) {
    ^bb0(%arg14: f32, %arg15: f32):
      linalg.yield %arg14 : f32
    } -> tensor<20x16xf32>
    %27 = tensor.empty() : tensor<20x16xf32>
    %28 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%22, %26 : tensor<20x16xf32>, tensor<20x16xf32>) outs(%27 : tensor<20x16xf32>) {
    ^bb0(%arg14: f32, %arg15: f32, %arg16: f32):
      %60 = arith.addf %arg14, %arg15 : f32
      linalg.yield %60 : f32
    } -> tensor<20x16xf32>
    %29 = func.call @selu_5.121(%28) : (tensor<20x16xf32>) -> tensor<20x16xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %30 = tensor.empty() : tensor<20x8xf32>
    %31 = linalg.fill ins(%cst_2 : f32) outs(%30 : tensor<20x8xf32>) -> tensor<20x8xf32>
    %32 = linalg.matmul ins(%29, %arg6 : tensor<20x16xf32>, tensor<16x8xf32>) outs(%31 : tensor<20x8xf32>) -> tensor<20x8xf32>
    %33 = tensor.expand_shape %arg7 [[0, 1]] : tensor<8xf32> into tensor<1x8xf32>
    %34 = tensor.collapse_shape %33 [[0, 1]] : tensor<1x8xf32> into tensor<8xf32>
    %35 = tensor.empty() : tensor<20x8xf32>
    %36 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%34 : tensor<8xf32>) outs(%35 : tensor<20x8xf32>) {
    ^bb0(%arg14: f32, %arg15: f32):
      linalg.yield %arg14 : f32
    } -> tensor<20x8xf32>
    %37 = tensor.empty() : tensor<20x8xf32>
    %38 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%32, %36 : tensor<20x8xf32>, tensor<20x8xf32>) outs(%37 : tensor<20x8xf32>) {
    ^bb0(%arg14: f32, %arg15: f32, %arg16: f32):
      %60 = arith.addf %arg14, %arg15 : f32
      linalg.yield %60 : f32
    } -> tensor<20x8xf32>
    %39 = func.call @selu_9.159(%38) : (tensor<20x8xf32>) -> tensor<20x8xf32>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %40 = tensor.empty() : tensor<20x4xf32>
    %41 = linalg.fill ins(%cst_3 : f32) outs(%40 : tensor<20x4xf32>) -> tensor<20x4xf32>
    %42 = linalg.matmul ins(%39, %arg8 : tensor<20x8xf32>, tensor<8x4xf32>) outs(%41 : tensor<20x4xf32>) -> tensor<20x4xf32>
    %43 = tensor.expand_shape %arg9 [[0, 1]] : tensor<4xf32> into tensor<1x4xf32>
    %44 = tensor.collapse_shape %43 [[0, 1]] : tensor<1x4xf32> into tensor<4xf32>
    %45 = tensor.empty() : tensor<20x4xf32>
    %46 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%44 : tensor<4xf32>) outs(%45 : tensor<20x4xf32>) {
    ^bb0(%arg14: f32, %arg15: f32):
      linalg.yield %arg14 : f32
    } -> tensor<20x4xf32>
    %47 = tensor.empty() : tensor<20x4xf32>
    %48 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%42, %46 : tensor<20x4xf32>, tensor<20x4xf32>) outs(%47 : tensor<20x4xf32>) {
    ^bb0(%arg14: f32, %arg15: f32, %arg16: f32):
      %60 = arith.addf %arg14, %arg15 : f32
      linalg.yield %60 : f32
    } -> tensor<20x4xf32>
    %49 = func.call @selu_13.197(%48) : (tensor<20x4xf32>) -> tensor<20x4xf32>
    %cst_4 = arith.constant 0.000000e+00 : f32
    %50 = tensor.empty() : tensor<20x6xf32>
    %51 = linalg.fill ins(%cst_4 : f32) outs(%50 : tensor<20x6xf32>) -> tensor<20x6xf32>
    %52 = linalg.matmul ins(%49, %arg10 : tensor<20x4xf32>, tensor<4x6xf32>) outs(%51 : tensor<20x6xf32>) -> tensor<20x6xf32>
    %53 = tensor.expand_shape %arg11 [[0, 1]] : tensor<6xf32> into tensor<1x6xf32>
    %54 = tensor.collapse_shape %53 [[0, 1]] : tensor<1x6xf32> into tensor<6xf32>
    %55 = tensor.empty() : tensor<20x6xf32>
    %56 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%54 : tensor<6xf32>) outs(%55 : tensor<20x6xf32>) {
    ^bb0(%arg14: f32, %arg15: f32):
      linalg.yield %arg14 : f32
    } -> tensor<20x6xf32>
    %57 = tensor.empty() : tensor<20x6xf32>
    %58 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%52, %56 : tensor<20x6xf32>, tensor<20x6xf32>) outs(%57 : tensor<20x6xf32>) {
    ^bb0(%arg14: f32, %arg15: f32, %arg16: f32):
      %60 = arith.addf %arg14, %arg15 : f32
      linalg.yield %60 : f32
    } -> tensor<20x6xf32>
    %59 = func.call @_mse.215(%58, %arg13) : (tensor<20x6xf32>, tensor<20x6xf32>) -> tensor<f32>
    return %59 : tensor<f32>
  }
  func.func private @selu.45(%arg0: tensor<20x13xf32>) -> tensor<20x13xf32> {
    %cst = arith.constant dense<1.67326319> : tensor<f32>
    %0 = func.call @elu.32(%arg0, %cst) : (tensor<20x13xf32>, tensor<f32>) -> tensor<20x13xf32>
    %cst_0 = arith.constant dense<1.05070102> : tensor<f32>
    %cst_1 = arith.constant dense<1.05070102> : tensor<20x13xf32>
    %1 = tensor.empty() : tensor<20x13xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%0, %cst_1 : tensor<20x13xf32>, tensor<20x13xf32>) outs(%1 : tensor<20x13xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %3 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %3 : f32
    } -> tensor<20x13xf32>
    return %2 : tensor<20x13xf32>
  }
  func.func private @elu.32(%arg0: tensor<20x13xf32>, %arg1: tensor<f32>) -> tensor<20x13xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<20x13xf32>
    %0 = tensor.empty() : tensor<20x13xi1>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<20x13xf32>, tensor<20x13xf32>) outs(%0 : tensor<20x13xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %12 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %12 : i1
    } -> tensor<20x13xi1>
    %2 = tensor.empty() : tensor<20x13xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%2 : tensor<20x13xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<20x13xf32>
    %4 = tensor.empty() : tensor<20x13xi1>
    %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<20x13xf32>, tensor<20x13xf32>) outs(%4 : tensor<20x13xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %12 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %12 : i1
    } -> tensor<20x13xi1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<f32>
    %6 = func.call @_where.21(%5, %cst_1, %arg0) : (tensor<20x13xi1>, tensor<f32>, tensor<20x13xf32>) -> tensor<20x13xf32>
    %7 = tensor.empty() : tensor<20x13xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<20x13xf32>) outs(%7 : tensor<20x13xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %12 = math.expm1 %arg2 : f32
      linalg.yield %12 : f32
    } -> tensor<20x13xf32>
    %9 = tensor.empty() : tensor<20x13xf32>
    %10 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %8 : tensor<20x13xf32>, tensor<20x13xf32>) outs(%9 : tensor<20x13xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %12 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %12 : f32
    } -> tensor<20x13xf32>
    %11 = func.call @_where_0.27(%1, %arg0, %10) : (tensor<20x13xi1>, tensor<20x13xf32>, tensor<20x13xf32>) -> tensor<20x13xf32>
    return %11 : tensor<20x13xf32>
  }
  func.func private @_where.21(%arg0: tensor<20x13xi1>, %arg1: tensor<f32>, %arg2: tensor<20x13xf32>) -> tensor<20x13xf32> {
    %0 = tensor.empty() : tensor<20x13xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%0 : tensor<20x13xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<20x13xf32>
    %2 = tensor.empty() : tensor<20x13xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1, %arg2 : tensor<20x13xi1>, tensor<20x13xf32>, tensor<20x13xf32>) outs(%2 : tensor<20x13xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %4 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %4 : f32
    } -> tensor<20x13xf32>
    return %3 : tensor<20x13xf32>
  }
  func.func private @_where_0.27(%arg0: tensor<20x13xi1>, %arg1: tensor<20x13xf32>, %arg2: tensor<20x13xf32>) -> tensor<20x13xf32> {
    %0 = tensor.empty() : tensor<20x13xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<20x13xi1>, tensor<20x13xf32>, tensor<20x13xf32>) outs(%0 : tensor<20x13xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %2 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %2 : f32
    } -> tensor<20x13xf32>
    return %1 : tensor<20x13xf32>
  }
  func.func private @selu_1.83(%arg0: tensor<20x32xf32>) -> tensor<20x32xf32> {
    %cst = arith.constant dense<1.67326319> : tensor<f32>
    %0 = func.call @elu_2.70(%arg0, %cst) : (tensor<20x32xf32>, tensor<f32>) -> tensor<20x32xf32>
    %cst_0 = arith.constant dense<1.05070102> : tensor<f32>
    %cst_1 = arith.constant dense<1.05070102> : tensor<20x32xf32>
    %1 = tensor.empty() : tensor<20x32xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%0, %cst_1 : tensor<20x32xf32>, tensor<20x32xf32>) outs(%1 : tensor<20x32xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %3 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %3 : f32
    } -> tensor<20x32xf32>
    return %2 : tensor<20x32xf32>
  }
  func.func private @elu_2.70(%arg0: tensor<20x32xf32>, %arg1: tensor<f32>) -> tensor<20x32xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<20x32xf32>
    %0 = tensor.empty() : tensor<20x32xi1>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<20x32xf32>, tensor<20x32xf32>) outs(%0 : tensor<20x32xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %12 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %12 : i1
    } -> tensor<20x32xi1>
    %2 = tensor.empty() : tensor<20x32xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%2 : tensor<20x32xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<20x32xf32>
    %4 = tensor.empty() : tensor<20x32xi1>
    %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<20x32xf32>, tensor<20x32xf32>) outs(%4 : tensor<20x32xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %12 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %12 : i1
    } -> tensor<20x32xi1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<f32>
    %6 = func.call @_where_3.59(%5, %cst_1, %arg0) : (tensor<20x32xi1>, tensor<f32>, tensor<20x32xf32>) -> tensor<20x32xf32>
    %7 = tensor.empty() : tensor<20x32xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<20x32xf32>) outs(%7 : tensor<20x32xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %12 = math.expm1 %arg2 : f32
      linalg.yield %12 : f32
    } -> tensor<20x32xf32>
    %9 = tensor.empty() : tensor<20x32xf32>
    %10 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %8 : tensor<20x32xf32>, tensor<20x32xf32>) outs(%9 : tensor<20x32xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %12 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %12 : f32
    } -> tensor<20x32xf32>
    %11 = func.call @_where_4.65(%1, %arg0, %10) : (tensor<20x32xi1>, tensor<20x32xf32>, tensor<20x32xf32>) -> tensor<20x32xf32>
    return %11 : tensor<20x32xf32>
  }
  func.func private @_where_3.59(%arg0: tensor<20x32xi1>, %arg1: tensor<f32>, %arg2: tensor<20x32xf32>) -> tensor<20x32xf32> {
    %0 = tensor.empty() : tensor<20x32xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%0 : tensor<20x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<20x32xf32>
    %2 = tensor.empty() : tensor<20x32xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1, %arg2 : tensor<20x32xi1>, tensor<20x32xf32>, tensor<20x32xf32>) outs(%2 : tensor<20x32xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %4 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %4 : f32
    } -> tensor<20x32xf32>
    return %3 : tensor<20x32xf32>
  }
  func.func private @_where_4.65(%arg0: tensor<20x32xi1>, %arg1: tensor<20x32xf32>, %arg2: tensor<20x32xf32>) -> tensor<20x32xf32> {
    %0 = tensor.empty() : tensor<20x32xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<20x32xi1>, tensor<20x32xf32>, tensor<20x32xf32>) outs(%0 : tensor<20x32xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %2 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %2 : f32
    } -> tensor<20x32xf32>
    return %1 : tensor<20x32xf32>
  }
  func.func private @selu_5.121(%arg0: tensor<20x16xf32>) -> tensor<20x16xf32> {
    %cst = arith.constant dense<1.67326319> : tensor<f32>
    %0 = func.call @elu_6.108(%arg0, %cst) : (tensor<20x16xf32>, tensor<f32>) -> tensor<20x16xf32>
    %cst_0 = arith.constant dense<1.05070102> : tensor<f32>
    %cst_1 = arith.constant dense<1.05070102> : tensor<20x16xf32>
    %1 = tensor.empty() : tensor<20x16xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%0, %cst_1 : tensor<20x16xf32>, tensor<20x16xf32>) outs(%1 : tensor<20x16xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %3 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %3 : f32
    } -> tensor<20x16xf32>
    return %2 : tensor<20x16xf32>
  }
  func.func private @elu_6.108(%arg0: tensor<20x16xf32>, %arg1: tensor<f32>) -> tensor<20x16xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<20x16xf32>
    %0 = tensor.empty() : tensor<20x16xi1>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<20x16xf32>, tensor<20x16xf32>) outs(%0 : tensor<20x16xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %12 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %12 : i1
    } -> tensor<20x16xi1>
    %2 = tensor.empty() : tensor<20x16xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%2 : tensor<20x16xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<20x16xf32>
    %4 = tensor.empty() : tensor<20x16xi1>
    %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<20x16xf32>, tensor<20x16xf32>) outs(%4 : tensor<20x16xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %12 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %12 : i1
    } -> tensor<20x16xi1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<f32>
    %6 = func.call @_where_7.97(%5, %cst_1, %arg0) : (tensor<20x16xi1>, tensor<f32>, tensor<20x16xf32>) -> tensor<20x16xf32>
    %7 = tensor.empty() : tensor<20x16xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<20x16xf32>) outs(%7 : tensor<20x16xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %12 = math.expm1 %arg2 : f32
      linalg.yield %12 : f32
    } -> tensor<20x16xf32>
    %9 = tensor.empty() : tensor<20x16xf32>
    %10 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %8 : tensor<20x16xf32>, tensor<20x16xf32>) outs(%9 : tensor<20x16xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %12 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %12 : f32
    } -> tensor<20x16xf32>
    %11 = func.call @_where_8.103(%1, %arg0, %10) : (tensor<20x16xi1>, tensor<20x16xf32>, tensor<20x16xf32>) -> tensor<20x16xf32>
    return %11 : tensor<20x16xf32>
  }
  func.func private @_where_7.97(%arg0: tensor<20x16xi1>, %arg1: tensor<f32>, %arg2: tensor<20x16xf32>) -> tensor<20x16xf32> {
    %0 = tensor.empty() : tensor<20x16xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%0 : tensor<20x16xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<20x16xf32>
    %2 = tensor.empty() : tensor<20x16xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1, %arg2 : tensor<20x16xi1>, tensor<20x16xf32>, tensor<20x16xf32>) outs(%2 : tensor<20x16xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %4 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %4 : f32
    } -> tensor<20x16xf32>
    return %3 : tensor<20x16xf32>
  }
  func.func private @_where_8.103(%arg0: tensor<20x16xi1>, %arg1: tensor<20x16xf32>, %arg2: tensor<20x16xf32>) -> tensor<20x16xf32> {
    %0 = tensor.empty() : tensor<20x16xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<20x16xi1>, tensor<20x16xf32>, tensor<20x16xf32>) outs(%0 : tensor<20x16xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %2 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %2 : f32
    } -> tensor<20x16xf32>
    return %1 : tensor<20x16xf32>
  }
  func.func private @selu_9.159(%arg0: tensor<20x8xf32>) -> tensor<20x8xf32> {
    %cst = arith.constant dense<1.67326319> : tensor<f32>
    %0 = func.call @elu_10.146(%arg0, %cst) : (tensor<20x8xf32>, tensor<f32>) -> tensor<20x8xf32>
    %cst_0 = arith.constant dense<1.05070102> : tensor<f32>
    %cst_1 = arith.constant dense<1.05070102> : tensor<20x8xf32>
    %1 = tensor.empty() : tensor<20x8xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%0, %cst_1 : tensor<20x8xf32>, tensor<20x8xf32>) outs(%1 : tensor<20x8xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %3 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %3 : f32
    } -> tensor<20x8xf32>
    return %2 : tensor<20x8xf32>
  }
  func.func private @elu_10.146(%arg0: tensor<20x8xf32>, %arg1: tensor<f32>) -> tensor<20x8xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<20x8xf32>
    %0 = tensor.empty() : tensor<20x8xi1>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<20x8xf32>, tensor<20x8xf32>) outs(%0 : tensor<20x8xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %12 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %12 : i1
    } -> tensor<20x8xi1>
    %2 = tensor.empty() : tensor<20x8xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%2 : tensor<20x8xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<20x8xf32>
    %4 = tensor.empty() : tensor<20x8xi1>
    %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<20x8xf32>, tensor<20x8xf32>) outs(%4 : tensor<20x8xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %12 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %12 : i1
    } -> tensor<20x8xi1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<f32>
    %6 = func.call @_where_11.135(%5, %cst_1, %arg0) : (tensor<20x8xi1>, tensor<f32>, tensor<20x8xf32>) -> tensor<20x8xf32>
    %7 = tensor.empty() : tensor<20x8xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<20x8xf32>) outs(%7 : tensor<20x8xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %12 = math.expm1 %arg2 : f32
      linalg.yield %12 : f32
    } -> tensor<20x8xf32>
    %9 = tensor.empty() : tensor<20x8xf32>
    %10 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %8 : tensor<20x8xf32>, tensor<20x8xf32>) outs(%9 : tensor<20x8xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %12 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %12 : f32
    } -> tensor<20x8xf32>
    %11 = func.call @_where_12.141(%1, %arg0, %10) : (tensor<20x8xi1>, tensor<20x8xf32>, tensor<20x8xf32>) -> tensor<20x8xf32>
    return %11 : tensor<20x8xf32>
  }
  func.func private @_where_11.135(%arg0: tensor<20x8xi1>, %arg1: tensor<f32>, %arg2: tensor<20x8xf32>) -> tensor<20x8xf32> {
    %0 = tensor.empty() : tensor<20x8xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%0 : tensor<20x8xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<20x8xf32>
    %2 = tensor.empty() : tensor<20x8xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1, %arg2 : tensor<20x8xi1>, tensor<20x8xf32>, tensor<20x8xf32>) outs(%2 : tensor<20x8xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %4 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %4 : f32
    } -> tensor<20x8xf32>
    return %3 : tensor<20x8xf32>
  }
  func.func private @_where_12.141(%arg0: tensor<20x8xi1>, %arg1: tensor<20x8xf32>, %arg2: tensor<20x8xf32>) -> tensor<20x8xf32> {
    %0 = tensor.empty() : tensor<20x8xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<20x8xi1>, tensor<20x8xf32>, tensor<20x8xf32>) outs(%0 : tensor<20x8xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %2 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %2 : f32
    } -> tensor<20x8xf32>
    return %1 : tensor<20x8xf32>
  }
  func.func private @selu_13.197(%arg0: tensor<20x4xf32>) -> tensor<20x4xf32> {
    %cst = arith.constant dense<1.67326319> : tensor<f32>
    %0 = func.call @elu_14.184(%arg0, %cst) : (tensor<20x4xf32>, tensor<f32>) -> tensor<20x4xf32>
    %cst_0 = arith.constant dense<1.05070102> : tensor<f32>
    %cst_1 = arith.constant dense<1.05070102> : tensor<20x4xf32>
    %1 = tensor.empty() : tensor<20x4xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%0, %cst_1 : tensor<20x4xf32>, tensor<20x4xf32>) outs(%1 : tensor<20x4xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %3 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %3 : f32
    } -> tensor<20x4xf32>
    return %2 : tensor<20x4xf32>
  }
  func.func private @elu_14.184(%arg0: tensor<20x4xf32>, %arg1: tensor<f32>) -> tensor<20x4xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<20x4xf32>
    %0 = tensor.empty() : tensor<20x4xi1>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<20x4xf32>, tensor<20x4xf32>) outs(%0 : tensor<20x4xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %12 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %12 : i1
    } -> tensor<20x4xi1>
    %2 = tensor.empty() : tensor<20x4xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%2 : tensor<20x4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<20x4xf32>
    %4 = tensor.empty() : tensor<20x4xi1>
    %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<20x4xf32>, tensor<20x4xf32>) outs(%4 : tensor<20x4xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %12 = arith.cmpf ogt, %arg2, %arg3 : f32
      linalg.yield %12 : i1
    } -> tensor<20x4xi1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<f32>
    %6 = func.call @_where_15.173(%5, %cst_1, %arg0) : (tensor<20x4xi1>, tensor<f32>, tensor<20x4xf32>) -> tensor<20x4xf32>
    %7 = tensor.empty() : tensor<20x4xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<20x4xf32>) outs(%7 : tensor<20x4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %12 = math.expm1 %arg2 : f32
      linalg.yield %12 : f32
    } -> tensor<20x4xf32>
    %9 = tensor.empty() : tensor<20x4xf32>
    %10 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %8 : tensor<20x4xf32>, tensor<20x4xf32>) outs(%9 : tensor<20x4xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %12 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %12 : f32
    } -> tensor<20x4xf32>
    %11 = func.call @_where_16.179(%1, %arg0, %10) : (tensor<20x4xi1>, tensor<20x4xf32>, tensor<20x4xf32>) -> tensor<20x4xf32>
    return %11 : tensor<20x4xf32>
  }
  func.func private @_where_15.173(%arg0: tensor<20x4xi1>, %arg1: tensor<f32>, %arg2: tensor<20x4xf32>) -> tensor<20x4xf32> {
    %0 = tensor.empty() : tensor<20x4xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%0 : tensor<20x4xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<20x4xf32>
    %2 = tensor.empty() : tensor<20x4xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1, %arg2 : tensor<20x4xi1>, tensor<20x4xf32>, tensor<20x4xf32>) outs(%2 : tensor<20x4xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %4 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %4 : f32
    } -> tensor<20x4xf32>
    return %3 : tensor<20x4xf32>
  }
  func.func private @_where_16.179(%arg0: tensor<20x4xi1>, %arg1: tensor<20x4xf32>, %arg2: tensor<20x4xf32>) -> tensor<20x4xf32> {
    %0 = tensor.empty() : tensor<20x4xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<20x4xi1>, tensor<20x4xf32>, tensor<20x4xf32>) outs(%0 : tensor<20x4xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %2 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %2 : f32
    } -> tensor<20x4xf32>
    return %1 : tensor<20x4xf32>
  }
  func.func private @_mse.215(%arg0: tensor<20x6xf32>, %arg1: tensor<20x6xf32>) -> tensor<f32> {
    %0 = tensor.empty() : tensor<20x6xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<20x6xf32>, tensor<20x6xf32>) outs(%0 : tensor<20x6xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %9 = arith.subf %arg2, %arg3 : f32
      linalg.yield %9 : f32
    } -> tensor<20x6xf32>
    %2 = tensor.empty() : tensor<20x6xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %1 : tensor<20x6xf32>, tensor<20x6xf32>) outs(%2 : tensor<20x6xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %9 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %9 : f32
    } -> tensor<20x6xf32>
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %4 = tensor.empty() : tensor<f32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<f32>) -> tensor<f32>
    %6 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction", "reduction"]} ins(%3 : tensor<20x6xf32>) outs(%5 : tensor<f32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %9 = arith.addf %arg2, %arg3 : f32
      linalg.yield %9 : f32
    } -> tensor<f32>
    %cst_1 = arith.constant dense<1.200000e+02> : tensor<f32>
    %7 = tensor.empty() : tensor<f32>
    %8 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = []} ins(%6, %cst_1 : tensor<f32>, tensor<f32>) outs(%7 : tensor<f32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %9 = arith.divf %arg2, %arg3 : f32
      linalg.yield %9 : f32
    } -> tensor<f32>
    return %8 : tensor<f32>
  }
  func.func @main() {
    %0 = arith.constant dense<0.01>: tensor<13x13xf32>
    %1 = arith.constant dense<-0.001>: tensor<13xf32>
    %2 = arith.constant dense<0.02>: tensor<13x32xf32>
    %3 = arith.constant dense<0.001>: tensor<32xf32>
    %4 = arith.constant dense<0.03>: tensor<32x16xf32>
    %5 = arith.constant dense<-0.001>: tensor<16xf32>
    %6 = arith.constant dense<-0.04>: tensor<16x8xf32>
    %7 = arith.constant dense<0.001>: tensor<8xf32>
    %8 = arith.constant dense<-0.05>: tensor<8x4xf32>
    %9 = arith.constant dense<-0.001>: tensor<4xf32>
    %10 = arith.constant dense<-0.06>: tensor<4x6xf32>
    %11 = arith.constant dense<0.001>: tensor<6xf32>
    %12 = arith.constant dense<0.02>: tensor<20x13xf32>
    %13 = arith.constant dense<-0.01>: tensor<20x6xf32>
    %14 = func.call @callee(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13) : (tensor<13x13xf32>, tensor<13xf32>, tensor<13x32xf32>, tensor<32xf32>, tensor<32x16xf32>, tensor<16xf32>, tensor<16x8xf32>, tensor<8xf32>, tensor<8x4xf32>, tensor<4xf32>, tensor<4x6xf32>, tensor<6xf32>, tensor<20x13xf32>, tensor<20x6xf32>) -> tensor<f32>
    %unranked = tensor.cast %14 : tensor<f32> to tensor<*xf32>
    func.call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 0 offset = 0 sizes = [] strides = [] data =
    //      CHECK: [0.00013329{{.*}}]
    return
  }
}
