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
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module @jit__unit_scale_traindata.47 {

  func.func private @printMemrefF32(tensor<*xf32>)

  func.func private @callee(%arg0: tensor<6xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<1x6xf32> {
    %0 = func.call @atleast_2d.6(%arg0) : (tensor<6xf32>) -> tensor<1x6xf32>
    %1 = func.call @atleast_1d.10(%arg1) : (tensor<i32>) -> tensor<1xi32>
    %2 = func.call @atleast_1d_0.14(%arg2) : (tensor<i32>) -> tensor<1xi32>
    %3 = tensor.empty() : tensor<1xi1>
    %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%1, %2 : tensor<1xi32>, tensor<1xi32>) outs(%3 : tensor<1xi1>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %21 = arith.cmpi eq, %arg3, %arg4 : i32
      linalg.yield %21 : i1
    } -> tensor<1xi1>
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %5 = func.call @_where_1.28(%4, %cst, %1) : (tensor<1xi1>, tensor<f32>, tensor<1xi32>) -> tensor<1xf32>
    %6 = tensor.expand_shape %5 [[0, 1]] : tensor<1xf32> into tensor<1x1xf32>
    %7 = tensor.collapse_shape %6 [[0, 1]] : tensor<1x1xf32> into tensor<1xf32>
    %8 = tensor.empty() : tensor<1x6xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<1xf32>) outs(%8 : tensor<1x6xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<1x6xf32>
    %10 = tensor.empty() : tensor<1x6xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%0, %9 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%10 : tensor<1x6xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %21 = arith.subf %arg3, %arg4 : f32
      linalg.yield %21 : f32
    } -> tensor<1x6xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<f32>
    %12 = tensor.empty() : tensor<1xi32>
    %13 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%2, %1 : tensor<1xi32>, tensor<1xi32>) outs(%12 : tensor<1xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %21 = arith.subi %arg3, %arg4 : i32
      linalg.yield %21 : i32
    } -> tensor<1xi32>
    %14 = func.call @_where.20(%4, %cst_0, %13) : (tensor<1xi1>, tensor<f32>, tensor<1xi32>) -> tensor<1xf32>
    %15 = tensor.expand_shape %14 [[0, 1]] : tensor<1xf32> into tensor<1x1xf32>
    %16 = tensor.collapse_shape %15 [[0, 1]] : tensor<1x1xf32> into tensor<1xf32>
    %17 = tensor.empty() : tensor<1x6xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<1xf32>) outs(%17 : tensor<1x6xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<1x6xf32>
    %19 = tensor.empty() : tensor<1x6xf32>
    %20 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%11, %18 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%19 : tensor<1x6xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %21 = arith.divf %arg3, %arg4 : f32
      linalg.yield %21 : f32
    } -> tensor<1x6xf32>
    return %20 : tensor<1x6xf32>
  }
  func.func private @atleast_2d.6(%arg0: tensor<6xf32>) -> tensor<1x6xf32> {
    %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<6xf32> into tensor<1x6xf32>
    return %0 : tensor<1x6xf32>
  }
  func.func private @atleast_1d.10(%arg0: tensor<i32>) -> tensor<1xi32> {
    %0 = tensor.expand_shape %arg0 [] : tensor<i32> into tensor<1xi32>
    return %0 : tensor<1xi32>
  }
  func.func private @atleast_1d_0.14(%arg0: tensor<i32>) -> tensor<1xi32> {
    %0 = tensor.expand_shape %arg0 [] : tensor<i32> into tensor<1xi32>
    return %0 : tensor<1xi32>
  }
  func.func private @_where_1.28(%arg0: tensor<1xi1>, %arg1: tensor<f32>, %arg2: tensor<1xi32>) -> tensor<1xf32> {
    %0 = tensor.expand_shape %arg1 [] : tensor<f32> into tensor<1xf32>
    %1 = tensor.empty() : tensor<1xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%arg2 : tensor<1xi32>) outs(%1 : tensor<1xf32>) {
    ^bb0(%arg3: i32, %arg4: f32):
      %5 = arith.sitofp %arg3 : i32 to f32
      linalg.yield %5 : f32
    } -> tensor<1xf32>
    %3 = tensor.empty() : tensor<1xf32>
    %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %0, %2 : tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) outs(%3 : tensor<1xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %5 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %5 : f32
    } -> tensor<1xf32>
    return %4 : tensor<1xf32>
  }
  func.func private @_where.20(%arg0: tensor<1xi1>, %arg1: tensor<f32>, %arg2: tensor<1xi32>) -> tensor<1xf32> {
    %0 = tensor.expand_shape %arg1 [] : tensor<f32> into tensor<1xf32>
    %1 = tensor.empty() : tensor<1xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%arg2 : tensor<1xi32>) outs(%1 : tensor<1xf32>) {
    ^bb0(%arg3: i32, %arg4: f32):
      %5 = arith.sitofp %arg3 : i32 to f32
      linalg.yield %5 : f32
    } -> tensor<1xf32>
    %3 = tensor.empty() : tensor<1xf32>
    %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg0, %0, %2 : tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) outs(%3 : tensor<1xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %5 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %5 : f32
    } -> tensor<1xf32>
    return %4 : tensor<1xf32>
  }
  func.func @main() {
    %0 = arith.constant dense<[0.1, 0.2, -0.1, -0.2, 0.3, -0.3]> : tensor<6xf32>
    %1 = arith.constant dense<4> : tensor<i32>
    %2 = arith.constant dense<-2> : tensor<i32>
    %3 = func.call @callee(%0, %1, %2) : (tensor<6xf32>, tensor<i32>, tensor<i32>) -> tensor<1x6xf32>
    %unranked = tensor.cast %3 : tensor<1x6xf32> to tensor<*xf32>
    func.call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [1, 6] strides = [6, 1] data =
    //      CHECK: [0.65, 0.633333, 0.683333, 0.7, 0.616667, 0.716667]
    return
  }
}
