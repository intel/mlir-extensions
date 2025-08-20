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
#map0 = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<() -> ()>
module @jit__get_lgt_birth.7 {

  func.func private @printMemrefF32(tensor<*xf32>)

  func.func private @callee(%arg0: tensor<f32>, %arg1: tensor<95xf32>) -> tensor<95xf32> {
    %0 = tensor.empty() : tensor<95xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel"]} ins(%arg0 : tensor<f32>) outs(%0 : tensor<95xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<95xf32>
    %cst = arith.constant dense<1.000000e+01> : tensor<f32>
    %cst_0 = arith.constant dense<1.000000e+01> : tensor<95xf32>
    %2 = tensor.empty() : tensor<95xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%cst_0, %arg1 : tensor<95xf32>, tensor<95xf32>) outs(%2 : tensor<95xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %17 = math.powf %arg2, %arg3 : f32
      linalg.yield %17 : f32
    } -> tensor<95xf32>
    %4 = tensor.empty() : tensor<95xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%1, %3 : tensor<95xf32>, tensor<95xf32>) outs(%4 : tensor<95xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %17 = arith.subf %arg2, %arg3 : f32
      linalg.yield %17 : f32
    } -> tensor<95xf32>
    %cst_1 = arith.constant dense<1.000000e-03> : tensor<f32>
    %cst_2 = arith.constant dense<1.000000e-03> : tensor<95xf32>
    %6 = tensor.empty() : tensor<95xi1>
    %7 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%5, %cst_2 : tensor<95xf32>, tensor<95xf32>) outs(%6 : tensor<95xi1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):
      %17 = arith.cmpf olt, %arg2, %arg3 : f32
      linalg.yield %17 : i1
    } -> tensor<95xi1>
    %cst_3 = arith.constant dense<1.000000e-03> : tensor<f32>
    %8 = call @_where.13(%7, %cst_3, %5) : (tensor<95xi1>, tensor<f32>, tensor<95xf32>) -> tensor<95xf32>
    %9 = tensor.empty() : tensor<95xf32>
    %10 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%8 : tensor<95xf32>) outs(%9 : tensor<95xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %17 = math.log %arg2 : f32
      linalg.yield %17 : f32
    } -> tensor<95xf32>
    %cst_4 = arith.constant dense<1.000000e+01> : tensor<f32>
    %11 = tensor.empty() : tensor<f32>
    %12 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%cst_4 : tensor<f32>) outs(%11 : tensor<f32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %17 = math.log %arg2 : f32
      linalg.yield %17 : f32
    } -> tensor<f32>
    %13 = tensor.empty() : tensor<95xf32>
    %14 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel"]} ins(%12 : tensor<f32>) outs(%13 : tensor<95xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<95xf32>
    %15 = tensor.empty() : tensor<95xf32>
    %16 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%10, %14 : tensor<95xf32>, tensor<95xf32>) outs(%15 : tensor<95xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %17 = arith.divf %arg2, %arg3 : f32
      linalg.yield %17 : f32
    } -> tensor<95xf32>
    return %16 : tensor<95xf32>
  }
  func.func private @_where.13(%arg0: tensor<95xi1>, %arg1: tensor<f32>, %arg2: tensor<95xf32>) -> tensor<95xf32> {
    %0 = tensor.empty() : tensor<95xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel"]} ins(%arg1 : tensor<f32>) outs(%0 : tensor<95xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<95xf32>
    %2 = tensor.empty() : tensor<95xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %1, %arg2 : tensor<95xi1>, tensor<95xf32>, tensor<95xf32>) outs(%2 : tensor<95xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %4 = arith.select %arg3, %arg4, %arg5 : f32
      linalg.yield %4 : f32
    } -> tensor<95xf32>
    return %3 : tensor<95xf32>
  }
  func.func @main() {
    %0 = arith.constant dense<1.0> : tensor<f32>
    %1 = arith.constant dense<0.8> : tensor<95xf32>
    %3 = func.call @callee(%0, %1) : (tensor<f32>, tensor<95xf32>) -> tensor<95xf32>
    %unranked = tensor.cast %3 : tensor<95xf32> to tensor<*xf32>
    func.call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    //         CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    //    CHECK-SAME: rank = 1 offset = 0 sizes = [95] strides = [1] data =
    //         CHECK: [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
    // CHECK-COUNT-8:  -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
    //         CHECK:  -3, -3, -3, -3, -3]
    return
  }
}
