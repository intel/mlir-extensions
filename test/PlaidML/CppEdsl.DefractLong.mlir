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
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10) -> (d1, -d0 + d3 + d8 + 1, -d0 + d4 + d5 + d9, d10)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10) -> (d0 * 4 + d2 * 2 + d7 - d8 * 2, d0 * 4 + d2 - d4 + d7 - d9 * 2 + 3, d6, d10)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10) -> (d1, d2 + d3 * 2 + 1, d4 + d5 * 2, d6)>
#set = affine_set<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10) : (d2 + d4 + d7 + (d9 - 1) * 2 + 1 >= 0, -d4 - d5 * 2 + 4 >= 0)>
module @defract_long {
func.func @main(%arg0: tensor<1x3x3x1xf32>, %arg1: tensor<1x3x3x1xf32>) -> tensor<1x5x5x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x3x3x1xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x3x3x1xf32>) outs(%0 : tensor<1x3x3x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<1x3x3x1xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %2 = tensor.pad %1 low[0, 0, 1, 0] high[0, 0, 1, 0] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x3x3x1xf32> to tensor<1x3x5x1xf32>
    %3 = tensor.empty() : tensor<1x3x3x1xf32>
    %4 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x3x3x1xf32>) outs(%3 : tensor<1x3x3x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<1x3x3x1xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %5 = tensor.pad %4 low[0, 0, 0, 0] high[0, 1, 0, 0] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x3x3x1xf32> to tensor<1x4x3x1xf32>
    %6 = tensor.empty() : tensor<1x5x5x1xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<1x5x5x1xf32>) -> tensor<1x5x5x1xf32>
    //%8 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["reduction", "parallel", "window", "window", "window", "window",
    %8 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["reduction", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%2, %5 : tensor<1x3x5x1xf32>, tensor<1x4x3x1xf32>) outs(%7 : tensor<1x5x5x1xf32>) attrs =  {constraints = #set, iterator_ranges = [1, 1, 1, 2, 2, 3, 1, 1, 1, 2, 1]} {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %9 = arith.mulf %arg2, %arg3 : f32
      %10 = arith.addf %arg4, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<1x5x5x1xf32>
    return %8 : tensor<1x5x5x1xf32>
  }
}
