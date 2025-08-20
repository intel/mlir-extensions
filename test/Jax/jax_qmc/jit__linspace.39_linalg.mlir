// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN:                                       --entry-point-result=void --filecheck
// RUN-GPU: %python_executable %imex_runner --requires=l0-runtime,igpu-fp64 %igpu_fp64 -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN-GPU:                                       --runner mlir-runner -e main \
// RUN-GPU:                                       --entry-point-result=void \
// RUN-GPU:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN-GPU: %python_executable %imex_runner --requires=sycl-runtime,igpu-fp64 %igpu_fp64 -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN-GPU:                                        --runner mlir-runner -e main \
// RUN-GPU:                                        --entry-point-result=void \
// RUN-GPU:                                        --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0)>
module @jit__linspace.39 {

  func.func private @printMemrefF64(tensor<*xf64>)

  func.func private @callee(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<100xf64> {
    %0 = tensor.empty() : tensor<f64>
    %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = []} ins(%arg0 : tensor<i64>) outs(%0 : tensor<f64>) {
    ^bb0(%arg2: i64, %arg3: f64):
      %27 = arith.sitofp %arg2 : i64 to f64
      linalg.yield %27 : f64
    } -> tensor<f64>
    %2 = tensor.expand_shape %1 [] : tensor<f64> into tensor<1xf64>
    %3 = tensor.collapse_shape %2 [] : tensor<1xf64> into tensor<f64>
    %4 = tensor.empty() : tensor<99xf64>
    %5 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%3 : tensor<f64>) outs(%4 : tensor<99xf64>) {
    ^bb0(%arg2: f64, %arg3: f64):
      linalg.yield %arg2 : f64
    } -> tensor<99xf64>
    %cst = arith.constant dense<1.000000e+00> : tensor<f64>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<99xf64>
    %6 = tensor.empty() : tensor<99xf64>
    %7 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%6 : tensor<99xf64>) {
    ^bb0(%arg2: f64):
      %27 = linalg.index 0 : index
      %28 = arith.index_cast %27 : index to i64
      %29 = arith.sitofp %28 : i64 to f64
      linalg.yield %29 : f64
    } -> tensor<99xf64>
    %cst_1 = arith.constant dense<9.900000e+01> : tensor<f64>
    %cst_2 = arith.constant dense<9.900000e+01> : tensor<99xf64>
    %8 = tensor.empty() : tensor<99xf64>
    %9 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%7, %cst_2 : tensor<99xf64>, tensor<99xf64>) outs(%8 : tensor<99xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):
      %27 = arith.divf %arg2, %arg3 : f64
      linalg.yield %27 : f64
    } -> tensor<99xf64>
    %10 = tensor.empty() : tensor<99xf64>
    %11 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%cst_0, %9 : tensor<99xf64>, tensor<99xf64>) outs(%10 : tensor<99xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):
      %27 = arith.subf %arg2, %arg3 : f64
      linalg.yield %27 : f64
    } -> tensor<99xf64>
    %12 = tensor.empty() : tensor<99xf64>
    %13 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%5, %11 : tensor<99xf64>, tensor<99xf64>) outs(%12 : tensor<99xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):
      %27 = arith.mulf %arg2, %arg3 : f64
      linalg.yield %27 : f64
    } -> tensor<99xf64>
    %14 = tensor.empty() : tensor<f64>
    %15 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = []} ins(%arg1 : tensor<i64>) outs(%14 : tensor<f64>) {
    ^bb0(%arg2: i64, %arg3: f64):
      %27 = arith.sitofp %arg2 : i64 to f64
      linalg.yield %27 : f64
    } -> tensor<f64>
    %16 = tensor.expand_shape %15 [] : tensor<f64> into tensor<1xf64>
    %17 = tensor.collapse_shape %16 [] : tensor<1xf64> into tensor<f64>
    %18 = tensor.empty() : tensor<99xf64>
    %19 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%17 : tensor<f64>) outs(%18 : tensor<99xf64>) {
    ^bb0(%arg2: f64, %arg3: f64):
      linalg.yield %arg2 : f64
    } -> tensor<99xf64>
    %20 = tensor.empty() : tensor<99xf64>
    %21 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%19, %9 : tensor<99xf64>, tensor<99xf64>) outs(%20 : tensor<99xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):
      %27 = arith.mulf %arg2, %arg3 : f64
      linalg.yield %27 : f64
    } -> tensor<99xf64>
    %22 = tensor.empty() : tensor<99xf64>
    %23 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%13, %21 : tensor<99xf64>, tensor<99xf64>) outs(%22 : tensor<99xf64>) {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):
      %27 = arith.addf %arg2, %arg3 : f64
      linalg.yield %27 : f64
    } -> tensor<99xf64>
    %24 = tensor.expand_shape %15 [] : tensor<f64> into tensor<1xf64>
    %c0 = arith.constant 0 : index
    %25 = tensor.empty() : tensor<100xf64>
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
  func.func @main() {
    %0 = arith.constant dense<8> : tensor<i64>
    %1 = arith.constant dense<3> : tensor<i64>
    %3 = func.call @callee(%0, %1) : (tensor<i64>, tensor<i64>) -> tensor<100xf64>
    %unranked = tensor.cast %3 : tensor<100xf64> to tensor<*xf64>
    func.call @printMemrefF64(%unranked) : (tensor<*xf64>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [100] strides = [1] data =
    //      CHECK: [8, 7.94949, 7.89899, 7.84848, 7.79798,
    //      CHECK:  7.74747, 7.69697, 7.64646, 7.59596, 7.54545,
    //      CHECK:  7.49495, 7.44444, 7.39394, 7.34343, 7.29293,
    //      CHECK:  7.24242, 7.19192, 7.14141, 7.09091, 7.0404,
    //      CHECK:  6.9899, 6.93939, 6.88889, 6.83838, 6.78788,
    //      CHECK:  6.73737, 6.68687, 6.63636, 6.58586, 6.53535,
    //      CHECK:  6.48485, 6.43434, 6.38384, 6.33333, 6.28283,
    //      CHECK:  6.23232, 6.18182, 6.13131, 6.08081, 6.0303,
    //      CHECK:  5.9798, 5.92929, 5.87879, 5.82828, 5.77778,
    //      CHECK:  5.72727, 5.67677, 5.62626, 5.57576, 5.52525,
    //      CHECK:  5.47475, 5.42424, 5.37374, 5.32323, 5.27273,
    //      CHECK:  5.22222, 5.17172, 5.12121, 5.07071, 5.0202,
    //      CHECK:  4.9697, 4.91919, 4.86869, 4.81818, 4.76768,
    //      CHECK:  4.71717, 4.66667, 4.61616, 4.56566, 4.51515,
    //      CHECK:  4.46465, 4.41414, 4.36364, 4.31313, 4.26263,
    //      CHECK:  4.21212, 4.16162, 4.11111, 4.06061, 4.0101,
    //      CHECK:  3.9596, 3.90909, 3.85859, 3.80808, 3.75758,
    //      CHECK:  3.70707, 3.65657, 3.60606, 3.55556, 3.50505,
    //      CHECK:  3.45455, 3.40404, 3.35354, 3.30303, 3.25253,
    //      CHECK:  3.20202, 3.15152, 3.10101, 3.05051, 3]
    return
  }
}
