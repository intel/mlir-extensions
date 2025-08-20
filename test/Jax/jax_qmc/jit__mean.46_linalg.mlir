// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN:                                       --entry-point-result=void --filecheck
// RUN: %python_executable %imex_runner --requires=l0-runtime,igpu-fp64 %igpu_fp64 -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime,igpu-fp64 %igpu_fp64 -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
#map0 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module @jit__mean.46 {

  func.func private @printMemrefF64(tensor<*xf64>)

  func.func private @callee(%arg0: tensor<1000x16x3xf64>) -> tensor<1000x3xf64> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = tensor.empty() : tensor<1000x3xf64>
    %1 = linalg.fill ins(%cst_0 : f64) outs(%0 : tensor<1000x3xf64>) -> tensor<1000x3xf64>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1000x16x3xf64>) outs(%1 : tensor<1000x3xf64>) {
    ^bb0(%arg1: f64, %arg2: f64):
      %5 = arith.addf %arg1, %arg2 : f64
      linalg.yield %5 : f64
    } -> tensor<1000x3xf64>
    %cst_1 = arith.constant dense<1.600000e+01> : tensor<f64>
    %cst_2 = arith.constant dense<1.600000e+01> : tensor<1000x3xf64>
    %3 = tensor.empty() : tensor<1000x3xf64>
    %4 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%2, %cst_2 : tensor<1000x3xf64>, tensor<1000x3xf64>) outs(%3 : tensor<1000x3xf64>) {
    ^bb0(%arg1: f64, %arg2: f64, %arg3: f64):
      %5 = arith.divf %arg1, %arg2 : f64
      linalg.yield %5 : f64
    } -> tensor<1000x3xf64>
    return %4 : tensor<1000x3xf64>
  }
  func.func @main() {
    %0 = arith.constant dense<1.99> : tensor<1000x16x3xf64>
    %3 = func.call @callee(%0) : (tensor<1000x16x3xf64>) -> tensor<1000x3xf64>
    %unranked = tensor.cast %3 : tensor<1000x3xf64> to tensor<*xf64>
    func.call @printMemrefF64(%unranked) : (tensor<*xf64>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [1000, 3] strides = [3, 1] data =
    // CHECK-COUNT-1000: [1.99, 1.99, 1.99]
    return
  }
}
