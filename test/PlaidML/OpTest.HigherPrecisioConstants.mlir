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
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
module @higher_precision_constants {
func.func @main() {
    %0 = arith.constant dense<0.5> : tensor<3x3xf32>
    %1 = call @test(%0) : (tensor<3x3xf32>) -> tensor<3x3xf64>
    %unranked = tensor.cast %1 : tensor<3x3xf64> to tensor<*xf64>
    call @printMemrefF64(%unranked) : (tensor<*xf64>) -> ()
    // CHECK:      [3.5,   3.5,   3.5]
    // CHECK-NEXT: [3.5,   3.5,   3.5]
    // CHECK-NEXT: [3.5,   3.5,   3.5]
    return
  }
func.func private @printMemrefF64(tensor<*xf64>)
func.func @test(%arg0: tensor<3x3xf32>) -> tensor<3x3xf64> {
    %cst = arith.constant 2.000000e+00 : f64
    %c1_i64 = arith.constant 1 : i64
    %0 = tensor.empty() : tensor<3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %c1_i64 : tensor<3x3xf32>, i64) outs(%0 : tensor<3x3xf32>) {
    ^bb0(%arg1: f32, %arg2: i64, %arg3: f32):
      %4 = arith.uitofp %arg2 : i64 to f32
      %5 = arith.addf %arg1, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<3x3xf32>
    %2 = tensor.empty() : tensor<3x3xf64>
    %3 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%1, %cst : tensor<3x3xf32>, f64) outs(%2 : tensor<3x3xf64>) {
    ^bb0(%arg1: f32, %arg2: f64, %arg3: f64):
      %4 = arith.extf %arg1 : f32 to f64
      %5 = arith.addf %4, %arg2 : f64
      linalg.yield %5 : f64
    } -> tensor<3x3xf64>
    return %3 : tensor<3x3xf64>
  }
}
