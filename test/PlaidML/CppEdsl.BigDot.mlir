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
#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @dot {
  func.func @main() {
    %0= arith.constant dense<1.0>:tensor<2048x2048xf32>
    %1= arith.constant dense<2.0>:tensor<2048x2048xf32>
    %2 = call @test(%0,%1) : (tensor<2048x2048xf32>,tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    %unranked = tensor.cast %2 : tensor<2048x2048xf32>to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @test(%arg0: tensor<2048x2048xf32>, %arg1: tensor<2048x2048xf32>) -> tensor<2048x2048xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2048x2048xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<2048x2048xf32>, tensor<2048x2048xf32>) outs(%1 : tensor<2048x2048xf32>) attrs =  {iterator_ranges = [2048, 2048, 2048]} {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %3 = arith.mulf %arg2, %arg3 : f32
      %4 = arith.addf %arg4, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<2048x2048xf32>
    return %2 : tensor<2048x2048xf32>
  }
}
// CHECK: 4096,   4096,   4096,   4096,   4096,   4096,   4096,   4096,   4096,   4096,   4096,   4096,   4096,   4096,   4096,   4096,
