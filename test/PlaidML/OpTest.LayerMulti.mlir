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
#map = affine_map<(d0, d1) -> (d0, d1)>
module @LayerMulti {
func.func @main() {
    %0 = arith.constant dense<0.5> : tensor<10x20xf32>
    %1 = arith.constant dense<1.5> : tensor<10x20xf32>
    %2 = arith.constant dense<2.0> : tensor<10x20xf32>
    %3 = call @test(%0, %1, %2) : (tensor<10x20xf32>, tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
    %unranked = tensor.cast %3 : tensor<10x20xf32> to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    // CHECK: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
    // CHECK-COUNT-8: 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
    // CHECK: 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    return
}
func.func private @printMemrefF32(tensor<*xf32>)
func.func @test(%arg0: tensor<10x20xf32>, %arg1: tensor<10x20xf32> {stdx.const}, %arg2: tensor<10x20xf32> {stdx.const}) -> tensor<10x20xf32> {
    %0 = tensor.empty() : tensor<10x20xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<10x20xf32>, tensor<10x20xf32>) outs(%0 : tensor<10x20xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %4 = arith.addf %arg3, %arg4 : f32
      linalg.yield %4 : f32
    } -> tensor<10x20xf32>
    %2 = tensor.empty() : tensor<10x20xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %arg2 : tensor<10x20xf32>, tensor<10x20xf32>) outs(%2 : tensor<10x20xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %4 = arith.addf %arg3, %arg4 : f32
      linalg.yield %4 : f32
    } -> tensor<10x20xf32>
    return %3 : tensor<10x20xf32>
  }
}
