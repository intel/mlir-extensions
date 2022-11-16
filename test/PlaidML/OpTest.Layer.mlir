// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                        --runner mlir-cpu-runner -e main \
// RUN:                                        --shared-libs=%mlir_runner_utils \
// RUN:                                        --entry-point-result=void | FileCheck %s
// RUN-GPU: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN-GPU:                                        --runner mlir-cpu-runner -e main \
// RUN-GPU:                                        --entry-point-result=void \
// RUN-GPU:                                        --shared-libs=%mlir_runner_utils,%sycl_runtime | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
module @relu {
func.func @main() {
    %0 = arith.constant dense<-1.5> : tensor<10x20xf32>
    %1 = call @test(%0) : (tensor<10x20xf32>) -> tensor<10x20xf32>
    %unranked = tensor.cast %1 : tensor<10x20xf32> to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    // CHECK: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    // CHECK-COUNT-8: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    // CHECK: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return
}
func.func private @printMemrefF32(tensor<*xf32>)
func.func @test(%arg0: tensor<10x20xf32>) -> tensor<10x20xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<10x20xi1>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst : tensor<10x20xf32>, f32) outs(%0 : tensor<10x20xi1>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i1):
      %4 = arith.cmpf olt, %arg1, %arg2 : f32
      linalg.yield %4 : i1
    } -> tensor<10x20xi1>
    %2 = tensor.empty() : tensor<10x20xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map1, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%1, %cst, %arg0 : tensor<10x20xi1>, f32, tensor<10x20xf32>) outs(%2 : tensor<10x20xf32>) {
    ^bb0(%arg1: i1, %arg2: f32, %arg3: f32, %arg4: f32):
      %4 = arith.select %arg1, %arg2, %arg3 : f32
      linalg.yield %4 : f32
    } -> tensor<10x20xf32>
    return %3 : tensor<10x20xf32>
  }
}
