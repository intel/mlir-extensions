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
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module @broadcast_non_numpy {
  func.func @test(%arg0: tensor<3xf32>) -> tensor<3x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<3x4xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3x4xf32>) -> tensor<3x4xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<3xf32>) outs(%1 : tensor<3x4xf32>) attrs =  {iterator_ranges = [3, 4], name = "broadcast"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<3x4xf32>
    return %2 : tensor<3x4xf32>
  }
  func.func @main() {
    %0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %2 = call @test(%0) : (tensor<3xf32>) -> tensor<3x4xf32>
    %unranked = tensor.cast %2 : tensor<3x4xf32> to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT: [1, 1, 1, 1]
    // CHECK-NEXT: [2, 2, 2, 2]
    // CHECK-NEXT: [3, 3, 3, 3]
    return
  }

  func.func private @printMemrefF32(%ptr : tensor<*xf32>)
}
