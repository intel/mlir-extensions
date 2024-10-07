// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils \
// RUN:                                       --entry-point-result=void --filecheck
// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module @broadcast_non_numpy {
  func.func @test(%arg0: tensor<3xbf16>) -> tensor<3x4xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<3x4xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<3x4xbf16>) -> tensor<3x4xbf16>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<3xbf16>) outs(%1 : tensor<3x4xbf16>) attrs =  {iterator_ranges = [3, 4], name = "broadcast"} {
    ^bb0(%arg1: bf16, %arg2: bf16):
      linalg.yield %arg1 : bf16
    } -> tensor<3x4xbf16>
    return %2 : tensor<3x4xbf16>
  }
  func.func @main() {
    %0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xbf16>
    %2 = call @test(%0) : (tensor<3xbf16>) -> tensor<3x4xbf16>
    %unranked = tensor.cast %2 : tensor<3x4xbf16> to tensor<*xbf16>
    call @printMemrefBF16(%unranked) : (tensor<*xbf16>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT: [1, 1, 1, 1]
    // CHECK-NEXT: [2, 2, 2, 2]
    // CHECK-NEXT: [3, 3, 3, 3]
    return
  }

  func.func private @printMemrefBF16(%ptr : tensor<*xbf16>)
}
