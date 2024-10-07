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
#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module @transpose {
  func.func @test(%arg0: tensor<10x20xbf16>) -> tensor<20x10xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<20x10xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<20x10xbf16>) -> tensor<20x10xbf16>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<10x20xbf16>) outs(%1 : tensor<20x10xbf16>) attrs =  {iterator_ranges = [20, 10], name = "transpose"} {
    ^bb0(%arg1: bf16, %arg2: bf16):
      linalg.yield %arg1 : bf16
    } -> tensor<20x10xbf16>
    return %2 : tensor<20x10xbf16>
  }
  func.func @main() {
    %0 = arith.constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
                              ]> : tensor<10x20xbf16>
    %2 = call @test(%0) : (tensor<10x20xbf16>) -> tensor<20x10xbf16>
    %unranked = tensor.cast %2 : tensor<20x10xbf16> to tensor<*xbf16>
    call @printMemrefBF16(%unranked) : (tensor<*xbf16>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT:  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    // CHECK-NEXT:  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    // CHECK-NEXT:  [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    // CHECK-NEXT:  [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    // CHECK-NEXT:  [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    // CHECK-NEXT:  [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    // CHECK-NEXT:  [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    // CHECK-NEXT:  [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    // CHECK-NEXT:  [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    // CHECK-NEXT:  [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    // CHECK-NEXT:  [11, 11, 11, 11, 11, 11, 11, 11, 11, 11]
    // CHECK-NEXT:  [12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
    // CHECK-NEXT:  [13, 13, 13, 13, 13, 13, 13, 13, 13, 13]
    // CHECK-NEXT:  [14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
    // CHECK-NEXT:  [15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
    // CHECK-NEXT:  [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    // CHECK-NEXT:  [17, 17, 17, 17, 17, 17, 17, 17, 17, 17]
    // CHECK-NEXT:  [18, 18, 18, 18, 18, 18, 18, 18, 18, 18]
    // CHECK-NEXT:  [19, 19, 19, 19, 19, 19, 19, 19, 19, 19]
    // CHECK-NEXT:  [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    return
  }

  func.func private @printMemrefBF16(%ptr : tensor<*xbf16>)
}
