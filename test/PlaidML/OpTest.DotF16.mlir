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
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @dot_f16 {
  func.func @test(%arg0: tensor<8x16xf16>, %arg1: tensor<16x32xf16>) -> tensor<8x32xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<8x32xf16>
    %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<8x32xf16>) -> tensor<8x32xf16>
    %2 = linalg.generic {
            indexing_maps = [#map0, #map1, #map2],
            iterator_types = ["parallel", "parallel", "reduction"]
          }
          ins(%arg0, %arg1 : tensor<8x16xf16>, tensor<16x32xf16>)
          outs(%1 : tensor<8x32xf16>)
          attrs =  {iterator_ranges = [8, 32, 16]} {
            ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
              %3 = arith.mulf %arg2, %arg3 : f16
              %4 = arith.addf %arg4, %3 : f16
              linalg.yield %4 : f16
         } -> tensor<8x32xf16>
    return %2 : tensor<8x32xf16>
  }

  func.func @main() {
    %0 = arith.constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
                              ]> : tensor<8x16xf16>
    %1 = arith.constant dense<1.0> : tensor<16x32xf16>
    %2 = call @test(%0, %1) : (tensor<8x16xf16>, tensor<16x32xf16>) -> tensor<8x32xf16>
    %unranked = tensor.cast %2 : tensor<8x32xf16> to tensor<*xf16>
    call @printMemrefF16(%unranked) : (tensor<*xf16>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT: [136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136]
    return
  }

  func.func private @printMemrefF16(%ptr : tensor<*xf16>)
}
