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
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module @convolution {
func.func @test(%arg0: tensor<1x56x56x64xbf16>, %arg1: tensor<3x3x64x64xbf16>) -> tensor<1x56x56x64xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<1x56x56x64xbf16>
    %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x56x56x64xbf16>) outs(%0 : tensor<1x56x56x64xbf16>) {
    ^bb0(%arg2: bf16, %arg3: bf16):
      linalg.yield %arg2 : bf16
    } -> tensor<1x56x56x64xbf16>
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %2 = tensor.pad %1 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
      tensor.yield %cst_0 : bf16
    } : tensor<1x56x56x64xbf16> to tensor<1x58x58x64xbf16>
    %3 = tensor.empty() : tensor<1x56x56x64xbf16>
    %4 = linalg.fill ins(%cst : bf16) outs(%3 : tensor<1x56x56x64xbf16>) -> tensor<1x56x56x64xbf16>
    %5 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%2, %arg1 : tensor<1x58x58x64xbf16>, tensor<3x3x64x64xbf16>) outs(%4 : tensor<1x56x56x64xbf16>) attrs =  {iterator_ranges = [1, 56, 56, 64, 3, 3, 64]} {
    ^bb0(%arg2: bf16, %arg3: bf16, %arg4: bf16):
      %6 = arith.mulf %arg2, %arg3 : bf16
      %7 = arith.addf %arg4, %6 : bf16
      linalg.yield %7 : bf16
    } -> tensor<1x56x56x64xbf16>
    return %5 : tensor<1x56x56x64xbf16>
  }
  func.func @main() {
    %0 = arith.constant dense<1.0> : tensor<1x56x56x64xbf16>
    %1 = arith.constant dense<0.5> : tensor<3x3x64x64xbf16>
    %2 = call @test(%0, %1) : (tensor<1x56x56x64xbf16>, tensor<3x3x64x64xbf16>) -> tensor<1x56x56x64xbf16>
    %unranked = tensor.cast %2 : tensor<1x56x56x64xbf16> to tensor<*xbf16>
    call @printMemrefBF16(%unranked) : (tensor<*xbf16>) -> ()
    //      CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-NEXT: [128,     128,     128,     128,     128,     128,     128,     128,     128,     128,     128
    return
  }
  func.func private @printMemrefBF16(%ptr : tensor<*xbf16>)
}
