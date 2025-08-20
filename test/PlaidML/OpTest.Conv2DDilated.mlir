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
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 * 2, d2 + d5 * 3, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module @conv_2d_dilated {
func.func @main() {
    %0 = arith.constant dense<1> : tensor<1x244x244x1xi64>
    %1 = arith.constant dense<1> : tensor<3x3x1x32xi64>
    %2 = call @test(%0, %1) : (tensor<1x244x244x1xi64>, tensor<3x3x1x32xi64>) -> tensor<1x240x238x32xi64>
    %unranked = tensor.cast %2 : tensor<1x240x238x32xi64> to tensor<*xi64>
    call @printMemrefI64(%unranked) : (tensor<*xi64>) -> ()
    // CHECK: [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    return
  }
func.func private @printMemrefI64(tensor<*xi64>)
func.func @test(%arg0: tensor<1x244x244x1xi64>, %arg1: tensor<3x3x1x32xi64>) -> tensor<1x240x238x32xi64> {
    %c0_i64 = arith.constant 0 : i64
    %0 = tensor.empty() : tensor<1x240x238x32xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1x240x238x32xi64>) -> tensor<1x240x238x32xi64>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x244x244x1xi64>, tensor<3x3x1x32xi64>) outs(%1 : tensor<1x240x238x32xi64>) attrs =  {iterator_ranges = [1, 240, 238, 32, 3, 3, 1]} {
    ^bb0(%arg2: i64, %arg3: i64, %arg4: i64):
      %3 = arith.muli %arg2, %arg3 : i64
      %4 = arith.addi %arg4, %3 : i64
      linalg.yield %4 : i64
    } -> tensor<1x240x238x32xi64>
    return %2 : tensor<1x240x238x32xi64>
  }
}
