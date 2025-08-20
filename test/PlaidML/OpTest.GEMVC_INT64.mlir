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
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0) -> ()>
module @gemvc {
func.func @main() {
    %0= arith.constant dense<[[1, 1, 1], [1, 1, 1], [1, 1, 1]]>:tensor<3x3xi64>
    %1= arith.constant dense<[1, 1, 1]>:tensor<3xi64>
    %2= arith.constant dense<[1, 1, 1]>:tensor<3xi64>
    %3 = call @test(%0,%1,%2) : (tensor<3x3xi64>,tensor<3xi64>,tensor<3xi64>) -> tensor<3xi64>
    %unranked = tensor.cast %3 : tensor<3xi64>to tensor<*xi64>
    call @printMemrefI64(%unranked) : (tensor<*xi64>) -> ()
    // CHECK: [19,  19,  19]
    return
}
func.func private @printMemrefI64(tensor<*xi64>)
func.func @test(%arg0: tensor<3x3xi64>, %arg1: tensor<3xi64>, %arg2: tensor<3xi64>)->tensor<3xi64>{
    %c0_i64 = arith.constant 0 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %0 = tensor.empty() : tensor<3x3xi64>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %c5_i64 : tensor<3x3xi64>, i64) outs(%0 : tensor<3x3xi64>) {
    ^bb0(%arg3: i64, %arg4: i64, %arg5: i64):
      %9 = arith.muli %arg3, %arg4 : i64
      linalg.yield %9 : i64
    } -> tensor<3x3xi64>
    %2 = tensor.empty() : tensor<3xi64>
    %3 = linalg.fill ins(%c0_i64 : i64) outs(%2 : tensor<3xi64>) -> tensor<3xi64>
    %4 = linalg.generic {indexing_maps = [#map0, #map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%1, %arg1 : tensor<3x3xi64>, tensor<3xi64>) outs(%3 : tensor<3xi64>) attrs =  {iterator_ranges = [3, 3]} {
    ^bb0(%arg3: i64, %arg4: i64, %arg5: i64):
      %9 = arith.muli %arg3, %arg4 : i64
      %10 = arith.addi %arg5, %9 : i64
      linalg.yield %10 : i64
    } -> tensor<3xi64>
    %5 = tensor.empty() : tensor<3xi64>
    %6 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel"]} ins(%arg2, %c4_i64 : tensor<3xi64>, i64) outs(%5 : tensor<3xi64>) {
    ^bb0(%arg3: i64, %arg4: i64, %arg5: i64):
      %9 = arith.muli %arg3, %arg4 : i64
      linalg.yield %9 : i64
    } -> tensor<3xi64>
    %7 = tensor.empty() : tensor<3xi64>
    %8 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%4, %6 : tensor<3xi64>, tensor<3xi64>) outs(%7 : tensor<3xi64>) {
    ^bb0(%arg3: i64, %arg4: i64, %arg5: i64):
      %9 = arith.addi %arg3, %arg4 : i64
      linalg.yield %9 : i64
    } -> tensor<3xi64>
    return %8 : tensor<3xi64>
  }
}
