// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN:                                       --entry-point-result=void --filecheck
// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=opencl-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%opencl_runtime --filecheck
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0) -> (d0)>
module @gemv {
func.func @main() {
    %0= arith.constant dense<[[1, 1, 1], [1, 1, 1], [1, 1, 1]]>:tensor<3x3xi64>
    %1= arith.constant dense<[1, 1, 1]>:tensor<3xi64>
    %2= arith.constant dense<[1, 1, 1]>:tensor<3xi64>
    %3 = call @test(%0,%1,%2) : (tensor<3x3xi64>,tensor<3xi64>,tensor<3xi64>) -> tensor<3xi64>
    %unranked = tensor.cast %3 : tensor<3xi64>to tensor<*xi64>
    call @printMemrefI64(%unranked) : (tensor<*xi64>) -> ()
    // CHECK: [4,  4,  4]
    return
}
func.func private @printMemrefI64(tensor<*xi64>)
func.func @test(%arg0: tensor<3x3xi64>, %arg1: tensor<3xi64>, %arg2: tensor<3xi64>)->tensor<3xi64>{
    %c0_i64 = arith.constant 0 : i64
    %0 = tensor.empty() : tensor<3xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<3xi64>) -> tensor<3xi64>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x3xi64>, tensor<3xi64>) outs(%1 : tensor<3xi64>) attrs =  {iterator_ranges = [3, 3]} {
    ^bb0(%arg3: i64, %arg4: i64, %arg5: i64):
      %5 = arith.muli %arg3, %arg4 : i64
      %6 = arith.addi %arg5, %5 : i64
      linalg.yield %6 : i64
    } -> tensor<3xi64>
    %3 = tensor.empty() : tensor<3xi64>
    %4 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%2, %arg2 : tensor<3xi64>, tensor<3xi64>) outs(%3 : tensor<3xi64>) {
    ^bb0(%arg3: i64, %arg4: i64, %arg5: i64):
      %5 = arith.addi %arg3, %arg4 : i64
      linalg.yield %5 : i64
    } -> tensor<3xi64>
    return %4 : tensor<3xi64>
  }
}
