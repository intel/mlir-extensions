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
#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module @max_pool_1d {
func.func @main() {
    %0= arith.constant dense<[1, 2, 3]>:tensor<3xi64>
    %1 = call @test(%0) : (tensor<3xi64>) -> tensor<1xi64>
    %unranked = tensor.cast %1 : tensor<1xi64>to tensor<*xi64>
    call @printMemrefI64(%unranked) : (tensor<*xi64>) -> ()
    return
}
func.func private @printMemrefI64(tensor<*xi64>)
func.func @test(%arg0: tensor<3xi64>)->tensor<1xi64>{
    %c0_i64 = arith.constant 0 : i64
    %0 = tensor.empty() : tensor<1xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<3xi64>) outs(%1 : tensor<1xi64>) attrs = {iterator_ranges = [1, 3]} {
    ^bb0(%arg1: i64, %arg2: i64):
      %3 = arith.cmpi ugt, %arg2, %arg1 : i64
      %4 = arith.select %3, %arg2, %arg1 : i64
      linalg.yield %4 : i64
    } -> tensor<1xi64>
    return %2 : tensor<1xi64>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [1] strides = {{.*}} data =
// CHECK:   3
