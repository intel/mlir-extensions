// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN:                                       --entry-point-result=void --filecheck
// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm-caching.pp \
// RUN:                                       --runner mlir-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/linalg-to-llvm-caching.pp \
// RUN:                                        --runner mlir-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
#map = affine_map<(d0, d1) -> (d0, d1)>
module @add {
func.func @main() {
    %0= arith.constant dense<[[1, 2, 3], [4, 5, 4102], [16777223, 4294967304, 1099511627785]]>:tensor<3x3xi64>
    %1= arith.constant dense<[[1, 4098, 3], [16777220, 5, 4294967302], [7, 1099511627784, 9]]>:tensor<3x3xi64>
    %2 = func.call @test(%0,%1) : (tensor<3x3xi64>,tensor<3x3xi64>) -> tensor<3x3xi64>
    %unranked = tensor.cast %2 : tensor<3x3xi64> to tensor<*xi64>
    func.call @printMemrefI64(%unranked) : (tensor<*xi64>) -> ()
    return
}
func.func private @printMemrefI64(tensor<*xi64>)
func.func @test(%arg0: tensor<3x3xi64>, %arg1: tensor<3x3xi64>)->tensor<3x3xi64>{
    %0 = tensor.empty() : tensor<3x3xi64>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<3x3xi64>, tensor<3x3xi64>) outs(%0 : tensor<3x3xi64>) {
    ^bb0(%arg2: i64, %arg3: i64, %arg4: i64):
      %2 = arith.addi %arg2, %arg3 : i64
      linalg.yield %2 : i64
    } -> tensor<3x3xi64>
    return %1 : tensor<3x3xi64>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [3, 3] strides = {{.*}} data =
// CHECK:   2
// CHECK:   4100
// CHECK:   6
// CHECK:   16777224
// CHECK:   10
// CHECK:   4294971404
// CHECK:   16777230
// CHECK:   1103806595088
// CHECK:   1099511627794
