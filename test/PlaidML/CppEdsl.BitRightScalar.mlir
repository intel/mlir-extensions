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
module @bit_right_scalar {
func.func @main() {
    %0= arith.constant dense<[[1024, 4096, 12288], [32768, 81920, 196608], [458752, 1048576, 2359296]]>:tensor<3x3xi64>
    %1 = call @test(%0) : (tensor<3x3xi64>) -> tensor<3x3xi64>
    %unranked = tensor.cast %1 : tensor<3x3xi64>to tensor<*xi64>
    call @printMemrefI64(%unranked) : (tensor<*xi64>) -> ()
    return
}
func.func private @printMemrefI64(tensor<*xi64>)
func.func @test(%arg0: tensor<3x3xi64>)->tensor<3x3xi64>{
    %c9_i64 = arith.constant 9 : i64
    %0 = tensor.empty() : tensor<3x3xi64>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %c9_i64 : tensor<3x3xi64>, i64) outs(%0 : tensor<3x3xi64>) {
    ^bb0(%arg1: i64, %arg2: i64, %arg3: i64):
      %2 = arith.shrui %arg1, %arg2 : i64
      linalg.yield %2 : i64
    } -> tensor<3x3xi64>
    return %1 : tensor<3x3xi64>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [3, 3] strides = {{.*}} data =
// CHECK:   2
// CHECK:   8
// CHECK:   24
// CHECK:   64
// CHECK:   160
// CHECK:   384
// CHECK:   896
// CHECK:   2048
// CHECK:   4608
