// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner mlir-cpu-runner -e main \
// RUN:                                       --shared-libs=%mlir_runner_utils \
// RUN:                                       --entry-point-result=void | FileCheck %s
// RUN-GPU: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN-GPU:                                        --runner mlir-cpu-runner -e main \
// RUN-GPU:                                        --entry-point-result=void \
// RUN-GPU:                                        --shared-libs=%mlir_runner_utils,%sycl_runtime | FileCheck %s
#map = affine_map<(d0, d1) -> (d0, d1)>
module @bit_or {
func.func @main() {
    %0= arith.constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]>:tensor<3x3xi64>
    %1= arith.constant dense<[[10, 11, 12], [13, 14, 15], [16, 17, 18]]>:tensor<3x3xi64>
    %2 = call @test(%0,%1) : (tensor<3x3xi64>,tensor<3x3xi64>) -> tensor<3x3xi64>
    %unranked = tensor.cast %2 : tensor<3x3xi64>to tensor<*xi64>
    call @printMemrefI64(%unranked) : (tensor<*xi64>) -> ()
    return
}
func.func private @printMemrefI64(tensor<*xi64>)
func.func @test(%arg0: tensor<3x3xi64>, %arg1: tensor<3x3xi64>)->tensor<3x3xi64>{
    %0 = tensor.empty() : tensor<3x3xi64>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<3x3xi64>, tensor<3x3xi64>) outs(%0 : tensor<3x3xi64>) {
    ^bb0(%arg2: i64, %arg3: i64, %arg4: i64):
      %2 = arith.ori %arg2, %arg3 : i64
      linalg.yield %2 : i64
    } -> tensor<3x3xi64>
    return %1 : tensor<3x3xi64>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [3, 3] strides = {{.*}} data =
// CHECK:   11
// CHECK:   11
// CHECK:   15
// CHECK:   13
// CHECK:   15
// CHECK:   15
// CHECK:   23
// CHECK:   25
// CHECK:   27
