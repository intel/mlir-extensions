// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner mlir-cpu-runner -e main \
// RUN:                                       --shared-libs=%mlir_runner_utils \
// RUN:                                       --entry-point-result=void | FileCheck %s
// RUN-GPU: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN-GPU:                                       --runner mlir-cpu-runner -e main \
// RUN-GPU:                                       --entry-point-result=void \
// RUN-GPU:                                       --shared-libs=%mlir_runner_utils,%sycl_runtime | FileCheck %s
#map = affine_map<(d0, d1) -> (d0, d1)>
module @reshape_fold {
func.func @main() {
    %0= arith.constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]>:tensor<3x3xi32>
    %1 = call @test(%0) : (tensor<3x3xi32>) -> tensor<3x3xi32>
    %unranked = tensor.cast %1 : tensor<3x3xi32>to tensor<*xi32>
    call @printMemrefI32(%unranked) : (tensor<*xi32>) -> ()
    return
}
func.func private @printMemrefI32(tensor<*xi32>)
func.func @test(%arg0: tensor<3x3xi32>)->tensor<3x3xi32>{
    %0 = tensor.empty() : tensor<3x3xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<3x3xi32>) outs(%0 : tensor<3x3xi32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      linalg.yield %arg1 : i32
    } -> tensor<3x3xi32>
    return %1 : tensor<3x3xi32>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [3, 3] strides = {{.*}} data =
// CHECK:   1
// CHECK:   2
// CHECK:   3
// CHECK:   4
// CHECK:   5
// CHECK:   6
// CHECK:   7
// CHECK:   8
// CHECK:   9
