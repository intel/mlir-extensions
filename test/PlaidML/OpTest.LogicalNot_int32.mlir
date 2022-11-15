// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-cpu.pp \
// RUN:                                       --runner mlir-cpu-runner -e main \
// RUN:                                       --shared-libs=%mlir_runner_utils \
// RUN:                                       --entry-point-result=void | FileCheck %s
// RUN: %python_executable %imex_runner -i %s --pass-pipeline-file=%p/linalg-to-llvm.pp \
// RUN:                                       --runner mlir-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%mlir_runner_utils,%sycl_runtime | FileCheck %s
#map = affine_map<(d0, d1) -> (d0, d1)>
module @logical_not {
func.func @main() {
    %0= arith.constant dense<[[1, 2, 3], [4, 0, 6], [7, 0, 9]]>:tensor<3x3xi32>
    %1 = call @test(%0) : (tensor<3x3xi32>) -> tensor<3x3xi1>
    %unranked = tensor.cast %1 : tensor<3x3xi1>to tensor<*xi1>
    call @printMemrefI32(%unranked) : (tensor<*xi1>) -> ()
    return
}
func.func private @printMemrefI32(tensor<*xi1>)
func.func @test(%arg0: tensor<3x3xi32>)->tensor<3x3xi1>{
    %0 = tensor.empty() : tensor<3x3xi1>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<3x3xi32>) outs(%0 : tensor<3x3xi1>) {
    ^bb0(%arg1: i32, %arg2: i1):
      %c0_i32 = arith.constant 0 : i32
      %2 = arith.cmpi eq, %arg1, %c0_i32 : i32
      linalg.yield %2 : i1
    } -> tensor<3x3xi1>
    return %1 : tensor<3x3xi1>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [3, 3] strides = {{.*}} data =
// CHECK:   0
// CHECK:   0
// CHECK:   0
// CHECK:   0
// CHECK:   1
// CHECK:   0
// CHECK:   0
// CHECK:   1
// CHECK:   0
