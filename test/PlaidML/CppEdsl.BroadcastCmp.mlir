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
#map1 = affine_map<(d0, d1) -> (d0, 0)>
module @broadcast_cmp {
func.func @main() {
    %0= arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]>:tensor<3x4xi64>
    %1= arith.constant dense<[[0], [6], [12]]>:tensor<3x1xi64>
    %2 = call @test(%0,%1) : (tensor<3x4xi64>,tensor<3x1xi64>) -> tensor<3x4xi64>
    %unranked = tensor.cast %2 : tensor<3x4xi64>to tensor<*xi64>
    call @printMemrefI64(%unranked) : (tensor<*xi64>) -> ()
    return
}
func.func private @printMemrefI64(tensor<*xi64>)
func.func @test(%arg0: tensor<3x4xi64>, %arg1: tensor<3x1xi64>)->tensor<3x4xi64>{
    %0 = tensor.empty() : tensor<3x4xi1>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<3x4xi64>, tensor<3x1xi64>) outs(%0 : tensor<3x4xi1>) {
    ^bb0(%arg2: i64, %arg3: i64, %arg4: i1):
      %4 = arith.cmpi uge, %arg2, %arg3 : i64
      linalg.yield %4 : i1
    } -> tensor<3x4xi1>
    %2 = tensor.empty() : tensor<3x4xi64>
    %3 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<3x4xi1>) outs(%2 : tensor<3x4xi64>) {
    ^bb0(%arg2: i1, %arg3: i64):
      %4 = arith.extui %arg2 : i1 to i64
      linalg.yield %4 : i64
    } -> tensor<3x4xi64>
    return %3 : tensor<3x4xi64>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [3, 4] strides = {{.*}} data =
// CHECK:   1
// CHECK:   1
// CHECK:   1
// CHECK:   1
// CHECK:   0
// CHECK:   0
// CHECK:   1
// CHECK:   1
// CHECK:   0
// CHECK:   0
// CHECK:   0
// CHECK:   0
