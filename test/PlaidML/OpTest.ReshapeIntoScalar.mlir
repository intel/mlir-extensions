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
#map = affine_map<() -> ()>
module @reshape_into_scalar {
func.func @main() {
    %0= arith.constant dense<[[[2]]]>:tensor<1x1x1xi32>
    %1 = call @test(%0) : (tensor<1x1x1xi32>) -> tensor<i32>
    %unranked = tensor.cast %1 : tensor<i32>to tensor<*xi32>
    call @printMemrefI32(%unranked) : (tensor<*xi32>) -> ()
    return
}
func.func private @printMemrefI32(tensor<*xi32>)
func.func @test(%arg0: tensor<1x1x1xi32>)->tensor<i32>{
    %0 = tensor.collapse_shape %arg0 [] : tensor<1x1x1xi32> into tensor<i32>
    %1 = tensor.empty() : tensor<i32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%0 : tensor<i32>) outs(%1 : tensor<i32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      linalg.yield %arg1 : i32
    } -> tensor<i32>
    return %2 : tensor<i32>
  }
}
// CHECK: Unranked Memref base@ = {{0x[0-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = {{.*}} strides = {{.*}} data =
// CHECK:   2
