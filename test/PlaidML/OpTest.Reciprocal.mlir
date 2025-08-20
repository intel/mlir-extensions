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
#map0 = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
module @reciprocal {
func.func @main() {
    %0= arith.constant dense<[1.0, 2.0, 4.0, 5.0, 8.0, 10.0]>:tensor<6xf32>
    %1 = call @test(%0) : (tensor<6xf32>) -> tensor<6xf32>
    %unranked = tensor.cast %1 : tensor<6xf32>to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
}
func.func private @printMemrefF32(tensor<*xf32>)
func.func @test(%arg0: tensor<6xf32>)->tensor<6xf32>{
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<6xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel"]} ins(%cst, %arg0 : f32, tensor<6xf32>) outs(%0 : tensor<6xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %2 = arith.divf %arg1, %arg2 : f32
      linalg.yield %2 : f32
    } -> tensor<6xf32>
    return %1 : tensor<6xf32>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [6] strides = {{.*}} data =
// CHECK:   1
// CHECK:   0.5
// CHECK:   0.25
// CHECK:   0.2
// CHECK:   0.125
// CHECK:   0.1
