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
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module @quantize {
func.func @test(%arg0: tensor<3xf32>)->tensor<3xi32>{
    %cst = arith.constant 2.560000e+02 : f32
    %0 = tensor.empty() : tensor<3xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel"]} ins(%arg0, %cst : tensor<3xf32>, f32) outs(%0 : tensor<3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %4 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %4 : f32
    } -> tensor<3xf32>
    %2 = tensor.empty() : tensor<3xi32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%1 : tensor<3xf32>) outs(%2 : tensor<3xi32>) {
    ^bb0(%arg1: f32, %arg2: i32):
      %4 = arith.fptosi %arg1 : f32 to i32
      linalg.yield %4 : i32
    } -> tensor<3xi32>
    return %3 : tensor<3xi32>
  }

  func.func @main() {
    %0= arith.constant dense<[0.1, 0.4, 0.3]>:tensor<3xf32>
    %1 = call @test(%0) : (tensor<3xf32>) -> tensor<3xi32>
    %unranked = tensor.cast %1 : tensor<3xi32>to tensor<*xi32>
    call @printMemrefI32(%unranked) : (tensor<*xi32>) -> ()
    return
  }

  func.func private @printMemrefI32(%ptr: tensor<*xi32>)
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [3] strides = {{.*}} data =
// CHECK:   25
// CHECK:   102
// CHECK:   76
